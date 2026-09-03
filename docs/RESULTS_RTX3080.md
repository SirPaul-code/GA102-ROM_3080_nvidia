# Measured results — RTX 3080 10 GB

Hardware / software reported by the test machine:

- GPU: NVIDIA GeForce RTX 3080 10 GB
- Compute capability: 8.6
- SMs: 68
- Memory bus: 320 bit
- Reported memory clock: 9.501 GHz
- Property bandwidth: 760.1 GB/s
- Driver: 595.79
- CUDA runtime: 13.3 (13030)
- CUDA Toolkit: 13.3

These are real measurements from the target GA102 machine, not projected numbers.

## V1 — initial mapping

Shape: M=8192, K=8192.

| Backend | Time | Logical throughput | Effective weight read |
|---|---:|---:|---:|
| I8 DP4A | 0.330 ms | 203.4 GMAC/s | 203.4 GB/s |
| W2 decode + DP4A | 0.347 ms | 193.7 GMAC/s | 48.4 GB/s |
| W1x2 CUDA POPC | 0.664 ms | 101.1 GMAC/s | 25.3 GB/s |
| W2 raw stream | 0.067 ms | — | 252.2 GB/s |

Raw SM86 BMMA probe:

- 1088 warps
- 4096 BMMA operations / warp
- 0.215 ms
- 680.70 TOP/s logical b1 AND contributions

## V2 — warp-cooperative mapping

Shape: M=8192, K=8192. Mapping: one warp per output row.

| Backend | Time | Logical throughput | Effective weight read | Speedup vs V1 same backend |
|---|---:|---:|---:|---:|
| I8 DP4A warp | 0.135 ms | 497.8 GMAC/s | 497.8 GB/s | 2.45x |
| W2 decode + DP4A warp | 0.063 ms | 1059.4 GMAC/s | 264.9 GB/s | 5.47x |
| W1x2 CUDA POPC warp | 0.044 ms | 1517.0 GMAC/s | 379.3 GB/s | 15.00x |

Vectorized 512 MiB DRAM read benchmark:

- 16-byte loads / thread (`uint4`)
- 0.965 ms
- 556.3 GB/s measured
- 73.2% of the 760.1 GB/s property peak

### V2 interpretation

The original kernels were primarily mapping / serialization limited. The important V2 change is not a different ternary approximation: all results remain exact integer dot products. The improvement came from assigning an entire warp to one output row and, for the POPC path, loading each positive/negative weight word once and reusing it across all eight activation bitplanes.

The fastest single-token exact path so far is therefore the two one-bit weight masks (`W==+1`, `W==-1`) plus eight signed-A8 activation bitplanes using ordinary CUDA integer `POPC`: **1.517 TMAC/s** logical ternary throughput on the 8192x8192 test.

## V3 — first exact BMMA mapping

Shape: M=8192, K=8192. BMMA tile: `m16n8k256.b1.b1.and.popc`.

| Backend in V3 binary | Time | Logical throughput | Reported one-pass W bytes |
|---|---:|---:|---:|
| W1x2 CUDA POPC warp | 0.345 ms | 194.4 GMAC/s | 48.6 GB/s |
| W1x2 BMMA TensorCore | 3.455 ms | 19.4 GMAC/s | 4.9 GB/s |

Additional BMMA counters:

- 262,144 BMMA instructions/run
- 2.49 TOP/s raw b1 contributions
- only 1 of BMMA's 8 N columns was useful; 7/8 were intentionally discarded
- exact CPU-reference correctness: PASS

### Critical interpretation of V3

The V3 number is **not** evidence that GA102 binary Tensor Cores are intrinsically slow. The raw V1 microarchitectural probe already showed 680.70 TOP/s when operands stayed in registers.

V3 exposed two mapping mistakes that are particularly important for a fixed-model runtime:

1. **Wrong weight memory layout for BMMA.** V3 kept weights row-major and each warp gathered its A fragments from rows separated by the full row stride. The accesses were exact but poorly coalesced.
2. **Wrong loop order for weight reuse.** V3 iterated activation bitplane outside K-chunk, so the same weight fragment was loaded again for every A8 bitplane. V2 POPC does the opposite: load the weight once, then apply all 8 bitplanes.

The V3 POPC baseline therefore also regressed from V2's 0.044 ms to 0.345 ms for the same reason. The fair best-known POPC baseline remains the V2 kernel.

## V4 — BMMA-native fixed weight layout

Shape: M=8192, K=8192. Fixed ternary weights are compiled offline into the exact `m16n8k256` A-fragment order (`uint4` per lane, contiguous per warp). Each 256-K positive/negative weight fragment is loaded once and reused across all eight A8 bitplanes.

### Single-token exact

| Backend | Time | Logical throughput | Physical W rate |
|---|---:|---:|---:|
| W1x2 CUDA POPC | 0.040 ms | 1674.9 GMAC/s | 418.7 GB/s |
| W1x2 BMMA packed | 0.041 ms | 1634.3 GMAC/s | 408.6 GB/s |

- BMMA/POPC speed ratio: **0.98x**
- exact CPU-reference correctness: PASS

This is a major recovery from V3: once the model weights are stored in the exact GA102-native fragment layout and weight reuse is correct, BMMA is essentially tied with the best single-token POPC path even though seven of its eight N columns are not useful.

### Eight-column exact

| Backend | Time | Logical throughput | Physical W rate |
|---|---:|---:|---:|
| W1x2 CUDA POPC x8 | 0.210 ms | 2555.6 GMAC/s | 79.9 GB/s |
| W1x2 BMMA packed N=8 | 0.033 ms | 16252.0 GMAC/s | 507.9 GB/s |

- BMMA/POPC speed ratio: **6.36x**
- BMMA instructions/run: 262,144
- raw b1 rate: 260.03 TOP/s
- exact CPU-reference correctness: PASS
- `0.033 / 8 = 0.0041 ms` is an amortized throughput-equivalent number, **not** single-token latency

A useful comparison is eight single-token POPC projections processed sequentially: roughly `8 * 0.040 = 0.320 ms`. The N=8 BMMA kernel performs the eight projections together in **0.033 ms**, about 9.7x less matrix time than that naive sequential path. The fairer weight-reuse software baseline is POPC x8 at 0.210 ms, against which BMMA is 6.36x faster.

### V4 interpretation

The project now has two distinct best execution modes:

1. **Single-token decode:** POPC and BMMA-native are effectively tied at about 1.6–1.7 TMAC/s for this shape. POPC is simpler; BMMA does not provide a meaningful single-token advantage yet.
2. **Multi-column / verification work:** BMMA becomes decisively better when all eight native N columns carry useful activation vectors. The fixed-model weight compiler is essential: the same arithmetic in the wrong row-major layout was unusably slow in V3.

V4 intentionally excluded dynamic B-fragment packing from the timed N=8 kernel. V5 therefore starts from raw signed INT8 `[8,K]` activations, packs them on the GPU with warp ballots into the exact BMMA B-fragment layout, and measures pack time, BMMA time, and the combined end-to-end pipeline separately.

## V5 — dynamic INT8 activation pack + BMMA N=8

Shape: M=8192, K=8192, N=8. Input activations are ordinary signed INT8 `[8,K]`. The fixed weights remain in the offline-compiled GA102 BMMA-native layout. A GPU packing kernel uses `__ballot_sync` to create the exact `uint2/lane` B fragments consumed by `m16n8k256`.

| Stage | Time |
|---|---:|
| GPU B-fragment pack | 0.0072 ms |
| BMMA matrix kernel | 0.0455 ms |
| Pack + BMMA pipeline | 0.0459 ms |

Pipeline result:

- exact CPU-reference correctness: PASS
- logical throughput: **11,696.3 GMAC/s (11.696 TMAC/s)**
- activation pack input: 64.0 KiB
- packed B output: 64.0 KiB
- isolated pack effective IO: 18.2 GB/s
- amortized `0.0459/8 = 0.0057 ms` is throughput-equivalent only, not one-token latency

### V5 interpretation and timing caveat

V5 removes the largest artificial advantage from V4: B fragments are no longer prepacked on the CPU. Starting from normal INT8 activation vectors, the exact GPU pipeline still sustains **11.7 TMAC/s** on the synthetic 8192x8192x8 workload.

The isolated timings are not additive in this run: `0.0072 + 0.0455` is larger than the measured sequential pipeline time `0.0459`. Inspection of the source confirms that the pipeline really launches the pack kernel followed by the BMMA kernel in the same default CUDA stream. The discrepancy is therefore attributed to run-state effects such as GPU boost/cache/timing-order variation, not a missing stage. Consequently, the measured **pipeline total and correctness are accepted**, while the printed `15.7% pack share` should not yet be treated as a precise stage decomposition. A hardened benchmark should measure stage boundaries under the same pipeline state and report repeated-run statistics.

## V6 — real BitNet b1.58 2B-4T projection shapes

V6 replaces the synthetic 8192x8192 matrix with the actual decoder projection dimensions of `microsoft/bitnet-b1.58-2B-4T`. The fixed runtime fuses Q/K/V into one `3840x2560` projection and gate/up into one `13824x2560` projection. To reduce L2-hot benchmark artifacts, each shape rotates through approximately 64 MiB of equivalent packed weight copies.

All single-token and N=8 results passed exact CPU-reference validation.

| Projection | Shape MxK | Packed W | Single pack+POPC | Single GMAC/s | N=8 pack+BMMA | N=8 GMAC/s |
|---|---:|---:|---:|---:|---:|---:|
| fused QKV | 3840x2560 | 2.344 MiB | 0.0170 ms | 576.7 | 0.0171 ms | 4600.2 |
| attention O | 2560x2560 | 1.562 MiB | 0.0136 ms | 481.7 | 0.0181 ms | 2890.2 |
| fused gate+up | 13824x2560 | 8.438 MiB | 0.0234 ms | 1513.2 | 0.0197 ms | 14392.5 |
| MLP down | 2560x6912 | 4.219 MiB | 0.0168 ms | 1050.2 | 0.0247 ms | 5738.5 |

Matrix-only physical weight-read observations:

- fused QKV single: 180.3 GB/s
- attention O single: 135.9 GB/s
- fused gate+up single: 424.4 GB/s
- MLP down single: 390.8 GB/s
- N=8 matrix paths vary from 156.9 to 348.7 GB/s for these smaller real shapes

Aggregate decoder-linears result:

- packed ternary W per layer: **16.562 MiB**
- packed ternary W for 30 layers: **496.875 MiB**
- logical ternary MAC/token: **2,084,044,800**
- single-token linear time/layer: **0.0709 ms**
- single-token 30-layer linears: **2.1265 ms**
- single-token linear-only ceiling: **470.3 token/s**
- N=8 linears/layer: **0.0796 ms for 8 states**
- N=8 30-layer linears: **2.3873 ms for 8 states**
- N=8 linear-only throughput: **3351.1 states/s**

### V6 interpretation

V6 is the first benchmark that maps the custom kernels onto the actual model's decoder shapes. The large MLP projections are already relatively close to the memory-throughput regime, while QKV/O are small enough that launch, occupancy and fixed overhead matter much more than the raw 760 GB/s property bandwidth.

The **470.3 token/s** number is explicitly a 30-layer ternary-linear ceiling, not end-to-end model throughput. V6 starts from raw signed A8 inputs and includes the dynamic bitplane/B-fragment packing used by the matrix kernels, but it still excludes BF16-to-A8 activation quantization, RMSNorm/subnorms, RoPE, KV-cache attention, ReLU2/gating, residuals, the final norm, tied LM head, sampling and runtime dependencies.

Similarly, **3351.1 states/s** is N=8 linear verification throughput. It must not be interpreted as 3351 autoregressive generated tokens/s.
