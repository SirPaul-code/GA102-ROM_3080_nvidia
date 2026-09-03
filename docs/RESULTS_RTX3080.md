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

### V4 interpretation

Once the model weights are stored in the exact GA102-native fragment layout and weight reuse is correct, BMMA is essentially tied with the best single-token POPC path even though seven of its eight N columns are not useful. When all eight N columns carry useful activation states, BMMA becomes decisively better.

## V5 — dynamic INT8 activation pack + BMMA N=8

Shape: M=8192, K=8192, N=8. Input activations are ordinary signed INT8 `[8,K]`. A GPU packing kernel uses `__ballot_sync` to create the exact `uint2/lane` B fragments consumed by `m16n8k256`.

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

The isolated timings are not additive in this run; the measured pipeline total and correctness are accepted, while the isolated pack-share percentage is not treated as an exact decomposition.

## V6 — real BitNet b1.58 2B-4T projection shapes

V6 replaces the synthetic 8192x8192 matrix with the actual decoder projection dimensions of `microsoft/bitnet-b1.58-2B-4T`. The fixed runtime fuses Q/K/V into one `3840x2560` projection and gate/up into one `13824x2560` projection. Each shape rotates through approximately 64 MiB of equivalent packed weight copies to reduce permanently-L2-hot artifacts.

All single-token and N=8 results passed exact CPU-reference validation.

| Projection | Shape MxK | Packed W | Single pack+POPC | Single GMAC/s | N=8 pack+BMMA | N=8 GMAC/s |
|---|---:|---:|---:|---:|---:|---:|
| fused QKV | 3840x2560 | 2.344 MiB | 0.0170 ms | 576.7 | 0.0171 ms | 4600.2 |
| attention O | 2560x2560 | 1.562 MiB | 0.0136 ms | 481.7 | 0.0181 ms | 2890.2 |
| fused gate+up | 13824x2560 | 8.438 MiB | 0.0234 ms | 1513.2 | 0.0197 ms | 14392.5 |
| MLP down | 2560x6912 | 4.219 MiB | 0.0168 ms | 1050.2 | 0.0247 ms | 5738.5 |

Aggregate decoder-linears result:

- packed ternary W per layer: **16.562 MiB**
- packed ternary W for 30 layers: **496.875 MiB**
- logical ternary MAC/token: **2,084,044,800**
- single-token linear time/layer: **0.0709 ms**
- single-token 30-layer linears: **2.1265 ms**
- single-token linear-only ceiling: **470.3 token/s**
- N=8 30-layer linears: **2.3873 ms for 8 states**
- N=8 linear-only throughput: **3351.1 states/s**

These are linear-only ceilings, not full-model generation rates.

## V7 — remaining decode primitives and BF16 LM head

V7 measures the non-ternary decode pieces separately on the same RTX 3080.

Per-token primitive timings included RMSNorm, BF16->A8 quantization, RoPE, ReLU2/gating, residuals and projection postscales. Kept as separate launches, those helpers summed to **0.1167 ms/layer**, or **3.5100 ms across 30 layers**.

The tied BF16 LM head (`128256 x 2560`, 626.250 MiB) achieved:

- correctness: PASS (sampled rows vs CPU BF16 reference)
- time: **0.9166 ms**
- effective weight-read rate: **716.4 GB/s**
- logical throughput: 358.2 GMAC/s

V7 also measured a K+V read-traffic floor across 30 layers. At context 4096 the traffic-only floor was 0.5713 ms at 550.7 GB/s, proving that full attention arithmetic rather than raw KV bytes still needed to be measured.

## V8 — fused support kernels + exact GQA decode attention

V8 combines helper stages that are naturally adjacent in the fixed model and adds exact QK scaling, softmax and weighted-V attention.

Fused support:

| Stage | Time/layer |
|---|---:|
| input RMSNorm + A8 quant | 0.0084 ms |
| QKV postscale + RoPE + KV append | 0.0061 ms |
| attn subnorm + O-input A8 quant | 0.0138 ms |
| O postscale + residual + norm + gate A8 | 0.0096 ms |
| gate/up + ReLU2 + FFN norm + down A8 | 0.0179 ms |
| down postscale + residual | 0.0066 ms |

Total fused support: **0.0624 ms/layer = 1.8712 ms/30L**, a **1.8758x** improvement over V7's launch-separated 3.5100 ms.

Exact attention correctness: PASS (`head0@128`, max abs error 0.000030; two GPU attention variants agreed exactly at BF16 output resolution).

| Context | Best V8 attention / layer | 30-layer attention | GPU accounting ceiling |
|---:|---:|---:|---:|
| 128 | 0.0121 ms | 0.3619 ms | 189.2 tok/s |
| 512 | 0.0404 ms | 1.2126 ms | 163.0 tok/s |
| 1024 | 0.0688 ms | 2.0625 ms | 143.2 tok/s |
| 2048 | 0.1401 ms | 4.2028 ms | 109.6 tok/s |
| 4096 | 0.3447 ms | 10.3404 ms | 65.5 tok/s |

At context 4096 attention became ~68% of the accounting time, so the next target was context parallelism rather than another ternary GEMV change.

## V9 — split-context exact attention

V9 implements Flash-Decoding-style context splitting with numerically stable softmax composition. Each chunk emits local `(m_i, l_i, O_i)` softmax state and a merge kernel reconstructs the exact full-context result using log-sum-exp scaling. Both split-Q and GQA4 KV-reuse variants were tested at chunk sizes 128/256/512.

Correctness:

- V8-style baseline `head0@128` vs CPU: PASS, max abs error 0.000030
- every tested split-Q and split-GQA4 configuration: PASS versus the baseline
- GPU cross-configuration BF16 differences at the selected larger-context runs were effectively zero

Best measured policy:

| Context | V8/V9 baseline | Best V9 path | Best time/layer | Attention speedup | GPU accounting ceiling |
|---:|---:|---|---:|---:|---:|
| 128 | 0.0130 ms | baseline | 0.0130 ms | 1.00x | 188.2 tok/s |
| 512 | 0.0333 ms | split-Q / 128 | 0.0189 ms | 1.76x | 182.1 tok/s |
| 1024 | 0.0641 ms | split-Q / 128 | 0.0196 ms | 3.27x | 181.5 tok/s |
| 2048 | 0.1279 ms | split-Q / 128 | 0.0275 ms | 4.64x | 174.0 tok/s |
| 4096 | 0.3399 ms | split-GQA4 / 128 | 0.0453 ms | **7.50x** | **159.2 tok/s** |

At context 4096 the 30-layer attention cost falls from roughly 10.2 ms to **1.3597 ms**. The fixed non-attention GPU accounting from V8 is 4.9223 ms, so attention is no longer the dominant cost. The bottleneck has shifted back to the fixed decoder body: 2.1265 ms ternary linears + 1.8712 ms fused support + 0.9166 ms BF16 LM head (plus final norm).

V9's practical dispatch implication is straightforward on this GA102/model pair:

- context <=128: baseline one-block/Q-head path
- context 512-2048: split-Q with 128-token chunks
- context ~4096: split-GQA4 with 128-token chunks

These are GPU-kernel accounting ceilings, not measured end-to-end text generation throughput.
