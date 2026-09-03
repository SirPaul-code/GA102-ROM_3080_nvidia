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

V4 addresses these issues by compiling the fixed weights offline into the exact PTX A-fragment order (`uint4` per lane, contiguous per warp), loading each 256-K weight fragment once and reusing it across all eight activation bitplanes. V4 also measures an 8-column path where every native BMMA N=8 output is useful, matching speculative verification / small-batch execution rather than throwing 7/8 of the Tensor Core result away.
