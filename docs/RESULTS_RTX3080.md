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

## Interpretation after V2

The original kernels were primarily mapping / serialization limited, not simply VRAM-bandwidth limited. Converting from one-thread-per-row to one-warp-per-row radically changed all three backends.

The current fastest exact ternary path is the two one-bit weight masks (`W==+1`, `W==-1`) plus eight signed-A8 activation bitplanes using ordinary CUDA integer `POPC`. It reaches 1.517 TMAC/s logical ternary dot-product throughput on the 8192x8192 test.

The raw 1-bit Tensor Core probe independently demonstrated that GA102 executes `mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc` at high throughput. V3 therefore maps the same exact ternary arithmetic onto BMMA rather than scalar CUDA `POPC`.

No BMMA GEMV speed claim should be made until V3 passes exact CPU-reference correctness on the target GPU.
