# Fixed target: microsoft/bitnet-b1.58-2B-4T

GA102-ROM now moves from generic 8192x8192 kernel experiments to the exact architecture of Microsoft's native ternary BitNet b1.58 2B-4T checkpoint.

Official model configuration:

- hidden size: 2560
- intermediate size: 6912
- layers: 30
- attention heads: 20
- KV heads: 5
- head dimension: 128
- KV projection width: 640
- vocabulary: 128256
- max context: 4096
- native W1.58/A8 BitLinear layers

References:

- https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/blob/main/config.json
- https://huggingface.co/microsoft/bitnet-b1.58-2B-4T

## Exact linear shapes per decoder layer

Without fusion:

| Projection | M (output) | K (input) | weights |
|---|---:|---:|---:|
| Q | 2560 | 2560 | 6,553,600 |
| K | 640 | 2560 | 1,638,400 |
| V | 640 | 2560 | 1,638,400 |
| O | 2560 | 2560 | 6,553,600 |
| gate | 6912 | 2560 | 17,694,720 |
| up | 6912 | 2560 | 17,694,720 |
| down | 2560 | 6912 | 17,694,720 |

Total per decoder layer: **69,468,160 ternary weights/MACs per token**.

Across 30 layers: **2,084,044,800 ternary linear MACs per decoded token**.

At a physical 2-bit representation, those 30 layers occupy exactly **496.875 MiB** before small metadata/scales.

## Fixed-runtime fusion plan

Because Q/K/V share the same input activation, GA102-ROM can compile them as one physical matrix and split/scalewise postprocess the output:

- fused QKV: **3840 x 2560**

Gate and up projections likewise share the same MLP input:

- fused gate+up: **13824 x 2560**

The four matrix passes per layer become:

| Fused pass | M | K | packed W size (2 bit) |
|---|---:|---:|---:|
| QKV | 3840 | 2560 | 2.34375 MiB |
| O | 2560 | 2560 | 1.56250 MiB |
| gate+up | 13824 | 2560 | 8.43750 MiB |
| down | 2560 | 6912 | 4.21875 MiB |

Total: **16.5625 MiB/layer**, or **496.875 MiB for all 30 layers**.

All four matrices naturally satisfy GA102-ROM's current BMMA constraints:

- every M is divisible by 16;
- K=2560 is exactly 10 x 256;
- K=6912 is exactly 27 x 256.

No padding is needed for `m16n8k256`.

## The output-head problem

The tied vocabulary/embedding matrix has:

- 128256 x 2560 = **328,335,360 values**
- at BF16: **656,670,720 bytes = 626.25 MiB**

The packed checkpoint file is about 1.18 GB. The arithmetic `496.875 MiB ternary decoder linears + 626.25 MiB BF16 tied vocab matrix` is already approximately the whole file size (before small scales/norms/metadata), so the tied vocabulary matrix is expected to become a major decode-bandwidth bottleneck once the ternary decoder is fast.

At the measured V2 sustained read rate of 556.3 GB/s, streaming a 626.25 MiB BF16 vocabulary matrix once has a weight-only lower bound of roughly **1.18 ms/token**. This is only a roofline estimate; the real LM-head implementation must be measured.

Potential fixed-model research directions for the LM head:

1. lower-bit surrogate + exact BF16 rerank;
2. provably safe candidate pruning using quantization-error bounds;
3. vocabulary clustering / hierarchical exact search;
4. separate GA102-native output-head layout;
5. exploit speculative N>1 logits work where a shared weight read can serve multiple hidden states.

## V6 benchmark requirement

Repeatedly benchmarking a small real projection can lie because QKV/O weights may fit substantially in L2. V6 therefore rotates through enough identical weight copies to exceed cache capacity and approximate the full-decoder streaming regime.

V6 measures the four fused real model shapes for:

- single-token raw-A8 pack + exact POPC;
- N=8 raw-A8 pack + exact BMMA;
- matrix-only and end-to-end pack+matrix timing;
- cold-ish rotating weight rings rather than one permanently hot matrix;
- aggregate 30-layer linear-only time.

The aggregate is intentionally labelled **linear-only**. It excludes attention score/value work, KV cache, RMSNorm/subln, RoPE, ReLU2/gating, residuals, output head, sampling and host/runtime overhead.
