# GA102-ROM research plan

## Goal

Maximize **single-user autoregressive decode throughput** for one fixed low-bit / ternary model on one fixed NVIDIA RTX 3080 10 GB (GA102 / SM86).

This project treats generality as overhead. We are allowed to specialize:

- model architecture;
- layer dimensions;
- checkpoint;
- quantization scheme;
- weight packing;
- GPU architecture (`sm_86` only);
- kernel launch geometry;
- decoding path.

Every optimization must preserve a clearly stated numerical contract and must be benchmarked against a correctness reference.

## Hardware facts we target

Ampere GA10x exposes several relevant execution paths:

1. signed INT8 dot-product / DP4A;
2. INT8 and INT4 Tensor Core integer MMA;
3. binary Tensor Core MMA (`b1 x b1 -> s32`) with `AND/XOR + POPC`;
4. structured 2:4 sparse Tensor Core acceleration;
5. asynchronous global-to-shared copies;
6. CUDA graphs / persistent-kernel execution techniques.

Official references:

- https://docs.nvidia.com/cuda/ampere-tuning-guide/
- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf

## Experiment 0 — establish physical ceilings

Measure on the exact card:

- packed-weight sequential bandwidth;
- L2 behavior;
- launch overhead;
- DP4A throughput for real BitNet shapes;
- raw binary MMA throughput;
- INT4 MMA throughput (next milestone).

The important comparison is not a vendor theoretical TOPS number. It is **achieved performance for the exact decode shapes**.

## Experiment 1 — ternary arithmetic representations

For weights `w in {-1, 0, +1}` compare the following exact representations.

### A. Unpacked W8/A8 DP4A

Store each ternary weight in an `int8_t` and use DP4A.

Pros: trivial arithmetic, strong baseline.
Cons: 8 bits transferred for a value containing only three states.

### B. Packed W2/A8 -> decode -> DP4A

Encode four weights per byte:

```text
00 = -1
01 =  0
10 = +1
11 = reserved
```

Load 2-bit weights, expand four lanes in registers and execute DP4A.

Question: does 4x lower weight traffic outweigh decode instructions?

### C. Two binary weight planes + activation bitplanes

Represent ternary W by two masks:

```text
P[i] = 1 iff W[i] == +1
N[i] = 1 iff W[i] == -1
```

For signed A8, decompose each activation into two's-complement bitplanes `B_0 ... B_7`.

Then exactly:

```text
W dot A = sum_b scale[b] * ( popc(P & B_b) - popc(N & B_b) )
scale = {1,2,4,8,16,32,64,-128}
```

The first implementation uses scalar CUDA `__popc` as a correctness baseline.

### D. BMMA bit-serial Tensor Core path

Map the same binary fragments to Ampere:

```text
mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc
```

A single warp-level instruction processes a logical `16 x 8 x 256` binary MMA tile.

Research question:

> Can a layout specialized for fixed BitNet layer dimensions amortize activation bitplane construction and turn binary Tensor Cores into a faster W1.58/A8 or W1.58/A4 decode engine than DP4A / INT4 IMMA?

This is **not assumed to be true**. It must be measured.

## Experiment 2 — native INT4 Tensor Cores

A ternary value fits trivially in signed INT4. A4 activations allow native integer Tensor Core MMA with no bit-serial reconstruction.

Compare:

- BMMA bit-serial W1.58/A4;
- dense INT4 IMMA;
- structured-sparse INT4 IMMA;
- DP4A-based W2/A8.

Metrics:

- exact/quantized numerical agreement;
- tokens/s-equivalent layer throughput;
- effective GB/s;
- Tensor Core utilization;
- achieved occupancy;
- register pressure;
- joules/token when external measurement is available.

## Experiment 3 — fixed checkpoint compiler

Input:

```text
one supported checkpoint
```

Output:

```text
model.ga102rom
```

The compiler will precompute:

- layer-specific weight layout;
- tile swizzles;
- positive/negative bitplanes or INT4 fragments;
- scales;
- optional 2:4 metadata;
- alignment and padding for exact SM86 kernels.

No runtime weight conversion is allowed in the final path.

## Experiment 4 — static decode runtime

Once the fastest arithmetic backend is known, specialize the rest of the transformer:

- fixed RMSNorm dimensions;
- fused norm + projection preparation;
- fixed RoPE geometry;
- fixed attention head count / head dimension;
- preallocated KV cache;
- zero dynamic allocation in decode;
- static launch graph;
- eliminate dtype/shape dispatch.

The first target is CUDA Graph capture. A persistent cooperative decode kernel is only attempted if profiling proves launch/synchronization overhead is material.

## Experiment 5 — LM head

For models with a large vocabulary, the output projection may stream more bytes than the packed transformer layers.

Investigate exact candidate pruning:

1. cheap quantized projection supplies a logit estimate and rigorous error bound;
2. rows whose maximum possible exact logit cannot beat the current winner are eliminated;
3. full-precision evaluation is performed only for unresolved candidates.

For greedy decoding the target is the **exact same argmax**, not an approximate token.

## Experiment 6 — multi-token verification

Single-token decode is GEMV-like and difficult to saturate on Tensor Cores. Speculative decoding can verify several candidate tokens together, turning parts of decode into GEMM-like work.

The fixed-model appliance may integrate a dedicated draft/head mechanism so that kernel shapes and verification are compiled specifically for the target checkpoint.

## Benchmark discipline

Every reported result must include:

- exact GPU model;
- driver and CUDA version;
- clock/power configuration if modified;
- matrix/model dimensions;
- batch size and context length;
- numerical mode;
- correctness status;
- warmup and iteration counts;
- median or stable CUDA-event timing;
- comparison baseline from the same machine.

Do not report theoretical TOPS as achieved application throughput.
