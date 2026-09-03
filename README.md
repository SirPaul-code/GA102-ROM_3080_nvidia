# GA102-ROM

**One model. One GPU. No generality tax.**

GA102-ROM is a research runtime for pushing single-user LLM inference on an NVIDIA RTX 3080 10 GB (GA102, compute capability 8.6) as close as possible to the physical limits of that exact GPU.

The project intentionally does **not** aim to be a general CUDA inference framework. The long-term target is a fixed ternary / low-bit model, fixed tensor shapes, fixed memory layout and native `sm_86` kernels.

## First runnable milestone

The current executable is a hardware + kernel laboratory. It provides:

- strict RTX 3080 / SM86 device inspection;
- ternary `W1.58/W2 + A8` matrix-vector microbenchmarks;
- INT8 `DP4A` reference backend;
- 2-bit packed ternary weights + `DP4A` backend;
- exact two-bitplane ternary weights + activation bitplane `POPC` backend;
- raw packed-weight streaming benchmark;
- an Ampere **BMMA `m16n8k256` binary Tensor Core probe** using inline PTX `and.popc`;
- CPU correctness verification for all exact ternary GEMV backends;
- CUDA-event timings, logical GMAC/s and effective weight bandwidth.

The BMMA probe proves and measures the otherwise poorly surfaced 1-bit Tensor Core path on GA102. It is deliberately **not yet** wired into the full ternary GEMV path; that is the next research step.

## Quick start — Windows 11 + RTX 3080

Requirements:

1. NVIDIA driver
2. CUDA Toolkit 12.x (11.8+ should also work)
3. Visual Studio 2022 Build Tools with **Desktop development with C++**
4. CMake 3.24+

Then:

```powershell
git clone https://github.com/SirPaul-code/GA102-ROM_3080_nvidia.git
cd GA102-ROM_3080_nvidia
.\run.ps1
```

Or double-click / run:

```bat
run.bat
```

The script finds CUDA/CMake, checks Visual Studio, builds **native `sm_86` code only**, then runs the complete benchmark suite.

Useful examples:

```powershell
.\run.ps1 -BenchArgs "--info"
.\run.ps1 -BenchArgs "--ternary --m 8192 --k 8192 --iters 100"
.\run.ps1 -BenchArgs "--bmma --bmma-loops 8192"
.\run.ps1 -Clean
```

## Why this exists

Autoregressive decode repeatedly streams model weights. On a fixed consumer GPU, supporting arbitrary models, arbitrary shapes and arbitrary dtypes adds dispatch, packing and kernel compromises we do not need.

GA102-ROM explores the opposite extreme:

```text
fixed checkpoint
+ fixed RTX 3080 / sm_86
+ fixed layouts
+ generated / fused native kernels
= software-defined fixed-function inference appliance
```

Ampere SM86 gives us several unusual pieces of hardware worth testing directly:

- `DP4A` packed signed INT8 dot products;
- INT4 Tensor Core MMA;
- 1-bit BMMA Tensor Core operations with `AND/XOR + POPC`;
- hardware 2:4 structured sparsity;
- asynchronous global-to-shared copies;
- CUDA graphs and persistent kernels.

The central question is not "can we write another llama.cpp?" It is:

> If the checkpoint and GA102 are fixed forever, how close can software get to the GPU's actual bandwidth and Tensor Core limits?

## Current research ladder

1. **Measure:** exact ternary GEMV kernels and GA102 BMMA throughput.
2. **BMMA ternary GEMV:** map positive/negative ternary weight masks and A8/A4 bitplanes onto `mma.sync...b1...and.popc`.
3. **INT4 Tensor Core comparison:** compare bit-serial BMMA against native IMMA for the exact same layer shapes.
4. **2:4 ternary sparsity:** exploit Ampere sparse Tensor Cores where model quality permits it.
5. **Model compiler:** checkpoint -> GA102-ROM fixed binary layout.
6. **Static full-model runtime:** fuse RMSNorm/RoPE/GEMV/attention around fixed dimensions and remove generic dispatch.
7. **Persistent decode:** keep the decode loop GPU-resident where practical.
8. **LM-head compression / exact candidate pruning.**
9. **Speculative multi-token verification** to turn decode GEMV into more Tensor-Core-friendly GEMM.

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the experiment plan.

## Hardware references

- NVIDIA Ampere tuning guide: https://docs.nvidia.com/cuda/ampere-tuning-guide/
- PTX ISA, warp-level MMA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- GA102 architecture whitepaper: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf

## Status

Early research prototype. Results are only meaningful when correctness checks pass. Benchmark output intentionally distinguishes measured numbers from theoretical estimates.
