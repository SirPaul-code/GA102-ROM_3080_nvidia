#pragma once

#include <cstdint>
#include <functional>
#include <cuda_runtime.h>

// Forward declaration is required because NVCC's host preprocessing pass does
// not define __CUDA_ARCH__, while the device definition in main.cu is guarded
// for Ampere. The actual implementation is emitted only for device code.
__global__ void bmma_probe(const uint32_t* A, const uint32_t* B, int32_t* out, int loops);
