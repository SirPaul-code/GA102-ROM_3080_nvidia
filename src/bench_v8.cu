#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t e__ = (call); \
    if (e__ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e__) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        std::exit(2); \
    } \
} while (0)

static constexpr int HIDDEN = 2560;
static constexpr int INTER = 6912;
static constexpr int Q_HEADS = 20;
static constexpr int KV_HEADS = 5;
static constexpr int HEAD_DIM = 128;
static constexpr int Q_DIM = Q_HEADS * HEAD_DIM;
static constexpr int KV_DIM = KV_HEADS * HEAD_DIM;
static constexpr int QKV_DIM = Q_DIM + 2 * KV_DIM;
static constexpr int GATEUP_DIM = 2 * INTER;
static constexpr int LAYERS = 30;
static constexpr float V6_LINEAR_MS = 2.1265f;
static constexpr float V7_LM_HEAD_MS = 0.9166f;

struct Options {
    int iters = 300;
    int max_context = 4096;
};

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "--max-context" && i + 1 < argc) o.max_context = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V8 fused decoder support + exact decode attention\n"
                      << "  --iters N        timing iterations (default 300)\n"
                      << "  --max-context N  max KV context (default 4096)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + a);
        }
    }
    if (o.iters <= 0 || o.max_context < 128 || o.max_context > 8192)
        throw std::runtime_error("iters must be positive; max-context must be 128..8192");
    return o;
}

static float time_ms(const std::function<void()>& launch, int warmup, int iters) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return total / iters;
}

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float out = (threadIdx.x < (blockDim.x >> 5)) ? scratch[lane] : 0.0f;
    if (warp == 0) out = warp_sum(out);
    __syncthreads();
    if (threadIdx.x == 0) scratch[0] = out;
    __syncthreads();
    return scratch[0];
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float block_max(float v, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float out = (threadIdx.x < (blockDim.x >> 5)) ? scratch[lane] : -CUDART_INF_F;
    if (warp == 0) out = warp_max(out);
    __syncthreads();
    if (threadIdx.x == 0) scratch[0] = out;
    __syncthreads();
    return scratch[0];
}

// Exact BitNet-style support primitive: RMSNorm followed by symmetric per-token A8 quantization.
// It matches the quantizer convention scale=127/max(abs(x)), q=round(x*scale) clipped to [-128,127].
__global__ void rmsnorm_quant_a8(const __nv_bfloat16* __restrict__ x,
                                 const __nv_bfloat16* __restrict__ gamma,
                                 int8_t* __restrict__ q,
                                 float* __restrict__ qscale,
                                 int n, float eps) {
    __shared__ float scratch[8];
    __shared__ float inv_rms;
    __shared__ float scale;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = bf16_to_f32(x[i]);
        ss += v * v;
    }
    const float sum = block_sum(ss, scratch);
    if (threadIdx.x == 0) inv_rms = rsqrtf(sum / (float)n + eps);
    __syncthreads();

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = bf16_to_f32(x[i]) * inv_rms * bf16_to_f32(gamma[i]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    const float mx = block_max(local_max, scratch);
    if (threadIdx.x == 0) {
        scale = 127.0f / fmaxf(mx, 1.0e-5f);
        qscale[0] = scale;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = bf16_to_f32(x[i]) * inv_rms * bf16_to_f32(gamma[i]);
        int qi = __float2int_rn(v * scale);
        qi = max(-128, min(127, qi));
        q[i] = (int8_t)qi;
    }
}

__global__ void rmsnorm_bf16(const __nv_bfloat16* __restrict__ x,
                             const __nv_bfloat16* __restrict__ gamma,
                             __nv_bfloat16* __restrict__ y,
                             int n, float eps) {
    __shared__ float scratch[8];
    __shared__ float inv_rms;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = bf16_to_f32(x[i]);
        ss += v * v;
    }
    const float sum = block_sum(ss, scratch);
    if (threadIdx.x == 0) inv_rms = rsqrtf(sum / (float)n + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[i] = f32_to_bf16(bf16_to_f32(x[i]) * inv_rms * bf16_to_f32(gamma[i]));
}

// QKV integer accumulator epilogue: dequant/postscale, RoPE for Q/K, and append K/V to cache.
__global__ void qkv_epilogue_rope_kv(const int32_t* __restrict__ qkv,
                                     __nv_bfloat16* __restrict__ qout,
                                     __nv_bfloat16* __restrict__ kcache,
                                     __nv_bfloat16* __restrict__ vcache,
                                     const float* __restrict__ cosv,
                                     const float* __restrict__ sinv,
                                     int position,
                                     float qscale, float kscale, float vscale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= QKV_DIM) return;

    if (i < Q_DIM) {
        const int d = i & 127;
        const int base = i - d;
        const int od = (d < 64) ? d + 64 : d - 64;
        const float x = (float)qkv[i] * qscale;
        const float xo = (float)qkv[base + od] * qscale;
        const float rot = (d < 64) ? -xo : xo;
        qout[i] = f32_to_bf16(x * cosv[d] + rot * sinv[d]);
    } else if (i < Q_DIM + KV_DIM) {
        const int j = i - Q_DIM;
        const int d = j & 127;
        const int base = i - d;
        const int od = (d < 64) ? d + 64 : d - 64;
        const float x = (float)qkv[i] * kscale;
        const float xo = (float)qkv[base + od] * kscale;
        const float rot = (d < 64) ? -xo : xo;
        kcache[(size_t)position * KV_DIM + j] = f32_to_bf16(x * cosv[d] + rot * sinv[d]);
    } else {
        const int j = i - Q_DIM - KV_DIM;
        vcache[(size_t)position * KV_DIM + j] = f32_to_bf16((float)qkv[i] * vscale);
    }
}

// Attention implementation A: one block per Q head. More parallel blocks, but each of the
// four Q heads in a GQA group independently traverses the same KV head. Shared memory holds L scores.
__global__ void attention_qhead_fused(const __nv_bfloat16* __restrict__ q,
                                      const __nv_bfloat16* __restrict__ kcache,
                                      const __nv_bfloat16* __restrict__ vcache,
                                      __nv_bfloat16* __restrict__ out,
                                      int context) {
    extern __shared__ float scores[];
    __shared__ float red[8];
    __shared__ float invsum;

    const int qh = blockIdx.x;
    const int kvh = qh >> 2;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const float scale = 0.08838834764831845f; // 1/sqrt(128)

    for (int p = warp; p < context; p += nwarps) {
        float acc = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += 32) {
            const float qv = bf16_to_f32(q[qh * HEAD_DIM + d]);
            const float kv = bf16_to_f32(kcache[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
            acc += qv * kv;
        }
        acc = warp_sum(acc);
        if (lane == 0) scores[p] = acc * scale;
    }
    __syncthreads();

    float lmax = -CUDART_INF_F;
    for (int p = threadIdx.x; p < context; p += blockDim.x) lmax = fmaxf(lmax, scores[p]);
    const float mx = block_max(lmax, red);

    float lsum = 0.0f;
    for (int p = threadIdx.x; p < context; p += blockDim.x) {
        const float e = expf(scores[p] - mx);
        scores[p] = e;
        lsum += e;
    }
    const float sm = block_sum(lsum, red);
    if (threadIdx.x == 0) invsum = 1.0f / sm;
    __syncthreads();

    if (threadIdx.x < HEAD_DIM) {
        const int d = threadIdx.x;
        float acc = 0.0f;
        for (int p = 0; p < context; ++p) {
            const float vv = bf16_to_f32(vcache[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
            acc += (scores[p] * invsum) * vv;
        }
        out[qh * HEAD_DIM + d] = f32_to_bf16(acc);
    }
}

// Attention implementation B: one block per KV head computes all four associated Q heads together.
// K and V are each loaded once and reused across the 4 Q heads, at the cost of only 5 blocks total.
__global__ void attention_gqa4_reuse(const __nv_bfloat16* __restrict__ q,
                                     const __nv_bfloat16* __restrict__ kcache,
                                     const __nv_bfloat16* __restrict__ vcache,
                                     __nv_bfloat16* __restrict__ out,
                                     int context) {
    extern __shared__ float scores[]; // [4][context]
    __shared__ float red[8];
    __shared__ float invsum[4];

    const int kvh = blockIdx.x;
    const int qbase = kvh * 4;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const float scale = 0.08838834764831845f;

    for (int p = warp; p < context; p += nwarps) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += 32) {
            const float kv = bf16_to_f32(kcache[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
            a0 += bf16_to_f32(q[(qbase + 0) * HEAD_DIM + d]) * kv;
            a1 += bf16_to_f32(q[(qbase + 1) * HEAD_DIM + d]) * kv;
            a2 += bf16_to_f32(q[(qbase + 2) * HEAD_DIM + d]) * kv;
            a3 += bf16_to_f32(q[(qbase + 3) * HEAD_DIM + d]) * kv;
        }
        a0 = warp_sum(a0); a1 = warp_sum(a1); a2 = warp_sum(a2); a3 = warp_sum(a3);
        if (lane == 0) {
            scores[(size_t)0 * context + p] = a0 * scale;
            scores[(size_t)1 * context + p] = a1 * scale;
            scores[(size_t)2 * context + p] = a2 * scale;
            scores[(size_t)3 * context + p] = a3 * scale;
        }
    }
    __syncthreads();

    for (int h = 0; h < 4; ++h) {
        float lmax = -CUDART_INF_F;
        float* s = scores + (size_t)h * context;
        for (int p = threadIdx.x; p < context; p += blockDim.x) lmax = fmaxf(lmax, s[p]);
        const float mx = block_max(lmax, red);
        float lsum = 0.0f;
        for (int p = threadIdx.x; p < context; p += blockDim.x) {
            const float e = expf(s[p] - mx);
            s[p] = e;
            lsum += e;
        }
        const float sm = block_sum(lsum, red);
        if (threadIdx.x == 0) invsum[h] = 1.0f / sm;
        __syncthreads();
    }

    if (threadIdx.x < HEAD_DIM) {
        const int d = threadIdx.x;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int p = 0; p < context; ++p) {
            const float vv = bf16_to_f32(vcache[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
            a0 += scores[(size_t)0 * context + p] * invsum[0] * vv;
            a1 += scores[(size_t)1 * context + p] * invsum[1] * vv;
            a2 += scores[(size_t)2 * context + p] * invsum[2] * vv;
            a3 += scores[(size_t)3 * context + p] * invsum[3] * vv;
        }
        out[(qbase + 0) * HEAD_DIM + d] = f32_to_bf16(a0);
        out[(qbase + 1) * HEAD_DIM + d] = f32_to_bf16(a1);
        out[(qbase + 2) * HEAD_DIM + d] = f32_to_bf16(a2);
        out[(qbase + 3) * HEAD_DIM + d] = f32_to_bf16(a3);
    }
}

// O accumulator epilogue + first residual + post-attention RMSNorm + A8 quant for fused gate/up.
__global__ void o_residual_postnorm_quant(const int32_t* __restrict__ oacc,
                                          const __nv_bfloat16* __restrict__ residual,
                                          const __nv_bfloat16* __restrict__ gamma,
                                          __nv_bfloat16* __restrict__ hidden_mid,
                                          int8_t* __restrict__ qout,
                                          float* __restrict__ qscale,
                                          float out_scale, float eps) {
    __shared__ float scratch[8];
    __shared__ float inv_rms;
    __shared__ float scale;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        const float v = bf16_to_f32(residual[i]) + (float)oacc[i] * out_scale;
        hidden_mid[i] = f32_to_bf16(v);
        ss += v * v;
    }
    const float sum = block_sum(ss, scratch);
    if (threadIdx.x == 0) inv_rms = rsqrtf(sum / (float)HIDDEN + eps);
    __syncthreads();

    float lm = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        const float v = bf16_to_f32(hidden_mid[i]) * inv_rms * bf16_to_f32(gamma[i]);
        lm = fmaxf(lm, fabsf(v));
    }
    const float mx = block_max(lm, scratch);
    if (threadIdx.x == 0) {
        scale = 127.0f / fmaxf(mx, 1.0e-5f);
        qscale[0] = scale;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        const float v = bf16_to_f32(hidden_mid[i]) * inv_rms * bf16_to_f32(gamma[i]);
        int qi = __float2int_rn(v * scale);
        qi = max(-128, min(127, qi));
        qout[i] = (int8_t)qi;
    }
}

// Fused gate/up dequant + ReLU^2(gate)*up + FFN subnorm + A8 quant for down projection.
__global__ void gateup_relu2_ffnnorm_quant(const int32_t* __restrict__ gu,
                                           const __nv_bfloat16* __restrict__ gamma,
                                           __nv_bfloat16* __restrict__ tmp,
                                           int8_t* __restrict__ qout,
                                           float* __restrict__ qscale,
                                           float gate_scale, float up_scale,
                                           float eps) {
    __shared__ float scratch[8];
    __shared__ float inv_rms;
    __shared__ float scale;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < INTER; i += blockDim.x) {
        const float g = (float)gu[i] * gate_scale;
        const float u = (float)gu[INTER + i] * up_scale;
        const float r = fmaxf(g, 0.0f);
        const float z = (r * r) * u;
        tmp[i] = f32_to_bf16(z);
        ss += z * z;
    }
    const float sum = block_sum(ss, scratch);
    if (threadIdx.x == 0) inv_rms = rsqrtf(sum / (float)INTER + eps);
    __syncthreads();

    float lm = 0.0f;
    for (int i = threadIdx.x; i < INTER; i += blockDim.x) {
        const float v = bf16_to_f32(tmp[i]) * inv_rms * bf16_to_f32(gamma[i]);
        lm = fmaxf(lm, fabsf(v));
    }
    const float mx = block_max(lm, scratch);
    if (threadIdx.x == 0) {
        scale = 127.0f / fmaxf(mx, 1.0e-5f);
        qscale[0] = scale;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < INTER; i += blockDim.x) {
        const float v = bf16_to_f32(tmp[i]) * inv_rms * bf16_to_f32(gamma[i]);
        int qi = __float2int_rn(v * scale);
        qi = max(-128, min(127, qi));
        qout[i] = (int8_t)qi;
    }
}

__global__ void down_residual_epilogue(const int32_t* __restrict__ down,
                                       const __nv_bfloat16* __restrict__ residual,
                                       __nv_bfloat16* __restrict__ out,
                                       float out_scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN)
        out[i] = f32_to_bf16(bf16_to_f32(residual[i]) + (float)down[i] * out_scale);
}

static void cpu_attention_head0(const std::vector<__nv_bfloat16>& q,
                                const std::vector<__nv_bfloat16>& k,
                                const std::vector<__nv_bfloat16>& v,
                                std::vector<float>& out,
                                int context) {
    std::vector<float> s(context);
    const float scale = 0.08838834764831845f;
    float mx = -std::numeric_limits<float>::infinity();
    for (int p = 0; p < context; ++p) {
        float a = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d)
            a += __bfloat162float(q[d]) * __bfloat162float(k[(size_t)p * KV_DIM + d]);
        s[p] = a * scale;
        mx = std::max(mx, s[p]);
    }
    float sm = 0.0f;
    for (float& x : s) { x = std::exp(x - mx); sm += x; }
    for (float& x : s) x /= sm;
    out.assign(HEAD_DIM, 0.0f);
    for (int d = 0; d < HEAD_DIM; ++d) {
        float a = 0.0f;
        for (int p = 0; p < context; ++p)
            a += s[p] * __bfloat162float(v[(size_t)p * KV_DIM + d]);
        out[d] = a;
    }
}

int main(int argc, char** argv) {
    try {
        const Options o = parse_args(argc, argv);
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 8 || prop.minor != 6 || std::string(prop.name).find("RTX 3080") == std::string::npos) {
            std::cerr << "V8 is intentionally restricted to RTX 3080 / SM86; found "
                      << prop.name << " sm_" << prop.major << prop.minor << "\n";
            return 3;
        }

        std::cout << "GA102-ROM V8: fused support kernels + exact GQA decode attention\n"
                  << "GPU               : " << prop.name << "\n"
                  << "model             : microsoft/bitnet-b1.58-2B-4T shapes\n"
                  << "heads             : Q=" << Q_HEADS << " KV=" << KV_HEADS << " dim=" << HEAD_DIM << "\n"
                  << "V6 linears        : " << V6_LINEAR_MS << " ms/token (30 layers)\n"
                  << "V7 BF16 LM head   : " << V7_LM_HEAD_MS << " ms/token\n";

        // Deterministic state buffers.
        std::vector<__nv_bfloat16> h_hidden(HIDDEN), h_gamma_h(HIDDEN), h_gamma_i(INTER);
        std::vector<__nv_bfloat16> h_q(Q_DIM), h_k((size_t)o.max_context * KV_DIM), h_v((size_t)o.max_context * KV_DIM);
        for (int i = 0; i < HIDDEN; ++i) {
            h_hidden[i] = __float2bfloat16((float)((i % 97) - 48) / 64.0f);
            h_gamma_h[i] = __float2bfloat16(1.0f + 0.001f * (float)(i % 11));
        }
        for (int i = 0; i < INTER; ++i) h_gamma_i[i] = __float2bfloat16(1.0f + 0.001f * (float)(i % 13));
        for (int h = 0; h < Q_HEADS; ++h)
            for (int d = 0; d < HEAD_DIM; ++d)
                h_q[h * HEAD_DIM + d] = __float2bfloat16((float)(((h * 131 + d * 17) % 101) - 50) / 80.0f);
        for (int p = 0; p < o.max_context; ++p) {
            for (int h = 0; h < KV_HEADS; ++h) {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    const int z = (p * 29 + h * 37 + d * 11) % 127;
                    h_k[(size_t)p * KV_DIM + h * HEAD_DIM + d] = __float2bfloat16((float)(z - 63) / 96.0f);
                    const int w = (p * 17 + h * 19 + d * 23) % 113;
                    h_v[(size_t)p * KV_DIM + h * HEAD_DIM + d] = __float2bfloat16((float)(w - 56) / 88.0f);
                }
            }
        }

        __nv_bfloat16 *d_hidden=nullptr, *d_gamma_h=nullptr, *d_gamma_i=nullptr;
        __nv_bfloat16 *d_q=nullptr, *d_k=nullptr, *d_v=nullptr, *d_attn=nullptr, *d_attn2=nullptr;
        __nv_bfloat16 *d_mid=nullptr, *d_tmpi=nullptr, *d_final=nullptr, *d_finalnorm=nullptr;
        int8_t *d_qh=nullptr, *d_qi=nullptr;
        float *d_scale_h=nullptr, *d_scale_i=nullptr;
        int32_t *d_qkv=nullptr, *d_o=nullptr, *d_gu=nullptr, *d_down=nullptr;
        float *d_cos=nullptr, *d_sin=nullptr;

        CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_gamma_h, HIDDEN*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_gamma_i, INTER*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_q, Q_DIM*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k, (size_t)o.max_context*KV_DIM*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_v, (size_t)o.max_context*KV_DIM*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_attn, Q_DIM*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_attn2, Q_DIM*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_mid, HIDDEN*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_tmpi, INTER*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_final, HIDDEN*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_finalnorm, HIDDEN*sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_qh, HIDDEN*sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_qi, INTER*sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_scale_h, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scale_i, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_qkv, QKV_DIM*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_o, HIDDEN*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_gu, GATEUP_DIM*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_down, HIDDEN*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_cos, HEAD_DIM*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sin, HEAD_DIM*sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(), HIDDEN*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma_h, h_gamma_h.data(), HIDDEN*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma_i, h_gamma_i.data(), INTER*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), Q_DIM*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size()*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size()*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        std::vector<int32_t> hqkv(QKV_DIM), ho(HIDDEN), hgu(GATEUP_DIM), hdown(HIDDEN);
        for (int i=0;i<QKV_DIM;++i) hqkv[i]=(i%257)-128;
        for (int i=0;i<HIDDEN;++i) { ho[i]=(i%193)-96; hdown[i]=(i%181)-90; }
        for (int i=0;i<GATEUP_DIM;++i) hgu[i]=(i%149)-74;
        CUDA_CHECK(cudaMemcpy(d_qkv, hqkv.data(), hqkv.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_o, ho.data(), ho.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gu, hgu.data(), hgu.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_down, hdown.data(), hdown.size()*sizeof(int32_t), cudaMemcpyHostToDevice));

        std::vector<float> hcos(HEAD_DIM), hsin(HEAD_DIM);
        for (int d=0; d<HEAD_DIM; ++d) {
            const float ang = 0.0007f * (float)(d+1);
            hcos[d] = std::cos(ang); hsin[d] = std::sin(ang);
        }
        CUDA_CHECK(cudaMemcpy(d_cos, hcos.data(), HEAD_DIM*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin, hsin.data(), HEAD_DIM*sizeof(float), cudaMemcpyHostToDevice));

        const int t = 256;
        auto launch_in_normq = [&]{ rmsnorm_quant_a8<<<1,t>>>(d_hidden,d_gamma_h,d_qh,d_scale_h,HIDDEN,1e-6f); };
        auto launch_qkv_epi = [&]{ qkv_epilogue_rope_kv<<<(QKV_DIM+t-1)/t,t>>>(d_qkv,d_q,d_k,d_v,d_cos,d_sin,0,0.001f,0.001f,0.001f); };
        auto launch_attn_normq = [&]{ rmsnorm_quant_a8<<<1,t>>>(d_attn,d_gamma_h,d_qh,d_scale_h,HIDDEN,1e-6f); };
        auto launch_o_fused = [&]{ o_residual_postnorm_quant<<<1,t>>>(d_o,d_hidden,d_gamma_h,d_mid,d_qh,d_scale_h,0.001f,1e-6f); };
        auto launch_gu_fused = [&]{ gateup_relu2_ffnnorm_quant<<<1,t>>>(d_gu,d_gamma_i,d_tmpi,d_qi,d_scale_i,0.0005f,0.0005f,1e-6f); };
        auto launch_down_epi = [&]{ down_residual_epilogue<<<(HIDDEN+t-1)/t,t>>>(d_down,d_mid,d_final,0.001f); };
        auto launch_final_norm = [&]{ rmsnorm_bf16<<<1,t>>>(d_final,d_gamma_h,d_finalnorm,HIDDEN,1e-6f); };

        const float in_normq_ms = time_ms(launch_in_normq,30,o.iters);
        const float qkv_epi_ms = time_ms(launch_qkv_epi,30,o.iters);
        const float attn_normq_ms = time_ms(launch_attn_normq,30,o.iters);
        const float o_fused_ms = time_ms(launch_o_fused,30,o.iters);
        const float gu_fused_ms = time_ms(launch_gu_fused,30,o.iters);
        const float down_epi_ms = time_ms(launch_down_epi,30,o.iters);
        const float final_norm_ms = time_ms(launch_final_norm,30,o.iters);
        const float support_layer_ms = in_normq_ms+qkv_epi_ms+attn_normq_ms+o_fused_ms+gu_fused_ms+down_epi_ms;

        std::cout << "\n=== V8 fused per-layer support ===\n"
                  << std::fixed << std::setprecision(4)
                  << "input RMSNorm + A8 quant          : " << in_normq_ms << " ms\n"
                  << "QKV postscale + RoPE + KV append  : " << qkv_epi_ms << " ms\n"
                  << "attn subnorm + O-input A8 quant   : " << attn_normq_ms << " ms\n"
                  << "O postscale+resid+norm+gate A8    : " << o_fused_ms << " ms\n"
                  << "gate/up+ReLU2+FFN norm+down A8    : " << gu_fused_ms << " ms\n"
                  << "down postscale + residual         : " << down_epi_ms << " ms\n"
                  << "fused support / layer             : " << support_layer_ms << " ms\n"
                  << "fused support / 30L               : " << support_layer_ms*LAYERS << " ms\n"
                  << "V7 launch-separated aux / 30L     : 3.5100 ms\n"
                  << "support fusion speedup             : " << (3.5100f/(support_layer_ms*LAYERS)) << "x\n"
                  << "final RMSNorm (once/token)         : " << final_norm_ms << " ms\n";

        // Attention correctness at context 128 against CPU head 0.
        const int check_ctx = 128;
        const size_t qhead_smem = (size_t)check_ctx*sizeof(float);
        attention_qhead_fused<<<Q_HEADS,256,qhead_smem>>>(d_q,d_k,d_v,d_attn,check_ctx);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> got(Q_DIM);
        CUDA_CHECK(cudaMemcpy(got.data(), d_attn, Q_DIM*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        std::vector<float> ref;
        cpu_attention_head0(h_q,h_k,h_v,ref,check_ctx);
        float max_err = 0.0f;
        for (int d=0; d<HEAD_DIM; ++d) max_err = std::max(max_err, std::fabs(ref[d]-__bfloat162float(got[d])));
        const bool attn_ok = max_err < 0.03f;

        std::cout << "\n=== V8 exact GQA decode attention ===\n"
                  << "correctness head0@128 : " << (attn_ok?"PASS":"FAIL")
                  << "  max_abs_err=" << std::setprecision(6) << max_err << "\n";
        if (!attn_ok) return 5;

        std::array<int,5> ctxs{{128,512,1024,2048,4096}};
        for (int ctx : ctxs) {
            if (ctx > o.max_context) continue;
            const size_t smem_q = (size_t)ctx*sizeof(float);
            const size_t smem_g = (size_t)4*ctx*sizeof(float);
            CUDA_CHECK(cudaFuncSetAttribute(attention_gqa4_reuse,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_g));
            auto launch_qh = [&]{ attention_qhead_fused<<<Q_HEADS,256,smem_q>>>(d_q,d_k,d_v,d_attn,ctx); };
            auto launch_g4 = [&]{ attention_gqa4_reuse<<<KV_HEADS,256,smem_g>>>(d_q,d_k,d_v,d_attn2,ctx); };
            const float qh_ms = time_ms(launch_qh,20,o.iters);
            const float g4_ms = time_ms(launch_g4,20,o.iters);
            const float best = std::min(qh_ms,g4_ms);

            launch_qh(); launch_g4(); CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<__nv_bfloat16> a(Q_DIM), b(Q_DIM);
            CUDA_CHECK(cudaMemcpy(a.data(),d_attn,Q_DIM*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b.data(),d_attn2,Q_DIM*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
            float diff=0.0f;
            for (int i=0;i<Q_DIM;++i) diff=std::max(diff,std::fabs(__bfloat162float(a[i])-__bfloat162float(b[i])));

            const double kv_bytes = (double)ctx*KV_DIM*2.0*2.0; // one K+V pass, BF16
            std::cout << std::fixed << std::setprecision(4)
                      << "ctx " << std::setw(4) << ctx
                      << "  qhead=" << qh_ms << " ms"
                      << "  gqa4-reuse=" << g4_ms << " ms"
                      << "  best=" << best << " ms/layer"
                      << "  best-30L=" << best*LAYERS << " ms"
                      << "  one-pass-KV=" << std::setprecision(1) << (kv_bytes/(best*1e6)) << " GB/s"
                      << "  cross-kernel-diff=" << std::setprecision(5) << diff << "\n";

            const float gpu_account = V6_LINEAR_MS + support_layer_ms*LAYERS + best*LAYERS + final_norm_ms + V7_LM_HEAD_MS;
            std::cout << std::setprecision(4)
                      << "          GPU decode accounting: " << gpu_account << " ms => "
                      << std::setprecision(1) << (1000.0f/gpu_account) << " tok/s ceiling\n";
        }

        std::cout << "\nGuardrails:\n"
                  << "  * attention kernels perform QK scaling, softmax, and weighted V; KV reads are included.\n"
                  << "  * V8 accounting combines measured V6 linears, fused support, measured attention, final norm, and V7 LM head.\n"
                  << "  * it is still a GPU-kernel accounting ceiling, not tokenizer/sampling/host end-to-end generation.\n"
                  << "  * matrix epilogues use representative scales; V8 measures runtime structure/cost, not checkpoint-level numerical equivalence.\n";

        cudaFree(d_hidden); cudaFree(d_gamma_h); cudaFree(d_gamma_i); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_attn); cudaFree(d_attn2); cudaFree(d_mid); cudaFree(d_tmpi); cudaFree(d_final); cudaFree(d_finalnorm);
        cudaFree(d_qh); cudaFree(d_qi); cudaFree(d_scale_h); cudaFree(d_scale_i); cudaFree(d_qkv); cudaFree(d_o);
        cudaFree(d_gu); cudaFree(d_down); cudaFree(d_cos); cudaFree(d_sin);

        std::cout << "\nV8 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
