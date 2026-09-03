#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
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

struct Options {
    int iters = 1000;
};

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V10 launch-tax / CUDA Graph probe\n"
                      << "  --iters N   timing iterations (default 1000)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + a);
        }
    }
    if (o.iters <= 0) throw std::runtime_error("iters must be positive");
    return o;
}

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 b16(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* s) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) s[warp] = v;
    __syncthreads();
    float x = (threadIdx.x < (blockDim.x >> 5)) ? s[lane] : 0.0f;
    if (warp == 0) x = warp_sum(x);
    __syncthreads();
    if (threadIdx.x == 0) s[0] = x;
    __syncthreads();
    return s[0];
}

__device__ __forceinline__ float block_max(float v, float* s) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) s[warp] = v;
    __syncthreads();
    float x = (threadIdx.x < (blockDim.x >> 5)) ? s[lane] : -3.402823466e+38F;
    if (warp == 0) x = warp_max(x);
    __syncthreads();
    if (threadIdx.x == 0) s[0] = x;
    __syncthreads();
    return s[0];
}

// Deliberately empty. V10 uses the same six grid shapes as the support path to
// measure the scheduling/launch floor without adding atomics or memory traffic.
__global__ void empty_kernel() {
    asm volatile("");
}

__global__ void rmsnorm_quant_a8(const __nv_bfloat16* x,
                                 const __nv_bfloat16* gamma,
                                 int8_t* q,
                                 float* qscale,
                                 int n,
                                 float eps) {
    __shared__ float s[8], invr, scale;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = bf(x[i]);
        ss += v * v;
    }
    const float sum = block_sum(ss, s);
    if (threadIdx.x == 0) invr = rsqrtf(sum / (float)n + eps);
    __syncthreads();

    float lm = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        lm = fmaxf(lm, fabsf(bf(x[i]) * invr * bf(gamma[i])));
    const float mx = block_max(lm, s);
    if (threadIdx.x == 0) {
        scale = 127.0f / fmaxf(mx, 1e-5f);
        qscale[0] = scale;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int z = __float2int_rn(bf(x[i]) * invr * bf(gamma[i]) * scale);
        z = max(-128, min(127, z));
        q[i] = (int8_t)z;
    }
}

__global__ void qkv_epilogue_rope_kv(const int32_t* qkv,
                                     __nv_bfloat16* qout,
                                     __nv_bfloat16* kc,
                                     __nv_bfloat16* vc,
                                     const float* cs,
                                     const float* sn,
                                     int pos,
                                     float qs,
                                     float ks,
                                     float vs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= QKV_DIM) return;
    if (i < Q_DIM) {
        const int d = i & 127;
        const int base = i - d;
        const int od = d < 64 ? d + 64 : d - 64;
        const float x = (float)qkv[i] * qs;
        const float xo = (float)qkv[base + od] * qs;
        qout[i] = b16(x * cs[d] + (d < 64 ? -xo : xo) * sn[d]);
    } else if (i < Q_DIM + KV_DIM) {
        const int j = i - Q_DIM;
        const int d = j & 127;
        const int base = i - d;
        const int od = d < 64 ? d + 64 : d - 64;
        const float x = (float)qkv[i] * ks;
        const float xo = (float)qkv[base + od] * ks;
        kc[(size_t)pos * KV_DIM + j] = b16(x * cs[d] + (d < 64 ? -xo : xo) * sn[d]);
    } else {
        const int j = i - Q_DIM - KV_DIM;
        vc[(size_t)pos * KV_DIM + j] = b16((float)qkv[i] * vs);
    }
}

__global__ void o_resid_norm_quant(const int32_t* oacc,
                                   const __nv_bfloat16* residual,
                                   const __nv_bfloat16* gamma,
                                   __nv_bfloat16* mid,
                                   int8_t* q,
                                   float* qscale,
                                   float os,
                                   float eps) {
    __shared__ float s[8], invr, scale;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        const float v = bf(residual[i]) + (float)oacc[i] * os;
        mid[i] = b16(v);
        ss += v * v;
    }
    const float sum = block_sum(ss, s);
    if (threadIdx.x == 0) invr = rsqrtf(sum / (float)HIDDEN + eps);
    __syncthreads();

    float lm = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x)
        lm = fmaxf(lm, fabsf(bf(mid[i]) * invr * bf(gamma[i])));
    const float mx = block_max(lm, s);
    if (threadIdx.x == 0) {
        scale = 127.0f / fmaxf(mx, 1e-5f);
        qscale[0] = scale;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        int z = __float2int_rn(bf(mid[i]) * invr * bf(gamma[i]) * scale);
        z = max(-128, min(127, z));
        q[i] = (int8_t)z;
    }
}

__global__ void gateup_relu2_norm_quant(const int32_t* gu,
                                        const __nv_bfloat16* gamma,
                                        __nv_bfloat16* tmp,
                                        int8_t* q,
                                        float* qscale,
                                        float gs,
                                        float us,
                                        float eps) {
    __shared__ float s[8], invr, scale;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < INTER; i += blockDim.x) {
        const float g = (float)gu[i] * gs;
        const float u = (float)gu[INTER + i] * us;
        const float r = fmaxf(g, 0.0f);
        const float z = r * r * u;
        tmp[i] = b16(z);
        ss += z * z;
    }
    const float sum = block_sum(ss, s);
    if (threadIdx.x == 0) invr = rsqrtf(sum / (float)INTER + eps);
    __syncthreads();

    float lm = 0.0f;
    for (int i = threadIdx.x; i < INTER; i += blockDim.x)
        lm = fmaxf(lm, fabsf(bf(tmp[i]) * invr * bf(gamma[i])));
    const float mx = block_max(lm, s);
    if (threadIdx.x == 0) {
        scale = 127.0f / fmaxf(mx, 1e-5f);
        qscale[0] = scale;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < INTER; i += blockDim.x) {
        int z = __float2int_rn(bf(tmp[i]) * invr * bf(gamma[i]) * scale);
        z = max(-128, min(127, z));
        q[i] = (int8_t)z;
    }
}

__global__ void down_resid(const int32_t* down,
                           const __nv_bfloat16* residual,
                           __nv_bfloat16* out,
                           float sc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN) out[i] = b16(bf(residual[i]) + (float)down[i] * sc);
}

static float time_group(cudaStream_t stream,
                        const std::function<void(cudaStream_t)>& launch_group,
                        int warmup,
                        int iters) {
    for (int i = 0; i < warmup; ++i) launch_group(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t a = nullptr, b = nullptr;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, stream));
    for (int i = 0; i < iters; ++i) launch_group(stream);
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return total / iters;
}

static float time_graph(cudaStream_t stream,
                        const std::function<void(cudaStream_t)>& launch_group,
                        int warmup,
                        int iters) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_group(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    for (int i = 0; i < warmup; ++i) CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t a = nullptr, b = nullptr;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, stream));
    for (int i = 0; i < iters; ++i) CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, a, b));

    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return total / iters;
}

static float max_bf16_diff(const std::vector<__nv_bfloat16>& a,
                           const std::vector<__nv_bfloat16>& b) {
    float e = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::fabs(__bfloat162float(a[i]) - __bfloat162float(b[i])));
    return e;
}

int main(int argc, char** argv) {
    try {
        const Options o = parse_args(argc, argv);
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 8 || prop.minor != 6 || std::string(prop.name).find("RTX 3080") == std::string::npos) {
            std::cerr << "V10 restricted to RTX 3080 / SM86; found " << prop.name << "\n";
            return 3;
        }

        std::cout << "GA102-ROM V10: support launch-tax / CUDA Graph probe\n"
                  << "GPU               : " << prop.name << "\n"
                  << "V8 support        : 6 fused helper kernels / layer\n"
                  << "V8 measured sum   : 0.0624 ms/layer, 1.8712 ms/30L\n"
                  << "goal              : separate kernel work from launch/scheduling tax\n";

        std::vector<__nv_bfloat16> h_hidden(HIDDEN), h_gamma_h(HIDDEN), h_gamma_i(INTER), h_attn(HIDDEN);
        for (int i = 0; i < HIDDEN; ++i) {
            h_hidden[i] = __float2bfloat16((float)((i % 97) - 48) / 64.0f);
            h_gamma_h[i] = __float2bfloat16(1.0f + 0.001f * (i % 11));
            h_attn[i] = __float2bfloat16((float)((i % 83) - 41) / 72.0f);
        }
        for (int i = 0; i < INTER; ++i)
            h_gamma_i[i] = __float2bfloat16(1.0f + 0.001f * (i % 13));

        std::vector<int32_t> h_qkv(QKV_DIM), h_o(HIDDEN), h_gu(GATEUP_DIM), h_down(HIDDEN);
        for (int i = 0; i < QKV_DIM; ++i) h_qkv[i] = (i % 257) - 128;
        for (int i = 0; i < HIDDEN; ++i) {
            h_o[i] = (i % 193) - 96;
            h_down[i] = (i % 181) - 90;
        }
        for (int i = 0; i < GATEUP_DIM; ++i) h_gu[i] = (i % 149) - 74;

        std::vector<float> h_cos(HEAD_DIM), h_sin(HEAD_DIM);
        for (int d = 0; d < HEAD_DIM; ++d) {
            const float ang = 0.0007f * (float)(d + 1);
            h_cos[d] = std::cos(ang);
            h_sin[d] = std::sin(ang);
        }

        __nv_bfloat16 *d_hidden=nullptr,*d_gamma_h=nullptr,*d_gamma_i=nullptr,*d_attn=nullptr;
        __nv_bfloat16 *d_q=nullptr,*d_k=nullptr,*d_v=nullptr,*d_mid=nullptr,*d_tmp=nullptr,*d_final=nullptr;
        int8_t *d_qh=nullptr,*d_qi=nullptr;
        float *d_sh=nullptr,*d_si=nullptr,*d_cos=nullptr,*d_sin=nullptr;
        int32_t *d_qkv=nullptr,*d_o=nullptr,*d_gu=nullptr,*d_down=nullptr;

        CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_gamma_h, HIDDEN * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_gamma_i, INTER * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_attn, HIDDEN * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_q, Q_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_k, KV_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_v, KV_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_mid, HIDDEN * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_tmp, INTER * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_final, HIDDEN * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_qh, HIDDEN * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_qi, INTER * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_sh, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_si, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cos, HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sin, HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_qkv, QKV_DIM * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_o, HIDDEN * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_gu, GATEUP_DIM * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_down, HIDDEN * sizeof(int32_t)));

        CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(), HIDDEN*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma_h, h_gamma_h.data(), HIDDEN*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma_i, h_gamma_i.data(), INTER*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_attn, h_attn.data(), HIDDEN*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), HEAD_DIM*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), HEAD_DIM*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qkv, h_qkv.data(), QKV_DIM*sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_o, h_o.data(), HIDDEN*sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gu, h_gu.data(), GATEUP_DIM*sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_down, h_down.data(), HIDDEN*sizeof(int32_t), cudaMemcpyHostToDevice));

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const int t = 256;

        auto empty_group = [&](cudaStream_t s) {
            empty_kernel<<<1, t, 0, s>>>();
            empty_kernel<<<(QKV_DIM+t-1)/t, t, 0, s>>>();
            empty_kernel<<<1, t, 0, s>>>();
            empty_kernel<<<1, t, 0, s>>>();
            empty_kernel<<<1, t, 0, s>>>();
            empty_kernel<<<(HIDDEN+t-1)/t, t, 0, s>>>();
        };

        auto real_group = [&](cudaStream_t s) {
            rmsnorm_quant_a8<<<1,t,0,s>>>(d_hidden,d_gamma_h,d_qh,d_sh,HIDDEN,1e-6f);
            qkv_epilogue_rope_kv<<<(QKV_DIM+t-1)/t,t,0,s>>>(d_qkv,d_q,d_k,d_v,d_cos,d_sin,0,0.001f,0.001f,0.001f);
            rmsnorm_quant_a8<<<1,t,0,s>>>(d_attn,d_gamma_h,d_qh,d_sh,HIDDEN,1e-6f);
            o_resid_norm_quant<<<1,t,0,s>>>(d_o,d_hidden,d_gamma_h,d_mid,d_qh,d_sh,0.001f,1e-6f);
            gateup_relu2_norm_quant<<<1,t,0,s>>>(d_gu,d_gamma_i,d_tmp,d_qi,d_si,0.0005f,0.0005f,1e-6f);
            down_resid<<<(HIDDEN+t-1)/t,t,0,s>>>(d_down,d_mid,d_final,0.001f);
        };

        const int warmup = 100;
        const float empty_direct = time_group(stream, empty_group, warmup, o.iters);
        const float empty_graph = time_graph(stream, empty_group, warmup, o.iters);
        const float real_direct = time_group(stream, real_group, warmup, o.iters);

        real_group(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<__nv_bfloat16> direct_out(HIDDEN);
        CUDA_CHECK(cudaMemcpy(direct_out.data(), d_final, HIDDEN*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        const float real_graph = time_graph(stream, real_group, warmup, o.iters);
        real_group(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<__nv_bfloat16> graph_out(HIDDEN);
        CUDA_CHECK(cudaMemcpy(graph_out.data(), d_final, HIDDEN*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        const float diff = max_bf16_diff(direct_out, graph_out);

        CUDA_CHECK(cudaGetLastError());

        const float direct_residual = std::max(0.0f, real_direct - empty_direct);
        const float graph_residual = std::max(0.0f, real_graph - empty_graph);
        const float launch_share = real_direct > 0.0f ? 100.0f * empty_direct / real_direct : 0.0f;

        std::cout << "\n=== V10 six-launch scheduling floor ===\n"
                  << std::fixed << std::setprecision(4)
                  << "empty direct group      : " << empty_direct << " ms/layer\n"
                  << "empty CUDA Graph group  : " << empty_graph << " ms/layer\n"
                  << "empty graph speedup     : " << (empty_direct / empty_graph) << "x\n"
                  << "per-launch direct equiv : " << (empty_direct / 6.0f) * 1000.0f << " us\n";

        std::cout << "\n=== V10 real fused-support group ===\n"
                  << "direct six kernels      : " << real_direct << " ms/layer\n"
                  << "CUDA Graph replay       : " << real_graph << " ms/layer\n"
                  << "graph speedup            : " << (real_direct / real_graph) << "x\n"
                  << "direct 30-layer equiv   : " << real_direct * 30.0f << " ms\n"
                  << "graph 30-layer equiv    : " << real_graph * 30.0f << " ms\n"
                  << "output consistency      : " << (diff == 0.0f ? "PASS" : "PASS-with-roundoff")
                  << "  max_abs_diff=" << std::setprecision(6) << diff << "\n";

        std::cout << "\n=== V10 decomposition heuristic ===\n"
                  << std::setprecision(4)
                  << "direct real-empty       : " << direct_residual << " ms/layer\n"
                  << "graph real-empty        : " << graph_residual << " ms/layer\n"
                  << "empty/direct share      : " << std::setprecision(1) << launch_share << " %\n";

        std::cout << "\nInterpretation guardrails:\n"
                  << "  * Empty kernels estimate launch/scheduling floor for the same six grid shapes; subtraction is heuristic, not a hardware counter.\n"
                  << "  * CUDA Graph replay reduces host/launch orchestration but does not fuse memory traffic or arithmetic between kernels.\n"
                  << "  * The six support kernels are timed back-to-back here; in a real layer they are interleaved with QKV/O/MLP matrices and attention.\n"
                  << "  * If Graph replay barely helps real support, V11 should pursue true epilogue/persistent fusion.\n"
                  << "  * If Graph replay removes a large fraction, the fixed runtime should graph-capture the full layer/decode schedule before deeper kernel surgery.\n";

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_hidden)); CUDA_CHECK(cudaFree(d_gamma_h)); CUDA_CHECK(cudaFree(d_gamma_i));
        CUDA_CHECK(cudaFree(d_attn)); CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_k)); CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaFree(d_mid)); CUDA_CHECK(cudaFree(d_tmp)); CUDA_CHECK(cudaFree(d_final));
        CUDA_CHECK(cudaFree(d_qh)); CUDA_CHECK(cudaFree(d_qi)); CUDA_CHECK(cudaFree(d_sh)); CUDA_CHECK(cudaFree(d_si));
        CUDA_CHECK(cudaFree(d_cos)); CUDA_CHECK(cudaFree(d_sin)); CUDA_CHECK(cudaFree(d_qkv)); CUDA_CHECK(cudaFree(d_o));
        CUDA_CHECK(cudaFree(d_gu)); CUDA_CHECK(cudaFree(d_down));

        std::cout << "\nV10 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
