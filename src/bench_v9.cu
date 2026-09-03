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

static constexpr int Q_HEADS = 20;
static constexpr int KV_HEADS = 5;
static constexpr int HEAD_DIM = 128;
static constexpr int Q_DIM = Q_HEADS * HEAD_DIM;
static constexpr int KV_DIM = KV_HEADS * HEAD_DIM;
static constexpr int LAYERS = 30;
static constexpr float SCALE = 0.08838834764831845f; // 1/sqrt(128)
static constexpr float NEG_SENTINEL = -3.402823466e+38F;
// V8 non-attention GPU accounting from measured components:
// V6 ternary linears 2.1265 + fused support 1.8712 + final norm 0.0080 + LM head 0.9166.
static constexpr float V8_FIXED_GPU_MS = 4.9223f;

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
            std::cout << "GA102-ROM V9 split-context exact decode attention\n"
                      << "  --iters N        timing iterations (default 300)\n"
                      << "  --max-context N  maximum context, 128..8192 (default 4096)\n";
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

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 b16(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
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
    float x = (threadIdx.x < (blockDim.x >> 5)) ? s[lane] : NEG_SENTINEL;
    if (warp == 0) x = warp_max(x);
    __syncthreads();
    if (threadIdx.x == 0) s[0] = x;
    __syncthreads();
    return s[0];
}

// V8 baseline: one CUDA block owns one Q head and traverses the entire context.
__global__ void attention_qhead_baseline(const __nv_bfloat16* q,
                                         const __nv_bfloat16* kc,
                                         const __nv_bfloat16* vc,
                                         __nv_bfloat16* out,
                                         int L) {
    extern __shared__ float score[];
    __shared__ float s[8], invsum;
    const int qh = blockIdx.x;
    const int kvh = qh >> 2;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nw = blockDim.x >> 5;

    for (int p = warp; p < L; p += nw) {
        float acc = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += 32)
            acc += bf(q[qh * HEAD_DIM + d]) * bf(kc[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
        acc = warp_sum(acc);
        if (lane == 0) score[p] = acc * SCALE;
    }
    __syncthreads();

    float lm = NEG_SENTINEL;
    for (int p = threadIdx.x; p < L; p += blockDim.x) lm = fmaxf(lm, score[p]);
    const float mx = block_max(lm, s);

    float ls = 0.0f;
    for (int p = threadIdx.x; p < L; p += blockDim.x) {
        const float e = expf(score[p] - mx);
        score[p] = e;
        ls += e;
    }
    const float sm = block_sum(ls, s);
    if (threadIdx.x == 0) invsum = 1.0f / sm;
    __syncthreads();

    if (threadIdx.x < HEAD_DIM) {
        const int d = threadIdx.x;
        float acc = 0.0f;
        for (int p = 0; p < L; ++p)
            acc += score[p] * invsum * bf(vc[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
        out[qh * HEAD_DIM + d] = b16(acc);
    }
}

// Split-Q partial: one block owns one Q head and one context chunk.
// It emits numerically stable softmax statistics (m_i, l_i) and an UNNORMALIZED
// weighted-V numerator. A second kernel merges chunks exactly using log-sum-exp rules.
__global__ void attention_split_q_partial(const __nv_bfloat16* q,
                                          const __nv_bfloat16* kc,
                                          const __nv_bfloat16* vc,
                                          float* part_m,
                                          float* part_l,
                                          float* part_o,
                                          int L,
                                          int chunk,
                                          int parts) {
    extern __shared__ float score[];
    __shared__ float s[8], smx, sl;

    const int qh = blockIdx.x / parts;
    const int part = blockIdx.x - qh * parts;
    if (qh >= Q_HEADS) return;
    const int kvh = qh >> 2;
    const int start = part * chunk;
    const int len = min(chunk, L - start);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nw = blockDim.x >> 5;

    for (int lp = warp; lp < len; lp += nw) {
        const int p = start + lp;
        float acc = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += 32)
            acc += bf(q[qh * HEAD_DIM + d]) * bf(kc[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
        acc = warp_sum(acc);
        if (lane == 0) score[lp] = acc * SCALE;
    }
    __syncthreads();

    float lm = NEG_SENTINEL;
    for (int lp = threadIdx.x; lp < len; lp += blockDim.x) lm = fmaxf(lm, score[lp]);
    const float mx = block_max(lm, s);

    float ls = 0.0f;
    for (int lp = threadIdx.x; lp < len; lp += blockDim.x) {
        const float e = expf(score[lp] - mx);
        score[lp] = e;
        ls += e;
    }
    const float sum = block_sum(ls, s);
    if (threadIdx.x == 0) { smx = mx; sl = sum; }
    __syncthreads();

    const size_t pi = (size_t)qh * parts + part;
    if (threadIdx.x == 0) {
        part_m[pi] = smx;
        part_l[pi] = sl;
    }
    if (threadIdx.x < HEAD_DIM) {
        const int d = threadIdx.x;
        float acc = 0.0f;
        for (int lp = 0; lp < len; ++lp) {
            const int p = start + lp;
            acc += score[lp] * bf(vc[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
        }
        part_o[pi * HEAD_DIM + d] = acc;
    }
}

// Split-GQA4 partial: one block owns one KV head + one context chunk and computes
// all four Q heads that share that KV head. K and V are therefore reused 4x inside the block.
__global__ void attention_split_gqa4_partial(const __nv_bfloat16* q,
                                             const __nv_bfloat16* kc,
                                             const __nv_bfloat16* vc,
                                             float* part_m,
                                             float* part_l,
                                             float* part_o,
                                             int L,
                                             int chunk,
                                             int parts) {
    extern __shared__ float score[]; // [4][chunk]
    __shared__ float s[8], local_m[4], local_l[4];

    const int kvh = blockIdx.x / parts;
    const int part = blockIdx.x - kvh * parts;
    if (kvh >= KV_HEADS) return;
    const int qb = kvh * 4;
    const int start = part * chunk;
    const int len = min(chunk, L - start);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nw = blockDim.x >> 5;

    for (int lp = warp; lp < len; lp += nw) {
        const int p = start + lp;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += 32) {
            const float k = bf(kc[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
            a0 += bf(q[(qb + 0) * HEAD_DIM + d]) * k;
            a1 += bf(q[(qb + 1) * HEAD_DIM + d]) * k;
            a2 += bf(q[(qb + 2) * HEAD_DIM + d]) * k;
            a3 += bf(q[(qb + 3) * HEAD_DIM + d]) * k;
        }
        a0 = warp_sum(a0); a1 = warp_sum(a1); a2 = warp_sum(a2); a3 = warp_sum(a3);
        if (lane == 0) {
            score[0 * chunk + lp] = a0 * SCALE;
            score[1 * chunk + lp] = a1 * SCALE;
            score[2 * chunk + lp] = a2 * SCALE;
            score[3 * chunk + lp] = a3 * SCALE;
        }
    }
    __syncthreads();

#pragma unroll
    for (int h = 0; h < 4; ++h) {
        float* z = score + (size_t)h * chunk;
        float lm = NEG_SENTINEL;
        for (int lp = threadIdx.x; lp < len; lp += blockDim.x) lm = fmaxf(lm, z[lp]);
        const float mx = block_max(lm, s);
        float ls = 0.0f;
        for (int lp = threadIdx.x; lp < len; lp += blockDim.x) {
            const float e = expf(z[lp] - mx);
            z[lp] = e;
            ls += e;
        }
        const float sum = block_sum(ls, s);
        if (threadIdx.x == 0) { local_m[h] = mx; local_l[h] = sum; }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const size_t pi = (size_t)(qb + h) * parts + part;
            part_m[pi] = local_m[h];
            part_l[pi] = local_l[h];
        }
    }

    if (threadIdx.x < HEAD_DIM) {
        const int d = threadIdx.x;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int lp = 0; lp < len; ++lp) {
            const int p = start + lp;
            const float v = bf(vc[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
            a0 += score[0 * chunk + lp] * v;
            a1 += score[1 * chunk + lp] * v;
            a2 += score[2 * chunk + lp] * v;
            a3 += score[3 * chunk + lp] * v;
        }
        part_o[((size_t)(qb + 0) * parts + part) * HEAD_DIM + d] = a0;
        part_o[((size_t)(qb + 1) * parts + part) * HEAD_DIM + d] = a1;
        part_o[((size_t)(qb + 2) * parts + part) * HEAD_DIM + d] = a2;
        part_o[((size_t)(qb + 3) * parts + part) * HEAD_DIM + d] = a3;
    }
}

// Numerically stable merge of chunk-softmax states:
// m = max_i(m_i)
// l = sum_i l_i * exp(m_i-m)
// O = sum_i O_i * exp(m_i-m) / l
__global__ void attention_split_merge(const float* part_m,
                                      const float* part_l,
                                      const float* part_o,
                                      __nv_bfloat16* out,
                                      int parts) {
    __shared__ float gm, gl;
    const int qh = blockIdx.x;
    if (qh >= Q_HEADS) return;

    if (threadIdx.x == 0) {
        float m = NEG_SENTINEL;
        for (int p = 0; p < parts; ++p)
            m = fmaxf(m, part_m[(size_t)qh * parts + p]);
        float l = 0.0f;
        for (int p = 0; p < parts; ++p) {
            const size_t pi = (size_t)qh * parts + p;
            l += part_l[pi] * expf(part_m[pi] - m);
        }
        gm = m;
        gl = l;
    }
    __syncthreads();

    if (threadIdx.x < HEAD_DIM) {
        const int d = threadIdx.x;
        float acc = 0.0f;
        for (int p = 0; p < parts; ++p) {
            const size_t pi = (size_t)qh * parts + p;
            acc += part_o[pi * HEAD_DIM + d] * expf(part_m[pi] - gm);
        }
        out[qh * HEAD_DIM + d] = b16(acc / gl);
    }
}

static void cpu_attention_head0(const std::vector<__nv_bfloat16>& q,
                                const std::vector<__nv_bfloat16>& k,
                                const std::vector<__nv_bfloat16>& v,
                                std::vector<float>& out,
                                int L) {
    std::vector<float> score(L);
    float mx = -std::numeric_limits<float>::infinity();
    for (int p = 0; p < L; ++p) {
        float acc = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d)
            acc += __bfloat162float(q[d]) * __bfloat162float(k[(size_t)p * KV_DIM + d]);
        score[p] = acc * SCALE;
        mx = std::max(mx, score[p]);
    }
    float sum = 0.0f;
    for (float& x : score) { x = std::exp(x - mx); sum += x; }
    for (float& x : score) x /= sum;
    out.assign(HEAD_DIM, 0.0f);
    for (int d = 0; d < HEAD_DIM; ++d) {
        float acc = 0.0f;
        for (int p = 0; p < L; ++p)
            acc += score[p] * __bfloat162float(v[(size_t)p * KV_DIM + d]);
        out[d] = acc;
    }
}

static float max_abs_diff(const std::vector<__nv_bfloat16>& a,
                          const std::vector<__nv_bfloat16>& b) {
    float e = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::fabs(__bfloat162float(a[i]) - __bfloat162float(b[i])));
    return e;
}

struct Candidate {
    std::string name;
    float ms = 1e30f;
    float diff = 0.0f;
};

int main(int argc, char** argv) {
    try {
        const Options o = parse_args(argc, argv);
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 8 || prop.minor != 6 || std::string(prop.name).find("RTX 3080") == std::string::npos) {
            std::cerr << "V9 restricted to RTX 3080 / SM86; found " << prop.name << "\n";
            return 3;
        }

        std::cout << "GA102-ROM V9: split-context exact GQA decode attention\n"
                  << "GPU               : " << prop.name << "\n"
                  << "model heads       : Q=20 KV=5 dim=128\n"
                  << "baseline          : V8 one block / Q head\n"
                  << "experiment        : split context + stable softmax merge\n"
                  << "chunk sweep       : 128 / 256 / 512 tokens\n"
                  << "V8 fixed GPU cost : " << V8_FIXED_GPU_MS << " ms/token excluding attention\n";

        std::vector<__nv_bfloat16> hq(Q_DIM);
        std::vector<__nv_bfloat16> hk((size_t)o.max_context * KV_DIM);
        std::vector<__nv_bfloat16> hv((size_t)o.max_context * KV_DIM);
        for (int h = 0; h < Q_HEADS; ++h)
            for (int d = 0; d < HEAD_DIM; ++d)
                hq[h * HEAD_DIM + d] = __float2bfloat16((float)(((h * 131 + d * 17) % 101) - 50) / 80.0f);
        for (int p = 0; p < o.max_context; ++p)
            for (int h = 0; h < KV_HEADS; ++h)
                for (int d = 0; d < HEAD_DIM; ++d) {
                    const int z = (p * 29 + h * 37 + d * 11) % 127;
                    const int w = (p * 17 + h * 19 + d * 23) % 113;
                    hk[(size_t)p * KV_DIM + h * HEAD_DIM + d] = __float2bfloat16((float)(z - 63) / 96.0f);
                    hv[(size_t)p * KV_DIM + h * HEAD_DIM + d] = __float2bfloat16((float)(w - 56) / 88.0f);
                }

        __nv_bfloat16 *dq = nullptr, *dk = nullptr, *dv = nullptr;
        __nv_bfloat16 *d_base = nullptr, *d_split = nullptr;
        float *d_pm = nullptr, *d_pl = nullptr, *d_po = nullptr;

        CUDA_CHECK(cudaMalloc(&dq, Q_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dk, hk.size() * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&dv, hv.size() * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_base, Q_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_split, Q_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(dq, hq.data(), Q_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dk, hk.data(), hk.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv, hv.data(), hv.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        const int max_parts = (o.max_context + 127) / 128;
        const size_t max_partial_count = (size_t)Q_HEADS * max_parts;
        CUDA_CHECK(cudaMalloc(&d_pm, max_partial_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pl, max_partial_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_po, max_partial_count * HEAD_DIM * sizeof(float)));

        // Baseline correctness against CPU at context 128.
        attention_qhead_baseline<<<Q_HEADS, 256, 128 * sizeof(float)>>>(dq, dk, dv, d_base, 128);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> baseline(Q_DIM);
        CUDA_CHECK(cudaMemcpy(baseline.data(), d_base, Q_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        std::vector<float> cpu;
        cpu_attention_head0(hq, hk, hv, cpu, 128);
        float cpu_err = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d)
            cpu_err = std::max(cpu_err, std::fabs(cpu[d] - __bfloat162float(baseline[d])));
        if (cpu_err >= 0.03f) {
            std::cerr << "V9 baseline CPU correctness failed: max_abs_err=" << cpu_err << "\n";
            return 4;
        }

        std::cout << "\n=== V9 correctness anchor ===\n"
                  << "baseline head0@128 vs CPU : PASS  max_abs_err=" << std::fixed << std::setprecision(6) << cpu_err << "\n";

        const std::array<int, 5> contexts{{128, 512, 1024, 2048, 4096}};
        const std::array<int, 3> chunks{{128, 256, 512}};

        std::cout << "\n=== V9 split-context attention sweep ===\n";
        for (int L : contexts) {
            if (L > o.max_context) continue;

            auto baseline_call = [&] {
                attention_qhead_baseline<<<Q_HEADS, 256, (size_t)L * sizeof(float)>>>(dq, dk, dv, d_base, L);
            };
            const float baseline_ms = time_ms(baseline_call, 20, o.iters);
            baseline_call();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(baseline.data(), d_base, Q_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

            Candidate best{"baseline", baseline_ms, 0.0f};
            std::cout << "\nctx " << L << "  baseline=" << std::fixed << std::setprecision(4) << baseline_ms << " ms/layer\n";

            for (int chunk : chunks) {
                const int parts = (L + chunk - 1) / chunk;

                auto split_q_call = [&] {
                    attention_split_q_partial<<<Q_HEADS * parts, 256, (size_t)chunk * sizeof(float)>>>(
                        dq, dk, dv, d_pm, d_pl, d_po, L, chunk, parts);
                    attention_split_merge<<<Q_HEADS, 128>>>(d_pm, d_pl, d_po, d_split, parts);
                };
                const float sq_ms = time_ms(split_q_call, 20, o.iters);
                split_q_call();
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<__nv_bfloat16> split(Q_DIM);
                CUDA_CHECK(cudaMemcpy(split.data(), d_split, Q_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                const float sq_diff = max_abs_diff(baseline, split);
                const bool sq_ok = sq_diff < 0.01f;

                auto split_g4_call = [&] {
                    attention_split_gqa4_partial<<<KV_HEADS * parts, 256, (size_t)4 * chunk * sizeof(float)>>>(
                        dq, dk, dv, d_pm, d_pl, d_po, L, chunk, parts);
                    attention_split_merge<<<Q_HEADS, 128>>>(d_pm, d_pl, d_po, d_split, parts);
                };
                const float sg_ms = time_ms(split_g4_call, 20, o.iters);
                split_g4_call();
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(split.data(), d_split, Q_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                const float sg_diff = max_abs_diff(baseline, split);
                const bool sg_ok = sg_diff < 0.01f;

                std::cout << "  chunk " << std::setw(3) << chunk
                          << "  parts=" << std::setw(2) << parts
                          << "  split-Q=" << std::setprecision(4) << sq_ms << " ms"
                          << " (" << (sq_ok ? "PASS" : "FAIL") << ", diff=" << std::setprecision(5) << sq_diff << ")"
                          << "  split-GQA4=" << std::setprecision(4) << sg_ms << " ms"
                          << " (" << (sg_ok ? "PASS" : "FAIL") << ", diff=" << std::setprecision(5) << sg_diff << ")\n";

                if (sq_ok && sq_ms < best.ms) best = {"split-Q/" + std::to_string(chunk), sq_ms, sq_diff};
                if (sg_ok && sg_ms < best.ms) best = {"split-GQA4/" + std::to_string(chunk), sg_ms, sg_diff};
            }

            const float speedup = baseline_ms / best.ms;
            const float attention_30 = best.ms * LAYERS;
            const float accounting = V8_FIXED_GPU_MS + attention_30;
            std::cout << "  BEST: " << best.name
                      << "  " << std::setprecision(4) << best.ms << " ms/layer"
                      << "  speedup=" << std::setprecision(2) << speedup << "x"
                      << "  attention-30L=" << std::setprecision(4) << attention_30 << " ms\n"
                      << "        V9 GPU accounting: " << accounting << " ms => "
                      << std::setprecision(1) << (1000.0f / accounting) << " tok/s ceiling\n";
        }

        std::cout << "\nGuardrails:\n"
                  << "  * Split kernels perform the same QK scaling, softmax and weighted-V math as the V8 baseline.\n"
                  << "  * Chunk results are merged with numerically stable log-sum-exp softmax composition.\n"
                  << "  * PASS compares BF16 outputs against the V8 baseline; CPU head0@128 anchors baseline correctness.\n"
                  << "  * V9 accounting reuses V8 measured non-attention GPU cost (4.9223 ms).\n"
                  << "  * This is still a GPU-kernel accounting ceiling, not end-to-end text generation throughput.\n";

        CUDA_CHECK(cudaFree(dq));
        CUDA_CHECK(cudaFree(dk));
        CUDA_CHECK(cudaFree(dv));
        CUDA_CHECK(cudaFree(d_base));
        CUDA_CHECK(cudaFree(d_split));
        CUDA_CHECK(cudaFree(d_pm));
        CUDA_CHECK(cudaFree(d_pl));
        CUDA_CHECK(cudaFree(d_po));

        std::cout << "\nV9 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
