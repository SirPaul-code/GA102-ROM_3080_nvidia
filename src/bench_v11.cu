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

using bf16 = __nv_bfloat16;

static constexpr int HIDDEN = 2560;
static constexpr int INTER = 6912;
static constexpr int Q_HEADS = 20;
static constexpr int KV_HEADS = 5;
static constexpr int HEAD_DIM = 128;
static constexpr int Q_DIM = Q_HEADS * HEAD_DIM;
static constexpr int KV_DIM = KV_HEADS * HEAD_DIM;
static constexpr int QKV_DIM = Q_DIM + 2 * KV_DIM;
static constexpr int GATEUP_DIM = 2 * INTER;
static constexpr int VOCAB = 128256;
static constexpr int LAYERS = 30;
static constexpr float ATTN_SCALE = 0.08838834764831845f;
static constexpr float NEG_SENTINEL = -3.402823466e+38F;

struct Options {
    int iters = 50;
    int ring_mib = 64;
    int max_context = 4096;
};

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "--ring-mib" && i + 1 < argc) o.ring_mib = std::stoi(argv[++i]);
        else if (a == "--max-context" && i + 1 < argc) o.max_context = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V11 full synthetic decoder CUDA Graph benchmark\n"
                      << "  --iters N        iterations/context (default 50)\n"
                      << "  --ring-mib N     packed-weight ring per projection (default 64 MiB)\n"
                      << "  --max-context N  128..4096 (default 4096)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + a);
        }
    }
    if (o.iters <= 0 || o.ring_mib < 16 || o.max_context < 128 || o.max_context > 4096)
        throw std::runtime_error("iters>0, ring-mib>=16, max-context=128..4096 required");
    return o;
}

__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 b16(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ int warp_sum_i32(int v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

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
    float x = (threadIdx.x < (blockDim.x >> 5)) ? s[lane] : NEG_SENTINEL;
    if (warp == 0) x = warp_max(x);
    __syncthreads();
    if (threadIdx.x == 0) s[0] = x;
    __syncthreads();
    return s[0];
}

__global__ void init_bf16(bf16* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        uint32_t z = (uint32_t)i * 747796405u + seed;
        z = ((z >> ((z >> 28) + 4)) ^ z) * 277803737u;
        z = (z >> 22) ^ z;
        const float v = ((float)(z & 0xffffu) / 32767.5f) - 1.0f;
        p[i] = b16(v);
    }
}

__global__ void init_masks(uint32_t* p, uint32_t* n, size_t count, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < count; i += stride) {
        uint32_t z = (uint32_t)i * 747796405u + seed;
        z = ((z >> ((z >> 28) + 4)) ^ z) * 277803737u;
        z = (z >> 22) ^ z;
        uint32_t q = z * 1664525u + 1013904223u;
        p[i] = z;
        n[i] = q & ~z;
    }
}

__global__ void copy_bf16(const bf16* src, bf16* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void pack_a8_to_planes(const int8_t* x, uint32_t* xb, int words) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int word = tid >> 5;
    const int lane = threadIdx.x & 31;
    if (word >= words) return;
    const uint8_t u = (uint8_t)x[word * 32 + lane];
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        const uint32_t mask = __ballot_sync(0xffffffffu, ((u >> bit) & 1u) != 0u);
        if (lane == 0) xb[(size_t)bit * words + word] = mask;
    }
}

__global__ void popc_single_warp(const uint32_t* pos, const uint32_t* neg,
                                 const uint32_t* xb, int32_t* y, int M, int words) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = tid >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;
    const uint32_t* pr = pos + (size_t)row * words;
    const uint32_t* nr = neg + (size_t)row * words;
    int acc = 0;
    for (int j = lane; j < words; j += 32) {
        const uint32_t p = pr[j];
        const uint32_t n = nr[j];
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t a = xb[(size_t)bit * words + j];
            const int cnt = __popc(p & a) - __popc(n & a);
            const int scale = (bit == 7) ? -128 : (1 << bit);
            acc += cnt * scale;
        }
    }
    acc = warp_sum_i32(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void rmsnorm_quant_a8(const bf16* x, const bf16* gamma,
                                 int8_t* q, float* qscale, int n, float eps) {
    __shared__ float s[8], invr, scale;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = bf(x[i]); ss += v * v; }
    const float sum = block_sum(ss, s);
    if (threadIdx.x == 0) invr = rsqrtf(sum / (float)n + eps);
    __syncthreads();
    float lm = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        lm = fmaxf(lm, fabsf(bf(x[i]) * invr * bf(gamma[i])));
    const float mx = block_max(lm, s);
    if (threadIdx.x == 0) { scale = 127.0f / fmaxf(mx, 1e-5f); qscale[0] = scale; }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int z = __float2int_rn(bf(x[i]) * invr * bf(gamma[i]) * scale);
        z = max(-128, min(127, z)); q[i] = (int8_t)z;
    }
}

__global__ void rmsnorm_bf16(const bf16* x, const bf16* gamma, bf16* y, int n, float eps) {
    __shared__ float s[8], invr;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = bf(x[i]); ss += v * v; }
    const float sum = block_sum(ss, s);
    if (threadIdx.x == 0) invr = rsqrtf(sum / (float)n + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = b16(bf(x[i]) * invr * bf(gamma[i]));
}

__global__ void qkv_epilogue_rope_kv(const int32_t* qkv, bf16* qout, bf16* kc, bf16* vc,
                                     const float* cs, const float* sn, int pos,
                                     float qs, float ks, float vs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= QKV_DIM) return;
    if (i < Q_DIM) {
        int d = i & 127, base = i - d, od = d < 64 ? d + 64 : d - 64;
        float x = (float)qkv[i] * qs, xo = (float)qkv[base + od] * qs;
        qout[i] = b16(x * cs[d] + (d < 64 ? -xo : xo) * sn[d]);
    } else if (i < Q_DIM + KV_DIM) {
        int j = i - Q_DIM, d = j & 127, base = i - d, od = d < 64 ? d + 64 : d - 64;
        float x = (float)qkv[i] * ks, xo = (float)qkv[base + od] * ks;
        kc[(size_t)pos * KV_DIM + j] = b16(x * cs[d] + (d < 64 ? -xo : xo) * sn[d]);
    } else {
        int j = i - Q_DIM - KV_DIM;
        vc[(size_t)pos * KV_DIM + j] = b16((float)qkv[i] * vs);
    }
}

__global__ void o_resid_norm_quant(const int32_t* oacc, const bf16* residual,
                                   const bf16* gamma, bf16* mid,
                                   int8_t* q, float* qscale, float os, float eps) {
    __shared__ float s[8], invr, scale;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float v = bf(residual[i]) + (float)oacc[i] * os; mid[i] = b16(v); ss += v * v;
    }
    const float sum = block_sum(ss, s);
    if (threadIdx.x == 0) invr = rsqrtf(sum / (float)HIDDEN + eps);
    __syncthreads();
    float lm = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x)
        lm = fmaxf(lm, fabsf(bf(mid[i]) * invr * bf(gamma[i])));
    const float mx = block_max(lm, s);
    if (threadIdx.x == 0) { scale = 127.0f / fmaxf(mx, 1e-5f); qscale[0] = scale; }
    __syncthreads();
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        int z = __float2int_rn(bf(mid[i]) * invr * bf(gamma[i]) * scale);
        z = max(-128, min(127, z)); q[i] = (int8_t)z;
    }
}

__global__ void gateup_relu2_norm_quant(const int32_t* gu, const bf16* gamma,
                                        bf16* tmp, int8_t* q, float* qscale,
                                        float gs, float us, float eps) {
    __shared__ float s[8], invr, scale;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < INTER; i += blockDim.x) {
        float g = (float)gu[i] * gs, u = (float)gu[INTER + i] * us;
        float r = fmaxf(g, 0.0f), z = r * r * u; tmp[i] = b16(z); ss += z * z;
    }
    const float sum = block_sum(ss, s);
    if (threadIdx.x == 0) invr = rsqrtf(sum / (float)INTER + eps);
    __syncthreads();
    float lm = 0.0f;
    for (int i = threadIdx.x; i < INTER; i += blockDim.x)
        lm = fmaxf(lm, fabsf(bf(tmp[i]) * invr * bf(gamma[i])));
    const float mx = block_max(lm, s);
    if (threadIdx.x == 0) { scale = 127.0f / fmaxf(mx, 1e-5f); qscale[0] = scale; }
    __syncthreads();
    for (int i = threadIdx.x; i < INTER; i += blockDim.x) {
        int z = __float2int_rn(bf(tmp[i]) * invr * bf(gamma[i]) * scale);
        z = max(-128, min(127, z)); q[i] = (int8_t)z;
    }
}

__global__ void down_resid(const int32_t* down, const bf16* residual, bf16* out, float sc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN) out[i] = b16(bf(residual[i]) + (float)down[i] * sc);
}

__global__ void attention_qhead_baseline(const bf16* q, const bf16* kc, const bf16* vc,
                                         bf16* out, int L) {
    extern __shared__ float score[];
    __shared__ float s[8], invsum;
    int qh = blockIdx.x, kvh = qh >> 2, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;
    for (int p = warp; p < L; p += nw) {
        float acc = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += 32)
            acc += bf(q[qh * HEAD_DIM + d]) * bf(kc[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
        acc = warp_sum(acc); if (lane == 0) score[p] = acc * ATTN_SCALE;
    }
    __syncthreads();
    float lm = NEG_SENTINEL;
    for (int p = threadIdx.x; p < L; p += blockDim.x) lm = fmaxf(lm, score[p]);
    float mx = block_max(lm, s), ls = 0.0f;
    for (int p = threadIdx.x; p < L; p += blockDim.x) { float e = expf(score[p] - mx); score[p] = e; ls += e; }
    float sm = block_sum(ls, s); if (threadIdx.x == 0) invsum = 1.0f / sm; __syncthreads();
    if (threadIdx.x < HEAD_DIM) {
        int d = threadIdx.x; float acc = 0.0f;
        for (int p = 0; p < L; ++p) acc += score[p] * invsum * bf(vc[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
        out[qh * HEAD_DIM + d] = b16(acc);
    }
}

__global__ void attention_split_q_partial(const bf16* q, const bf16* kc, const bf16* vc,
                                          float* pm, float* pl, float* po,
                                          int L, int chunk, int parts) {
    extern __shared__ float score[];
    __shared__ float s[8], smx, sl;
    int qh = blockIdx.x / parts, part = blockIdx.x - qh * parts;
    if (qh >= Q_HEADS) return;
    int kvh = qh >> 2, start = part * chunk, len = min(chunk, L - start);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int lp = warp; lp < len; lp += nw) {
        int p = start + lp; float acc = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += 32)
            acc += bf(q[qh * HEAD_DIM + d]) * bf(kc[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
        acc = warp_sum(acc); if (lane == 0) score[lp] = acc * ATTN_SCALE;
    }
    __syncthreads();
    float lm = NEG_SENTINEL;
    for (int lp = threadIdx.x; lp < len; lp += blockDim.x) lm = fmaxf(lm, score[lp]);
    float mx = block_max(lm, s), ls = 0.0f;
    for (int lp = threadIdx.x; lp < len; lp += blockDim.x) { float e = expf(score[lp] - mx); score[lp] = e; ls += e; }
    float sum = block_sum(ls, s); if (threadIdx.x == 0) { smx = mx; sl = sum; } __syncthreads();
    size_t pi = (size_t)qh * parts + part;
    if (threadIdx.x == 0) { pm[pi] = smx; pl[pi] = sl; }
    if (threadIdx.x < HEAD_DIM) {
        int d = threadIdx.x; float acc = 0.0f;
        for (int lp = 0; lp < len; ++lp) {
            int p = start + lp; acc += score[lp] * bf(vc[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
        }
        po[pi * HEAD_DIM + d] = acc;
    }
}

__global__ void attention_split_gqa4_partial(const bf16* q, const bf16* kc, const bf16* vc,
                                             float* pm, float* pl, float* po,
                                             int L, int chunk, int parts) {
    extern __shared__ float score[];
    __shared__ float s[8], local_m[4], local_l[4];
    int kvh = blockIdx.x / parts, part = blockIdx.x - kvh * parts;
    if (kvh >= KV_HEADS) return;
    int qb = kvh * 4, start = part * chunk, len = min(chunk, L - start);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int lp = warp; lp < len; lp += nw) {
        int p = start + lp; float a0=0,a1=0,a2=0,a3=0;
        for (int d = lane; d < HEAD_DIM; d += 32) {
            float k = bf(kc[(size_t)p * KV_DIM + kvh * HEAD_DIM + d]);
            a0 += bf(q[(qb+0)*HEAD_DIM+d])*k; a1 += bf(q[(qb+1)*HEAD_DIM+d])*k;
            a2 += bf(q[(qb+2)*HEAD_DIM+d])*k; a3 += bf(q[(qb+3)*HEAD_DIM+d])*k;
        }
        a0=warp_sum(a0);a1=warp_sum(a1);a2=warp_sum(a2);a3=warp_sum(a3);
        if (lane == 0) { score[lp]=a0*ATTN_SCALE; score[chunk+lp]=a1*ATTN_SCALE; score[2*chunk+lp]=a2*ATTN_SCALE; score[3*chunk+lp]=a3*ATTN_SCALE; }
    }
    __syncthreads();
#pragma unroll
    for (int h=0; h<4; ++h) {
        float* z = score + (size_t)h * chunk; float lm = NEG_SENTINEL;
        for (int lp=threadIdx.x; lp<len; lp+=blockDim.x) lm=fmaxf(lm,z[lp]);
        float mx=block_max(lm,s), ls=0.0f;
        for (int lp=threadIdx.x; lp<len; lp+=blockDim.x) { float e=expf(z[lp]-mx); z[lp]=e; ls+=e; }
        float sum=block_sum(ls,s); if(threadIdx.x==0){local_m[h]=mx;local_l[h]=sum;} __syncthreads();
    }
    if(threadIdx.x==0) for(int h=0;h<4;++h){size_t pi=(size_t)(qb+h)*parts+part;pm[pi]=local_m[h];pl[pi]=local_l[h];}
    if(threadIdx.x<HEAD_DIM){
        int d=threadIdx.x;float a0=0,a1=0,a2=0,a3=0;
        for(int lp=0;lp<len;++lp){int p=start+lp;float v=bf(vc[(size_t)p*KV_DIM+kvh*HEAD_DIM+d]);a0+=score[lp]*v;a1+=score[chunk+lp]*v;a2+=score[2*chunk+lp]*v;a3+=score[3*chunk+lp]*v;}
        po[((size_t)(qb+0)*parts+part)*HEAD_DIM+d]=a0;po[((size_t)(qb+1)*parts+part)*HEAD_DIM+d]=a1;
        po[((size_t)(qb+2)*parts+part)*HEAD_DIM+d]=a2;po[((size_t)(qb+3)*parts+part)*HEAD_DIM+d]=a3;
    }
}

__global__ void attention_split_merge(const float* pm, const float* pl, const float* po,
                                      bf16* out, int parts) {
    __shared__ float gm, gl;
    int qh = blockIdx.x;
    if (threadIdx.x == 0) {
        float m = NEG_SENTINEL;
        for (int p=0;p<parts;++p) m=fmaxf(m,pm[(size_t)qh*parts+p]);
        float l=0.0f;
        for(int p=0;p<parts;++p){size_t pi=(size_t)qh*parts+p;l+=pl[pi]*expf(pm[pi]-m);} gm=m;gl=l;
    }
    __syncthreads();
    if(threadIdx.x<HEAD_DIM){int d=threadIdx.x;float acc=0.0f;for(int p=0;p<parts;++p){size_t pi=(size_t)qh*parts+p;acc+=po[pi*HEAD_DIM+d]*expf(pm[pi]-gm);}out[qh*HEAD_DIM+d]=b16(acc/gl);}
}

__global__ void lmhead_bf16_warp(const bf16* w, const bf16* x, float* logits, int vocab, int hidden) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x,row=tid>>5,lane=threadIdx.x&31;
    if(row>=vocab)return;const bf16* wr=w+(size_t)row*hidden;float acc=0.0f;
    for(int j=lane;j<hidden;j+=32)acc=fmaf(bf(wr[j]),bf(x[j]),acc);
    acc=warp_sum(acc);if(lane==0)logits[row]=acc;
}

struct MaskRing {
    const char* name;
    int M, K, words, copies;
    size_t words_per_copy;
    uint32_t *pos=nullptr,*neg=nullptr;

    void init(int ring_mib, uint32_t seed) {
        words = K / 32;
        words_per_copy = (size_t)M * words;
        size_t packed_bytes = 2 * words_per_copy * sizeof(uint32_t);
        size_t target = (size_t)ring_mib << 20;
        copies = (int)std::max<size_t>(1, (target + packed_bytes - 1) / packed_bytes);
        size_t total_words = words_per_copy * copies;
        CUDA_CHECK(cudaMalloc(&pos, total_words * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&neg, total_words * sizeof(uint32_t)));
        int blocks=(int)std::min<size_t>(65535,(total_words+255)/256);
        init_masks<<<blocks,256>>>(pos,neg,total_words,seed);
        CUDA_CHECK(cudaGetLastError());
    }
    const uint32_t* p(int layer) const { return pos + (size_t)(layer % copies) * words_per_copy; }
    const uint32_t* n(int layer) const { return neg + (size_t)(layer % copies) * words_per_copy; }
    double mib_total() const { return (double)(2 * words_per_copy * copies * sizeof(uint32_t)) / 1048576.0; }
    void release(){ if(pos)cudaFree(pos);if(neg)cudaFree(neg); }
};

struct Runtime {
    Options o;
    MaskRing qkv{"QKV",QKV_DIM,HIDDEN}, out{"O",HIDDEN,HIDDEN}, gu{"gate+up",GATEUP_DIM,HIDDEN}, down{"down",HIDDEN,INTER};
    bf16 *embed=nullptr,*hidden=nullptr,*gamma_h=nullptr,*gamma_i=nullptr,*q=nullptr,*attn=nullptr,*mid=nullptr,*tmpi=nullptr,*finalnorm=nullptr;
    bf16 *kcache=nullptr,*vcache=nullptr,*lmw=nullptr;
    int8_t *qh=nullptr,*qi=nullptr;
    uint32_t *planes_h=nullptr,*planes_i=nullptr;
    float *scale_h=nullptr,*scale_i=nullptr,*cs=nullptr,*sn=nullptr,*pm=nullptr,*pl=nullptr,*po=nullptr,*logits=nullptr;
    int32_t *qkv_acc=nullptr,*o_acc=nullptr,*gu_acc=nullptr,*down_acc=nullptr;
    size_t layer_kv_stride=0;

    explicit Runtime(const Options& opts):o(opts){
        qkv.init(o.ring_mib,0x1234u);out.init(o.ring_mib,0x2345u);gu.init(o.ring_mib,0x3456u);down.init(o.ring_mib,0x4567u);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMalloc(&embed,HIDDEN*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&hidden,HIDDEN*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&gamma_h,HIDDEN*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&gamma_i,INTER*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&q,Q_DIM*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&attn,HIDDEN*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&mid,HIDDEN*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&tmpi,INTER*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&finalnorm,HIDDEN*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&qh,HIDDEN));CUDA_CHECK(cudaMalloc(&qi,INTER));
        CUDA_CHECK(cudaMalloc(&planes_h,(size_t)8*(HIDDEN/32)*sizeof(uint32_t)));CUDA_CHECK(cudaMalloc(&planes_i,(size_t)8*(INTER/32)*sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&scale_h,sizeof(float)));CUDA_CHECK(cudaMalloc(&scale_i,sizeof(float)));
        CUDA_CHECK(cudaMalloc(&qkv_acc,QKV_DIM*sizeof(int32_t)));CUDA_CHECK(cudaMalloc(&o_acc,HIDDEN*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&gu_acc,GATEUP_DIM*sizeof(int32_t)));CUDA_CHECK(cudaMalloc(&down_acc,HIDDEN*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&cs,HEAD_DIM*sizeof(float)));CUDA_CHECK(cudaMalloc(&sn,HEAD_DIM*sizeof(float)));
        layer_kv_stride=(size_t)o.max_context*KV_DIM;
        CUDA_CHECK(cudaMalloc(&kcache,(size_t)LAYERS*layer_kv_stride*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&vcache,(size_t)LAYERS*layer_kv_stride*sizeof(bf16)));
        int max_parts=(o.max_context+127)/128;size_t partials=(size_t)Q_HEADS*max_parts;
        CUDA_CHECK(cudaMalloc(&pm,partials*sizeof(float)));CUDA_CHECK(cudaMalloc(&pl,partials*sizeof(float)));CUDA_CHECK(cudaMalloc(&po,partials*HEAD_DIM*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&lmw,(size_t)VOCAB*HIDDEN*sizeof(bf16)));CUDA_CHECK(cudaMalloc(&logits,VOCAB*sizeof(float)));

        init_bf16<<<128,256>>>(embed,HIDDEN,0x1111u);init_bf16<<<128,256>>>(gamma_h,HIDDEN,0x2222u);init_bf16<<<128,256>>>(gamma_i,INTER,0x3333u);
        init_bf16<<<65535,256>>>(kcache,(size_t)LAYERS*layer_kv_stride,0x4444u);init_bf16<<<65535,256>>>(vcache,(size_t)LAYERS*layer_kv_stride,0x5555u);
        init_bf16<<<65535,256>>>(lmw,(size_t)VOCAB*HIDDEN,0x6666u);
        std::vector<float> hc(HEAD_DIM),hs(HEAD_DIM);for(int d=0;d<HEAD_DIM;++d){float a=0.0007f*(d+1);hc[d]=std::cos(a);hs[d]=std::sin(a);}CUDA_CHECK(cudaMemcpy(cs,hc.data(),HEAD_DIM*sizeof(float),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(sn,hs.data(),HEAD_DIM*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~Runtime(){
        qkv.release();out.release();gu.release();down.release();
        cudaFree(embed);cudaFree(hidden);cudaFree(gamma_h);cudaFree(gamma_i);cudaFree(q);cudaFree(attn);cudaFree(mid);cudaFree(tmpi);cudaFree(finalnorm);
        cudaFree(kcache);cudaFree(vcache);cudaFree(lmw);cudaFree(qh);cudaFree(qi);cudaFree(planes_h);cudaFree(planes_i);cudaFree(scale_h);cudaFree(scale_i);
        cudaFree(cs);cudaFree(sn);cudaFree(pm);cudaFree(pl);cudaFree(po);cudaFree(logits);cudaFree(qkv_acc);cudaFree(o_acc);cudaFree(gu_acc);cudaFree(down_acc);
    }

    void linear(cudaStream_t s,const MaskRing& w,int layer,const uint32_t* planes,int32_t* y){
        int threads=128;int blocks=(w.M*32+threads-1)/threads;
        popc_single_warp<<<blocks,threads,0,s>>>(w.p(layer),w.n(layer),planes,y,w.M,w.words);
    }

    void pack_hidden(cudaStream_t s){int threads=256;int total=(HIDDEN/32)*32;pack_a8_to_planes<<<total/threads,threads,0,s>>>(qh,planes_h,HIDDEN/32);}
    void pack_inter(cudaStream_t s){int threads=256;int total=(INTER/32)*32;pack_a8_to_planes<<<total/threads,threads,0,s>>>(qi,planes_i,INTER/32);}

    void attention(cudaStream_t s,int layer,int L){
        bf16* kl=kcache+(size_t)layer*layer_kv_stride;bf16* vl=vcache+(size_t)layer*layer_kv_stride;
        if(L<=128){attention_qhead_baseline<<<Q_HEADS,256,(size_t)L*sizeof(float),s>>>(q,kl,vl,attn,L);return;}
        int chunk=128,parts=(L+chunk-1)/chunk;
        if(L<4096){
            attention_split_q_partial<<<Q_HEADS*parts,256,(size_t)chunk*sizeof(float),s>>>(q,kl,vl,pm,pl,po,L,chunk,parts);
        }else{
            attention_split_gqa4_partial<<<KV_HEADS*parts,256,(size_t)4*chunk*sizeof(float),s>>>(q,kl,vl,pm,pl,po,L,chunk,parts);
        }
        attention_split_merge<<<Q_HEADS,128,0,s>>>(pm,pl,po,attn,parts);
    }

    void launch(cudaStream_t s,int L){
        int t=256;
        copy_bf16<<<(HIDDEN+t-1)/t,t,0,s>>>(embed,hidden,HIDDEN);
        for(int layer=0;layer<LAYERS;++layer){
            rmsnorm_quant_a8<<<1,t,0,s>>>(hidden,gamma_h,qh,scale_h,HIDDEN,1e-6f);
            pack_hidden(s);linear(s,qkv,layer,planes_h,qkv_acc);
            bf16* kl=kcache+(size_t)layer*layer_kv_stride;bf16* vl=vcache+(size_t)layer*layer_kv_stride;
            qkv_epilogue_rope_kv<<<(QKV_DIM+t-1)/t,t,0,s>>>(qkv_acc,q,kl,vl,cs,sn,L-1,0.001f,0.001f,0.001f);
            attention(s,layer,L);
            rmsnorm_quant_a8<<<1,t,0,s>>>(attn,gamma_h,qh,scale_h,HIDDEN,1e-6f);
            pack_hidden(s);linear(s,out,layer,planes_h,o_acc);
            o_resid_norm_quant<<<1,t,0,s>>>(o_acc,hidden,gamma_h,mid,qh,scale_h,0.001f,1e-6f);
            pack_hidden(s);linear(s,gu,layer,planes_h,gu_acc);
            gateup_relu2_norm_quant<<<1,t,0,s>>>(gu_acc,gamma_i,tmpi,qi,scale_i,0.0005f,0.0005f,1e-6f);
            pack_inter(s);linear(s,down,layer,planes_i,down_acc);
            down_resid<<<(HIDDEN+t-1)/t,t,0,s>>>(down_acc,mid,hidden,0.001f);
        }
        rmsnorm_bf16<<<1,t,0,s>>>(hidden,gamma_h,finalnorm,HIDDEN,1e-6f);
        int lm_threads=128;int lm_blocks=(VOCAB*32+lm_threads-1)/lm_threads;
        lmhead_bf16_warp<<<lm_blocks,lm_threads,0,s>>>(lmw,finalnorm,logits,VOCAB,HIDDEN);
    }
};

static float time_direct(Runtime& r,cudaStream_t s,int L,int warmup,int iters){
    for(int i=0;i<warmup;++i)r.launch(s,L);CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a,b;CUDA_CHECK(cudaEventCreate(&a));CUDA_CHECK(cudaEventCreate(&b));CUDA_CHECK(cudaEventRecord(a,s));
    for(int i=0;i<iters;++i)r.launch(s,L);CUDA_CHECK(cudaEventRecord(b,s));CUDA_CHECK(cudaEventSynchronize(b));float total=0;CUDA_CHECK(cudaEventElapsedTime(&total,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return total/iters;
}

static float time_graph(Runtime& r,cudaStream_t s,int L,int warmup,int iters,cudaGraphExec_t* out_exec=nullptr){
    CUDA_CHECK(cudaStreamSynchronize(s));cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));r.launch(s,L);CUDA_CHECK(cudaStreamEndCapture(s,&g));CUDA_CHECK(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
    for(int i=0;i<warmup;++i)CUDA_CHECK(cudaGraphLaunch(e,s));CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a,b;CUDA_CHECK(cudaEventCreate(&a));CUDA_CHECK(cudaEventCreate(&b));CUDA_CHECK(cudaEventRecord(a,s));
    for(int i=0;i<iters;++i)CUDA_CHECK(cudaGraphLaunch(e,s));CUDA_CHECK(cudaEventRecord(b,s));CUDA_CHECK(cudaEventSynchronize(b));float total=0;CUDA_CHECK(cudaEventElapsedTime(&total,a,b));cudaEventDestroy(a);cudaEventDestroy(b);
    if(out_exec)*out_exec=e;else cudaGraphExecDestroy(e);cudaGraphDestroy(g);return total/iters;
}

int main(int argc,char**argv){
    try{
        Options o=parse_args(argc,argv);int dev=0;CUDA_CHECK(cudaGetDevice(&dev));cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
        if(prop.major!=8||prop.minor!=6||std::string(prop.name).find("RTX 3080")==std::string::npos){std::cerr<<"V11 restricted to RTX 3080 / SM86; found "<<prop.name<<"\n";return 3;}
        std::cout<<"GA102-ROM V11: full synthetic 30-layer decoder CUDA Graph\nGPU               : "<<prop.name
                 <<"\nmodel shapes      : microsoft/bitnet-b1.58-2B-4T"
                 <<"\npath              : POPC linears + V9 attention + fused support + BF16 LM head"
                 <<"\nweight rings      : ~"<<o.ring_mib<<" MiB per projection family"
                 <<"\nLM head           : 128256 x 2560 BF16 (626.25 MiB)"
                 <<"\ncomparison        : identical direct launch schedule vs one CUDA Graph replay\n";
        Runtime r(o);cudaStream_t s;CUDA_CHECK(cudaStreamCreate(&s));
        std::cout<<std::fixed<<std::setprecision(3)
                 <<"packed ring QKV    : "<<r.qkv.mib_total()<<" MiB (copies="<<r.qkv.copies<<")\n"
                 <<"packed ring O      : "<<r.out.mib_total()<<" MiB (copies="<<r.out.copies<<")\n"
                 <<"packed ring gate/up: "<<r.gu.mib_total()<<" MiB (copies="<<r.gu.copies<<")\n"
                 <<"packed ring down   : "<<r.down.mib_total()<<" MiB (copies="<<r.down.copies<<")\n";

        std::array<int,5> ctxs{{128,512,1024,2048,4096}};
        std::cout<<"\n=== V11 full decoder schedule ===\n";
        for(int L:ctxs){if(L>o.max_context)continue;
            float direct=time_direct(r,s,L,3,o.iters);
            cudaGraphExec_t exec=nullptr;float graph=time_graph(r,s,L,3,o.iters,&exec);
            CUDA_CHECK(cudaGraphLaunch(exec,s));CUDA_CHECK(cudaStreamSynchronize(s));std::vector<float>glog(64);CUDA_CHECK(cudaMemcpy(glog.data(),r.logits,64*sizeof(float),cudaMemcpyDeviceToHost));
            r.launch(s,L);CUDA_CHECK(cudaStreamSynchronize(s));std::vector<float>dlog(64);CUDA_CHECK(cudaMemcpy(dlog.data(),r.logits,64*sizeof(float),cudaMemcpyDeviceToHost));
            float diff=0.0f;for(int i=0;i<64;++i)diff=std::max(diff,std::fabs(glog[i]-dlog[i]));cudaGraphExecDestroy(exec);
            std::cout<<std::fixed<<std::setprecision(4)
                     <<"ctx "<<std::setw(4)<<L<<"  direct="<<direct<<" ms  graph="<<graph<<" ms"
                     <<"  speedup="<<std::setprecision(2)<<(direct/graph)<<"x"
                     <<"  graph-rate="<<std::setprecision(1)<<(1000.0f/graph)<<" tok/s"
                     <<"  output-diff="<<std::setprecision(6)<<diff<<"\n";
        }

        std::cout<<"\nGuardrails:\n"
                 <<"  * V11 is the first single timed GPU schedule containing all 30 decoder layers plus final norm and full BF16 LM head.\n"
                 <<"  * Projection dimensions, packing, attention structure and memory footprints are model-realistic; weights/hidden values/scales are synthetic.\n"
                 <<"  * Weight rings intentionally exceed L2 and rotate by layer; they are not the actual Microsoft checkpoint.\n"
                 <<"  * Direct vs Graph uses identical kernels and dependencies. Graph changes orchestration only.\n"
                 <<"  * Reported tok/s is therefore a synthetic GPU-runtime ceiling, not validated checkpoint text-generation throughput.\n";
        CUDA_CHECK(cudaStreamDestroy(s));std::cout<<"\nV11 completed.\n";return 0;
    }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
