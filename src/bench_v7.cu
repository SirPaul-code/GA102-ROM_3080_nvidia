#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
static constexpr int VOCAB = 128256;
static constexpr int LAYERS = 30;
static constexpr int MAX_CTX = 4096;
static constexpr float RMS_EPS = 1.0e-5f;

struct Options {
    int iters = 200;
    float linear_ms = 2.1265f;
};

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "--linear-ms" && i + 1 < argc) o.linear_ms = std::stof(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V7 BitNet decode-overhead benchmark\n"
                      << "  --iters N       primitive timing iterations (default 200)\n"
                      << "  --linear-ms X   V6 30-layer ternary linear time (default 2.1265 ms)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + a);
        }
    }
    if (o.iters <= 0 || o.linear_ms <= 0.0f)
        throw std::runtime_error("iters and linear-ms must be positive");
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

static float host_bf16_to_float(const bf16& v) {
    uint16_t hi = 0;
    std::memcpy(&hi, &v, sizeof(hi));
    uint32_t bits = (uint32_t)hi << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

__global__ void init_bf16(bf16* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        uint32_t z = (uint32_t)i * 747796405u + seed;
        z = ((z >> ((z >> 28) + 4)) ^ z) * 277803737u;
        z = (z >> 22) ^ z;
        float v = ((float)(z & 0xffffu) / 32767.5f) - 1.0f;
        p[i] = __float2bfloat16_rn(v);
    }
}

__global__ void init_i32(int32_t* p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        p[i] = (int32_t)((i * 1103515245ull + 12345ull) & 0x3ffffu) - 0x1ffff;
}

__global__ void init_u4(uint4* p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        uint32_t z = (uint32_t)i * 747796405u + 2891336453u;
        z = ((z >> ((z >> 28) + 4)) ^ z) * 277803737u;
        z = (z >> 22) ^ z;
        p[i] = make_uint4(z, z * 1664525u + 1013904223u,
                          z ^ 0x9e3779b9u, z * 2246822519u + 3266489917u);
    }
}

__global__ void rmsnorm_bf16(const bf16* __restrict__ x,
                             const bf16* __restrict__ w,
                             bf16* __restrict__ y,
                             int n, float eps) {
    __shared__ float red[256];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = __bfloat162float(x[i]);
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = __bfloat162float(x[i]);
        const float g = __bfloat162float(w[i]);
        y[i] = __float2bfloat16_rn(v * inv * g);
    }
}

__global__ void quantize_bf16_to_a8(const bf16* __restrict__ x,
                                    int8_t* __restrict__ q,
                                    float* __restrict__ inv_scale,
                                    int n) {
    __shared__ float red[256];
    const int tid = threadIdx.x;
    float m = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        m = fmaxf(m, fabsf(__bfloat162float(x[i])));
    red[tid] = m;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float amax = fmaxf(red[0], 1.0e-5f);
    const float scale = 127.0f / amax;
    if (tid == 0) *inv_scale = 1.0f / scale;
    for (int i = tid; i < n; i += blockDim.x) {
        int v = __float2int_rn(__bfloat162float(x[i]) * scale);
        v = (v < -128) ? -128 : ((v > 127) ? 127 : v);
        q[i] = (int8_t)v;
    }
}

__global__ void postscale_i32_to_bf16(const int32_t* __restrict__ x,
                                      bf16* __restrict__ y,
                                      int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn((float)x[i] * scale);
}

__global__ void relu2_gate_mul_bf16(const bf16* __restrict__ gate,
                                    const bf16* __restrict__ up,
                                    bf16* __restrict__ out,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = fmaxf(__bfloat162float(gate[i]), 0.0f);
        float u = __bfloat162float(up[i]);
        out[i] = __float2bfloat16_rn(g * g * u);
    }
}

__global__ void residual_add_bf16(const bf16* __restrict__ a,
                                  const bf16* __restrict__ b,
                                  bf16* __restrict__ y,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = __bfloat162float(a[i]) + __bfloat162float(b[i]);
        y[i] = __float2bfloat16_rn(v);
    }
}

__global__ void rope_qk_bf16(bf16* q, bf16* k,
                             const float* __restrict__ c,
                             const float* __restrict__ s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = (Q_HEADS + KV_HEADS) * (HEAD_DIM / 2);
    if (i >= pairs) return;
    const int head = i / (HEAD_DIM / 2);
    const int d = i % (HEAD_DIM / 2);
    bf16* base = (head < Q_HEADS)
        ? q + head * HEAD_DIM
        : k + (head - Q_HEADS) * HEAD_DIM;
    const float x1 = __bfloat162float(base[d]);
    const float x2 = __bfloat162float(base[d + HEAD_DIM / 2]);
    const float co = c[d];
    const float si = s[d];
    base[d] = __float2bfloat16_rn(x1 * co - x2 * si);
    base[d + HEAD_DIM / 2] = __float2bfloat16_rn(x2 * co + x1 * si);
}

__device__ __forceinline__ unsigned long long warp_xor_u64(unsigned long long v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v ^= __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void stream_u4(const uint4* __restrict__ p, size_t n,
                          unsigned long long* __restrict__ sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long v = 0;
    for (; i < n; i += stride) {
        const uint4 x = p[i];
        v ^= ((unsigned long long)x.x << 32) | x.y;
        v ^= ((unsigned long long)x.z << 32) | x.w;
    }
    v = warp_xor_u64(v);
    if ((threadIdx.x & 31) == 0) atomicXor(sink, v);
}

__device__ __forceinline__ float warp_sum_f32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void lmhead_bf16_warp(const bf16* __restrict__ w,
                                 const bf16* __restrict__ x,
                                 float* __restrict__ logits,
                                 int vocab, int hidden) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = tid >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= vocab) return;
    const bf16* wr = w + (size_t)row * hidden;
    float acc = 0.0f;
    for (int j = lane; j < hidden; j += 32)
        acc = fmaf(__bfloat162float(wr[j]), __bfloat162float(x[j]), acc);
    acc = warp_sum_f32(acc);
    if (lane == 0) logits[row] = acc;
}

static float bench_rms(const bf16* x, const bf16* w, bf16* y, int n, int iters) {
    auto f = [&] { rmsnorm_bf16<<<1, 256>>>(x, w, y, n, RMS_EPS); };
    return time_ms(f, 20, iters);
}

static float bench_quant(const bf16* x, int8_t* q, float* scale, int n, int iters) {
    auto f = [&] { quantize_bf16_to_a8<<<1, 256>>>(x, q, scale, n); };
    return time_ms(f, 20, iters);
}

static float bench_post(const int32_t* x, bf16* y, int n, int iters) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    auto f = [&] { postscale_i32_to_bf16<<<blocks, threads>>>(x, y, n, 1.0e-4f); };
    return time_ms(f, 20, iters);
}

int main(int argc, char** argv) {
    try {
        const Options o = parse_args(argc, argv);
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 8 || prop.minor != 6 ||
            std::string(prop.name).find("RTX 3080") == std::string::npos) {
            std::cerr << "V7 is intentionally restricted to RTX 3080 / SM86; found "
                      << prop.name << " sm_" << prop.major << prop.minor << "\n";
            return 3;
        }

        std::cout << "GA102-ROM V7: remaining BitNet decode costs\n"
                  << "GPU               : " << prop.name << "\n"
                  << "model             : microsoft/bitnet-b1.58-2B-4T shapes\n"
                  << "V6 linear baseline: " << std::fixed << std::setprecision(4)
                  << o.linear_ms << " ms/token (30 ternary layers)\n"
                  << "dtype proxy        : native CUDA bfloat16 for non-ternary state\n";

        bf16 *dh0=nullptr, *dh1=nullptr, *dhnorm=nullptr;
        bf16 *di0=nullptr, *di1=nullptr, *dinorm=nullptr;
        bf16 *dq=nullptr, *dk=nullptr, *dpost=nullptr;
        int8_t *dqh=nullptr, *dqi=nullptr;
        float *dscale=nullptr;
        int32_t *dproj=nullptr;
        float *dcos=nullptr, *dsin=nullptr;

        CUDA_CHECK(cudaMalloc(&dh0, HIDDEN*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&dh1, HIDDEN*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&dhnorm, HIDDEN*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&di0, INTER*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&di1, INTER*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&dinorm, INTER*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&dq, Q_HEADS*HEAD_DIM*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&dk, KV_HEADS*HEAD_DIM*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&dpost, 2*INTER*sizeof(bf16)));
        CUDA_CHECK(cudaMalloc(&dqh, HIDDEN));
        CUDA_CHECK(cudaMalloc(&dqi, INTER));
        CUDA_CHECK(cudaMalloc(&dscale, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dproj, 2*INTER*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&dcos, (HEAD_DIM/2)*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dsin, (HEAD_DIM/2)*sizeof(float)));

        init_bf16<<<32,256>>>(dh0,HIDDEN,1); init_bf16<<<32,256>>>(dh1,HIDDEN,2);
        init_bf16<<<32,256>>>(dhnorm,HIDDEN,3);
        init_bf16<<<32,256>>>(di0,INTER,4); init_bf16<<<32,256>>>(di1,INTER,5);
        init_bf16<<<32,256>>>(dinorm,INTER,6);
        init_bf16<<<32,256>>>(dq,Q_HEADS*HEAD_DIM,7);
        init_bf16<<<32,256>>>(dk,KV_HEADS*HEAD_DIM,8);
        init_i32<<<64,256>>>(dproj,2*INTER);

        std::vector<float> hc(HEAD_DIM/2), hs(HEAD_DIM/2);
        for (int i=0;i<HEAD_DIM/2;++i) {
            float a = 0.013f*(float)i;
            hc[i]=std::cos(a); hs[i]=std::sin(a);
        }
        CUDA_CHECK(cudaMemcpy(dcos,hc.data(),hc.size()*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dsin,hs.data(),hs.size()*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        const float rms_h = bench_rms(dh0,dhnorm,dh1,HIDDEN,o.iters);
        const float rms_i = bench_rms(di0,dinorm,di1,INTER,o.iters);
        const float q_h = bench_quant(dh0,dqh,dscale,HIDDEN,o.iters);
        const float q_i = bench_quant(di0,dqi,dscale,INTER,o.iters);

        const int t=256;
        const int bi=(INTER+t-1)/t;
        const int bh=(HIDDEN+t-1)/t;
        auto relu_launch=[&]{relu2_gate_mul_bf16<<<bi,t>>>(di0,di1,dinorm,INTER);};
        auto res_launch=[&]{residual_add_bf16<<<bh,t>>>(dh0,dh1,dhnorm,HIDDEN);};
        const int rope_pairs=(Q_HEADS+KV_HEADS)*(HEAD_DIM/2);
        auto rope_launch=[&]{rope_qk_bf16<<<(rope_pairs+t-1)/t,t>>>(dq,dk,dcos,dsin);};
        const float relu_ms=time_ms(relu_launch,20,o.iters);
        const float residual_ms=time_ms(res_launch,20,o.iters);
        const float rope_ms=time_ms(rope_launch,20,o.iters);

        const float post_qkv=bench_post(dproj,dpost,3840,o.iters);
        const float post_o=bench_post(dproj,dpost,2560,o.iters);
        const float post_gu=bench_post(dproj,dpost,13824,o.iters);
        const float post_down=bench_post(dproj,dpost,2560,o.iters);

        std::cout << "\n=== V7 per-token primitive timings ===\n"
                  << std::setprecision(4)
                  << "RMSNorm hidden 2560       : " << rms_h << " ms\n"
                  << "RMSNorm intermediate 6912 : " << rms_i << " ms\n"
                  << "BF16->A8 quant hidden     : " << q_h << " ms\n"
                  << "BF16->A8 quant inter      : " << q_i << " ms\n"
                  << "RoPE Q20+K5 x128          : " << rope_ms << " ms\n"
                  << "ReLU2(gate)*up 6912       : " << relu_ms << " ms\n"
                  << "residual add 2560         : " << residual_ms << " ms\n"
                  << "postscale QKV 3840         : " << post_qkv << " ms\n"
                  << "postscale O 2560           : " << post_o << " ms\n"
                  << "postscale gate+up 13824    : " << post_gu << " ms\n"
                  << "postscale down 2560        : " << post_down << " ms\n";

        const double layer_aux =
            3.0*rms_h + rms_i +
            3.0*q_h + q_i +
            rope_ms + relu_ms + 2.0*residual_ms +
            post_qkv + post_o + post_gu + post_down;
        const double aux30 = layer_aux*LAYERS + rms_h;

        std::cout << "\nlaunch-separated auxiliary/layer : " << layer_aux << " ms\n"
                  << "launch-separated auxiliary/30L  : " << aux30 << " ms\n"
                  << "note: many of these stages are intentionally fusable in the fixed runtime.\n";

        const size_t vecs_per_token = (2ull*KV_HEADS*HEAD_DIM*sizeof(bf16))/sizeof(uint4);
        const size_t max_vecs_layer = (size_t)MAX_CTX*vecs_per_token;
        const size_t kv_vecs = (size_t)LAYERS*max_vecs_layer;
        uint4* dkv=nullptr;
        unsigned long long* dsink=nullptr;
        CUDA_CHECK(cudaMalloc(&dkv,kv_vecs*sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&dsink,sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(dsink,0,sizeof(unsigned long long)));
        init_u4<<<prop.multiProcessorCount*8,256>>>(dkv,kv_vecs);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::array<int,5> contexts{{128,512,1024,2048,4096}};
        std::array<float,5> kv_ms{};
        std::cout << "\n=== V7 KV-cache read traffic floor (30 sequential layer launches) ===\n";
        for (size_t ci=0;ci<contexts.size();++ci) {
            const int ctx=contexts[ci];
            const size_t n=(size_t)ctx*vecs_per_token;
            auto launch_all=[&]{
                for (int l=0;l<LAYERS;++l) {
                    const uint4* p=dkv+(size_t)l*max_vecs_layer;
                    stream_u4<<<prop.multiProcessorCount*8,256>>>(p,n,dsink);
                }
            };
            const int kiters=std::max(10,std::min(o.iters,100));
            kv_ms[ci]=time_ms(launch_all,5,kiters);
            const double bytes=(double)LAYERS*ctx*2560.0;
            const double gbps=bytes/(kv_ms[ci]*1.0e6);
            std::cout << "context " << std::setw(4) << ctx << " : "
                      << std::fixed << std::setprecision(4) << kv_ms[ci] << " ms  "
                      << std::setprecision(1) << gbps << " GB/s aggregate\n";
        }

        const size_t lm_values=(size_t)VOCAB*HIDDEN;
        const size_t lm_bytes=lm_values*sizeof(bf16);
        bf16* dlmw=nullptr;
        float* dlogits=nullptr;
        CUDA_CHECK(cudaMalloc(&dlmw,lm_bytes));
        CUDA_CHECK(cudaMalloc(&dlogits,(size_t)VOCAB*sizeof(float)));
        init_bf16<<<prop.multiProcessorCount*16,256>>>(dlmw,lm_values,0x1234abcd);
        CUDA_CHECK(cudaDeviceSynchronize());
        const int lm_threads=128;
        const int lm_warps=lm_threads/32;
        const int lm_blocks=(VOCAB+lm_warps-1)/lm_warps;
        auto lm_launch=[&]{lmhead_bf16_warp<<<lm_blocks,lm_threads>>>(dlmw,dh0,dlogits,VOCAB,HIDDEN);};

        lm_launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<bf16> hw4((size_t)4*HIDDEN), hx(HIDDEN);
        std::vector<float> gl4(4);
        CUDA_CHECK(cudaMemcpy(hw4.data(),dlmw,hw4.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hx.data(),dh0,hx.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gl4.data(),dlogits,4*sizeof(float),cudaMemcpyDeviceToHost));
        bool lm_ok=true;
        for(int r=0;r<4;++r){
            double ref=0.0;
            for(int j=0;j<HIDDEN;++j)
                ref+=(double)host_bf16_to_float(hw4[(size_t)r*HIDDEN+j])*(double)host_bf16_to_float(hx[j]);
            const double err=std::fabs((double)gl4[r]-ref);
            if(err>1.0e-2*std::max(1.0,std::fabs(ref))){lm_ok=false;break;}
        }
        if(!lm_ok){std::cerr<<"LM-head BF16 proxy correctness check failed\n";return 4;}

        const int lm_iters=std::max(5,std::min(o.iters,50));
        const float lm_ms=time_ms(lm_launch,5,lm_iters);
        const double lm_gbps=(double)lm_bytes/(lm_ms*1.0e6);
        const double lm_gmacs=(double)VOCAB*HIDDEN/(lm_ms*1.0e6);
        std::cout << "\n=== V7 tied BF16 LM head ===\n"
                  << "shape             : " << VOCAB << " x " << HIDDEN << "\n"
                  << "weight bytes      : " << std::fixed << std::setprecision(3)
                  << (double)lm_bytes/(1024.0*1024.0) << " MiB\n"
                  << "correctness       : PASS (first 4 rows vs CPU BF16 values)\n"
                  << "kernel time       : " << std::setprecision(4) << lm_ms << " ms\n"
                  << "weight-read rate  : " << std::setprecision(1) << lm_gbps << " GB/s\n"
                  << "logical throughput: " << lm_gmacs << " GMAC/s\n";

        std::cout << "\n=== V7 decode accounting (NOT full-model token/s) ===\n";
        std::cout << "V6 ternary linears       : " << std::fixed << std::setprecision(4)
                  << o.linear_ms << " ms\n"
                  << "launch-separated aux 30L : " << aux30 << " ms\n"
                  << "BF16 LM head              : " << lm_ms << " ms\n";
        for (size_t ci=0;ci<contexts.size();++ci) {
            const double subtotal=o.linear_ms+aux30+lm_ms+kv_ms[ci];
            std::cout << "ctx " << std::setw(4) << contexts[ci]
                      << " + KV traffic floor : " << std::setprecision(4) << subtotal
                      << " ms  => " << std::setprecision(1) << (1000.0/subtotal)
                      << " tok/s accounting ceiling\n";
        }
        std::cout << "\nGuardrails:\n"
                  << "  * accounting ceiling is NOT measured end-to-end generation throughput.\n"
                  << "  * KV number measures K+V read traffic and 30 launch costs, not QK/softmax/V arithmetic.\n"
                  << "  * auxiliary accounting is launch-separated; a fixed runtime can fuse norm/quant/postscale/residual stages.\n"
                  << "  * tokenizer, sampling, host scheduling and cross-layer dependencies are not included.\n";

        CUDA_CHECK(cudaFree(dh0)); CUDA_CHECK(cudaFree(dh1)); CUDA_CHECK(cudaFree(dhnorm));
        CUDA_CHECK(cudaFree(di0)); CUDA_CHECK(cudaFree(di1)); CUDA_CHECK(cudaFree(dinorm));
        CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk)); CUDA_CHECK(cudaFree(dpost));
        CUDA_CHECK(cudaFree(dqh)); CUDA_CHECK(cudaFree(dqi)); CUDA_CHECK(cudaFree(dscale));
        CUDA_CHECK(cudaFree(dproj)); CUDA_CHECK(cudaFree(dcos)); CUDA_CHECK(cudaFree(dsin));
        CUDA_CHECK(cudaFree(dkv)); CUDA_CHECK(cudaFree(dsink)); CUDA_CHECK(cudaFree(dlmw)); CUDA_CHECK(cudaFree(dlogits));

        std::cout << "\nV7 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
