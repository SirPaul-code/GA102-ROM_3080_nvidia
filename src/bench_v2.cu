#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
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

struct Options {
    int m = 8192;
    int k = 8192;
    int iters = 100;
    int stream_mib = 512;
};

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--m" && i + 1 < argc) o.m = std::stoi(argv[++i]);
        else if (a == "--k" && i + 1 < argc) o.k = std::stoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "--stream-mib" && i + 1 < argc) o.stream_mib = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V2 warp-cooperative benchmark\n"
                      << "  --m N          rows (default 8192)\n"
                      << "  --k N          columns (default 8192, multiple of 32)\n"
                      << "  --iters N      timing iterations (default 100)\n"
                      << "  --stream-mib N vectorized DRAM test size (default 512 MiB)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown/incomplete argument: " + a);
        }
    }
    if (o.m <= 0 || o.k <= 0 || o.iters <= 0 || o.stream_mib < 64)
        throw std::runtime_error("Invalid benchmark parameters");
    if ((o.k & 31) != 0)
        throw std::runtime_error("V2 requires K to be a multiple of 32");
    return o;
}

__device__ __forceinline__ int warp_sum_i32(int v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// One warp owns one output row. Each lane performs DP4A on a disjoint set
// of 4-element chunks, then lane 0 writes the warp reduction.
__global__ void gemv_i8_dp4a_warp(const int8_t* __restrict__ w,
                                  const int8_t* __restrict__ x,
                                  int32_t* __restrict__ y,
                                  int M, int K) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = tid >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= M) return;

    const int8_t* wr = w + (size_t)warp * K;
    int acc = 0;
    for (int i = lane * 4; i < K; i += 32 * 4) {
        const int pw = *reinterpret_cast<const int*>(wr + i);
        const int px = *reinterpret_cast<const int*>(x + i);
        acc = __dp4a(pw, px, acc);
    }
    acc = warp_sum_i32(acc);
    if (lane == 0) y[warp] = acc;
}

// Same warp-per-row mapping, but the ternary weights stay packed at 2 bits
// until the exact 4-weight chunk is consumed. Four ternary values are decoded
// into one DP4A operand register locally in each lane.
__global__ void gemv_w2_dp4a_warp(const uint8_t* __restrict__ w2,
                                  const int8_t* __restrict__ x,
                                  int32_t* __restrict__ y,
                                  int M, int K) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = tid >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= M) return;

    const int packed_per_row = K >> 2;
    const uint8_t* wr = w2 + (size_t)warp * packed_per_row;
    int acc = 0;

    for (int g = lane; g < packed_per_row; g += 32) {
        const uint8_t q = wr[g];
        const uint32_t a = (uint8_t)((q       & 3u) - 1);
        const uint32_t b = (uint8_t)(((q >> 2) & 3u) - 1);
        const uint32_t c = (uint8_t)(((q >> 4) & 3u) - 1);
        const uint32_t d = (uint8_t)(((q >> 6) & 3u) - 1);
        const int pw = (int)(a | (b << 8) | (c << 16) | (d << 24));
        const int px = *reinterpret_cast<const int*>(x + (g << 2));
        acc = __dp4a(pw, px, acc);
    }

    acc = warp_sum_i32(acc);
    if (lane == 0) y[warp] = acc;
}

// Exact two-bitplane ternary dot product. Compared with V1, the 32 lanes
// cooperate on the words of one row instead of one thread serially scanning it.
__global__ void gemv_bitplane_popc_warp(const uint32_t* __restrict__ pos,
                                        const uint32_t* __restrict__ neg,
                                        const uint32_t* __restrict__ xb,
                                        int32_t* __restrict__ y,
                                        int M, int words_per_row) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = tid >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= M) return;

    const uint32_t* pr = pos + (size_t)warp * words_per_row;
    const uint32_t* nr = neg + (size_t)warp * words_per_row;
    int acc = 0;

    for (int j = lane; j < words_per_row; j += 32) {
        const uint32_t p = pr[j];
        const uint32_t n = nr[j];
        #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t a = xb[(size_t)bit * words_per_row + j];
            const int cnt = __popc(p & a) - __popc(n & a);
            const int scale = (bit == 7) ? -128 : (1 << bit);
            acc += cnt * scale;
        }
    }

    acc = warp_sum_i32(acc);
    if (lane == 0) y[warp] = acc;
}

// Non-compressible initialization for the DRAM read benchmark.
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

__device__ __forceinline__ unsigned long long warp_xor_u64(unsigned long long v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v ^= __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// 16-byte vector loads and only one atomic per warp. This is intended to be a
// much cleaner approximation of sustained global-memory read bandwidth than V1.
__global__ void stream_u4(const uint4* __restrict__ p, size_t n,
                          unsigned long long* __restrict__ sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long s = 0;
    for (; i < n; i += stride) {
        const uint4 v = p[i];
        s ^= ((unsigned long long)v.x << 32) | v.y;
        s ^= ((unsigned long long)v.z << 32) | v.w;
    }
    s = warp_xor_u64(s);
    if ((threadIdx.x & 31) == 0)
        atomicXor(sink, s);
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

static void cpu_reference(const std::vector<int8_t>& w,
                          const std::vector<int8_t>& x,
                          std::vector<int32_t>& y,
                          int M, int K) {
    for (int r = 0; r < M; ++r) {
        int32_t acc = 0;
        const int8_t* wr = w.data() + (size_t)r * K;
        for (int c = 0; c < K; ++c)
            acc += (int32_t)wr[c] * (int32_t)x[c];
        y[r] = acc;
    }
}

static bool same(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "Mismatch row " << i << ": ref=" << a[i]
                      << " gpu=" << b[i] << "\n";
            return false;
        }
    }
    return true;
}

static void benchmark_gemv(const Options& o, const cudaDeviceProp& prop) {
    const int M = o.m;
    const int K = o.k;

    std::cout << "\n=== V2 warp-cooperative ternary GEMV ===\n"
              << "shape             : M=" << M << " K=" << K << "\n"
              << "mapping           : 1 warp / output row\n";

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> wd(-1, 1);
    std::uniform_int_distribution<int> xd(-127, 127);

    std::vector<int8_t> hw((size_t)M * K), hx(K);
    for (auto& v : hw) v = (int8_t)wd(rng);
    for (auto& v : hx) v = (int8_t)xd(rng);

    std::vector<int32_t> ref(M), got(M);
    cpu_reference(hw, hx, ref, M, K);

    const int packed_per_row = K >> 2;
    std::vector<uint8_t> hw2((size_t)M * packed_per_row, 0);
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; ++c) {
            const uint8_t code = (uint8_t)(hw[(size_t)r * K + c] + 1);
            hw2[(size_t)r * packed_per_row + (c >> 2)] |=
                (uint8_t)(code << (2 * (c & 3)));
        }
    }

    const int words = K >> 5;
    std::vector<uint32_t> hpos((size_t)M * words, 0);
    std::vector<uint32_t> hneg((size_t)M * words, 0);
    std::vector<uint32_t> hxb((size_t)8 * words, 0);

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; ++c) {
            const int8_t v = hw[(size_t)r * K + c];
            if (v > 0) hpos[(size_t)r * words + (c >> 5)] |= 1u << (c & 31);
            if (v < 0) hneg[(size_t)r * words + (c >> 5)] |= 1u << (c & 31);
        }
    }
    for (int c = 0; c < K; ++c) {
        const uint8_t u = (uint8_t)hx[c];
        for (int b = 0; b < 8; ++b)
            if ((u >> b) & 1u)
                hxb[(size_t)b * words + (c >> 5)] |= 1u << (c & 31);
    }

    int8_t *dw = nullptr, *dx = nullptr;
    uint8_t* dw2 = nullptr;
    uint32_t *dpos = nullptr, *dneg = nullptr, *dxb = nullptr;
    int32_t* dy = nullptr;

    CUDA_CHECK(cudaMalloc(&dw, hw.size()));
    CUDA_CHECK(cudaMalloc(&dx, hx.size()));
    CUDA_CHECK(cudaMalloc(&dw2, hw2.size()));
    CUDA_CHECK(cudaMalloc(&dpos, hpos.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dneg, hneg.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dxb, hxb.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dy, (size_t)M * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw2, hw2.data(), hw2.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpos, hpos.data(), hpos.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dneg, hneg.data(), hneg.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dxb, hxb.data(), hxb.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    const int threads = 256; // 8 warps / block
    const int warps_per_block = threads / 32;
    const int blocks = (M + warps_per_block - 1) / warps_per_block;

    auto run_one = [&](const char* name,
                       const std::function<void()>& launch,
                       double weight_bytes) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got.data(), dy, (size_t)M * sizeof(int32_t), cudaMemcpyDeviceToHost));
        const bool ok = same(ref, got);
        if (!ok) std::exit(4);

        const float ms = time_ms(launch, 10, o.iters);
        const double macs = (double)M * K;
        const double gmacs = macs / (ms * 1e6);
        const double logical_wgb = weight_bytes / (ms * 1e6);
        std::cout << std::left << std::setw(24) << name
                  << " correctness=PASS  " << std::right
                  << std::fixed << std::setprecision(3) << ms << " ms  "
                  << std::setprecision(1) << gmacs << " GMAC/s  "
                  << "weight-read~" << logical_wgb << " GB/s\n";
    };

    run_one("I8 DP4A warp", [&] {
        gemv_i8_dp4a_warp<<<blocks, threads>>>(dw, dx, dy, M, K);
    }, (double)hw.size());

    run_one("W2 decode+DP4A warp", [&] {
        gemv_w2_dp4a_warp<<<blocks, threads>>>(dw2, dx, dy, M, K);
    }, (double)hw2.size());

    run_one("W1x2 POPC warp", [&] {
        gemv_bitplane_popc_warp<<<blocks, threads>>>(dpos, dneg, dxb, dy, M, words);
    }, (double)(hpos.size() + hneg.size()) * sizeof(uint32_t));

    CUDA_CHECK(cudaFree(dw));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dw2));
    CUDA_CHECK(cudaFree(dpos));
    CUDA_CHECK(cudaFree(dneg));
    CUDA_CHECK(cudaFree(dxb));
    CUDA_CHECK(cudaFree(dy));

    (void)prop;
}

static void benchmark_stream(const Options& o, const cudaDeviceProp& p) {
    const size_t bytes = (size_t)o.stream_mib * 1024ull * 1024ull;
    const size_t n = bytes / sizeof(uint4);
    uint4* d = nullptr;
    unsigned long long* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(sink, 0, sizeof(unsigned long long)));

    const int threads = 256;
    const int blocks = std::max(1, p.multiProcessorCount * 8);
    init_u4<<<blocks, threads>>>(d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = std::min(o.iters, 50);
    const float ms = time_ms([&] {
        stream_u4<<<blocks, threads>>>(d, n, sink);
    }, 5, iters);
    CUDA_CHECK(cudaGetLastError());

    const double gbps = (double)bytes / (ms * 1e6);
    int mem_clock_khz = 0;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
    const double theoretical = 2.0 * (double)mem_clock_khz * 1000.0 *
                               ((double)p.memoryBusWidth / 8.0) / 1e9;

    std::cout << "\n=== V2 vectorized DRAM read ===\n"
              << "buffer            : " << o.stream_mib << " MiB non-compressible data\n"
              << "load width         : 16 bytes/thread (uint4)\n"
              << "kernel time        : " << std::fixed << std::setprecision(3) << ms << " ms\n"
              << "measured read BW   : " << std::setprecision(1) << gbps << " GB/s\n"
              << "property peak      : " << theoretical << " GB/s\n"
              << "peak utilization   : " << (100.0 * gbps / theoretical) << " %\n";

    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFree(sink));
}

int main(int argc, char** argv) {
    try {
        const Options o = parse_args(argc, argv);
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
        if (!(p.major == 8 && p.minor == 6) ||
            std::string(p.name).find("RTX 3080") == std::string::npos) {
            std::cerr << "V2 is intentionally restricted to RTX 3080 / SM86. Found: "
                      << p.name << " sm_" << p.major << p.minor << "\n";
            return 3;
        }

        std::cout << "GA102-ROM V2\n"
                  << "GPU               : " << p.name << "\n"
                  << "SMs               : " << p.multiProcessorCount << "\n";

        benchmark_gemv(o, p);
        benchmark_stream(o, p);
        std::cout << "\nV2 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
