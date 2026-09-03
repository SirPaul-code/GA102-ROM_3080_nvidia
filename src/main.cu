#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    bool info = false;
    bool ternary = false;
    bool bmma = false;
    bool allow_other_gpu = false;
    int m = 8192;
    int k = 8192;
    int iters = 100;
    int bmma_loops = 4096;
};

static void usage() {
    std::cout
        << "GA102-ROM research lab\n\n"
        << "  --all                  run all tests\n"
        << "  --info                 print GPU information\n"
        << "  --ternary              benchmark exact ternary GEMV backends\n"
        << "  --bmma                 run Ampere 1-bit Tensor Core probe\n"
        << "  --m N                  output rows (default 8192)\n"
        << "  --k N                  row width (default 8192; must be multiple of 32)\n"
        << "  --iters N              timing iterations (default 100)\n"
        << "  --bmma-loops N         BMMA ops per warp (default 4096)\n"
        << "  --allow-other-gpu      skip RTX 3080 identity check\n";
}

static Options parse_args(int argc, char** argv) {
    Options o;
    bool chosen = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        if (a == "--all") { o.info = o.ternary = o.bmma = true; chosen = true; continue; }
        if (a == "--info") { o.info = true; chosen = true; continue; }
        if (a == "--ternary") { o.ternary = true; chosen = true; continue; }
        if (a == "--bmma") { o.bmma = true; chosen = true; continue; }
        if (a == "--allow-other-gpu") { o.allow_other_gpu = true; continue; }
        if ((a == "--m" || a == "--k" || a == "--iters" || a == "--bmma-loops") && i + 1 < argc) {
            int v = std::stoi(argv[++i]);
            if (a == "--m") o.m = v;
            else if (a == "--k") o.k = v;
            else if (a == "--iters") o.iters = v;
            else o.bmma_loops = v;
            continue;
        }
        throw std::runtime_error("Unknown or incomplete argument: " + a);
    }
    if (!chosen) o.info = o.ternary = o.bmma = true;
    if (o.m <= 0 || o.k <= 0 || o.iters <= 0 || o.bmma_loops <= 0)
        throw std::runtime_error("Dimensions and iteration counts must be positive");
    return o;
}

static cudaDeviceProp require_device(bool allow_other) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    bool sm86 = p.major == 8 && p.minor == 6;
    bool is3080 = std::string(p.name).find("RTX 3080") != std::string::npos;
    if (!allow_other && (!sm86 || !is3080)) {
        std::cerr << "Expected RTX 3080 / SM86; found " << p.name
                  << " (sm_" << p.major << p.minor << ").\n"
                  << "Use --allow-other-gpu only for development.\n";
        std::exit(3);
    }
    return p;
}

static void print_device(const cudaDeviceProp& p) {
    int dev = 0;
    int memory_clock_khz = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, dev));

    // CUDA 13 removed cudaDeviceProp::memoryClockRate. NVIDIA's supported
    // replacement is cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate).
    // memory_clock_khz is the memory clock in kHz; DDR transfer rate is 2x.
    double bw = 2.0 * (double)memory_clock_khz * 1000.0
              * ((double)p.memoryBusWidth / 8.0) / 1e9;

    std::cout << "\n=== Device ===\n"
              << "name              : " << p.name << "\n"
              << "compute capability: sm_" << p.major << p.minor << "\n"
              << "SMs               : " << p.multiProcessorCount << "\n"
              << "VRAM              : " << std::fixed << std::setprecision(2)
              << (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GiB\n"
              << "memory bus        : " << p.memoryBusWidth << " bit\n"
              << "memory clock      : " << std::setprecision(3)
              << ((double)memory_clock_khz / 1e6) << " GHz\n"
              << "property BW       : " << std::setprecision(1) << bw << " GB/s\n"
              << "CUDA runtime      : " << CUDART_VERSION << "\n";
}

__global__ void gemv_i8_dp4a(const int8_t* __restrict__ w,
                             const int8_t* __restrict__ x,
                             int32_t* __restrict__ y,
                             int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const int8_t* wr = w + (size_t)row * K;
    int acc = 0;
    for (int i = 0; i < K; i += 4) {
        // K is constrained to a multiple of 32, so these 32-bit loads are aligned.
        int pw = *reinterpret_cast<const int*>(wr + i);
        int px = *reinterpret_cast<const int*>(x + i);
        acc = __dp4a(pw, px, acc);
    }
    y[row] = acc;
}

__global__ void gemv_w2_dp4a(const uint8_t* __restrict__ w2,
                             const int8_t* __restrict__ x,
                             int32_t* __restrict__ y,
                             int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const int packed_per_row = K / 4;
    const uint8_t* wr = w2 + (size_t)row * packed_per_row;
    int acc = 0;

    for (int g = 0; g < packed_per_row; ++g) {
        uint8_t q = wr[g];
        int8_t a = (int8_t)((q & 3u) - 1);
        int8_t b = (int8_t)(((q >> 2) & 3u) - 1);
        int8_t c = (int8_t)(((q >> 4) & 3u) - 1);
        int8_t d = (int8_t)(((q >> 6) & 3u) - 1);

        uint32_t pw = (uint8_t)a
                    | ((uint32_t)(uint8_t)b << 8)
                    | ((uint32_t)(uint8_t)c << 16)
                    | ((uint32_t)(uint8_t)d << 24);
        int px = *reinterpret_cast<const int*>(x + g * 4);
        acc = __dp4a((int)pw, px, acc);
    }
    y[row] = acc;
}

// Exact ternary dot product using two 1-bit weight masks and eight activation
// bitplanes. Signed INT8 reconstruction is b0 + 2*b1 + ... + 64*b6 - 128*b7.
__global__ void gemv_bitplane_popc(const uint32_t* __restrict__ pos,
                                   const uint32_t* __restrict__ neg,
                                   const uint32_t* __restrict__ xb,
                                   int32_t* __restrict__ y,
                                   int M, int words_per_row) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const uint32_t* pr = pos + (size_t)row * words_per_row;
    const uint32_t* nr = neg + (size_t)row * words_per_row;
    int acc = 0;

    #pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        int count = 0;
        const uint32_t* plane = xb + (size_t)bit * words_per_row;
        for (int j = 0; j < words_per_row; ++j) {
            uint32_t a = plane[j];
            count += __popc(pr[j] & a) - __popc(nr[j] & a);
        }
        const int scale = (bit == 7) ? -128 : (1 << bit);
        acc += count * scale;
    }
    y[row] = acc;
}

__global__ void stream_w2(const uint8_t* __restrict__ w,
                          uint64_t bytes,
                          uint64_t* __restrict__ sink) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint64_t sum = 0;
    for (; i < bytes; i += stride) sum += w[i];
    if ((threadIdx.x & 31) == 0)
        atomicAdd((unsigned long long*)sink, (unsigned long long)sum);
}

// Raw Ampere binary Tensor Core probe. A warp repeatedly executes the native
// m16n8k256 b1 AND+POPC MMA path. This is a hardware-path probe, not yet the
// final ternary GEMV mapping.
__global__ void bmma_probe(const uint32_t* __restrict__ A,
                           const uint32_t* __restrict__ B,
                           int32_t* __restrict__ out,
                           int loops) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    uint32_t a0 = A[(warp * 4 + (lane & 3)) & 4095];
    uint32_t b0 = B[(warp * 2 + (lane & 1)) & 4095];
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    for (int i = 0; i < loops; ++i) {
        asm volatile(
            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
            : "r"(a0), "r"(a0), "r"(a0), "r"(a0), "r"(b0), "r"(b0));

        // Create a true data dependency so ptxas cannot collapse the loop into
        // a constant-expression benchmark.
        a0 ^= (uint32_t)(i + lane);
        b0 ^= (uint32_t)(i * 17 + lane);
    }

    if (lane < 4)
        out[warp * 4 + lane] = lane == 0 ? c0 : lane == 1 ? c1 : lane == 2 ? c2 : c3;
#else
    (void)A; (void)B; (void)out; (void)loops;
#endif
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

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms / iters;
}

static void cpu_reference(const std::vector<int8_t>& w,
                          const std::vector<int8_t>& x,
                          std::vector<int32_t>& y,
                          int M, int K) {
    for (int r = 0; r < M; ++r) {
        int32_t acc = 0;
        for (int c = 0; c < K; ++c)
            acc += (int32_t)w[(size_t)r * K + c] * (int32_t)x[c];
        y[r] = acc;
    }
}

static bool same(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "Mismatch at row " << i << ": " << a[i] << " != " << b[i] << "\n";
            return false;
        }
    }
    return true;
}

static void run_ternary(int M, int K, int iters) {
    if ((K & 31) != 0)
        throw std::runtime_error("--k must be a multiple of 32 for the current exact benchmark kernels");

    std::cout << "\n=== Exact ternary GEMV lab ===\n";
    std::cout << "shape: M=" << M << " K=" << K << "\n";

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> wd(-1, 1);
    std::uniform_int_distribution<int> xd(-127, 127);

    std::vector<int8_t> hw((size_t)M * K), hx(K);
    for (auto& v : hw) v = (int8_t)wd(rng);
    for (auto& v : hx) v = (int8_t)xd(rng);

    std::vector<int32_t> ref(M);
    cpu_reference(hw, hx, ref, M, K);

    const int packed_per_row = K / 4;
    std::vector<uint8_t> hw2((size_t)M * packed_per_row, 0);
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; ++c) {
            uint8_t code = (uint8_t)(hw[(size_t)r * K + c] + 1);
            hw2[(size_t)r * packed_per_row + (c >> 2)]
                |= (uint8_t)(code << (2 * (c & 3)));
        }
    }

    const int words = K / 32;
    std::vector<uint32_t> hpos((size_t)M * words, 0);
    std::vector<uint32_t> hneg((size_t)M * words, 0);
    std::vector<uint32_t> hxb((size_t)8 * words, 0);

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; ++c) {
            int8_t v = hw[(size_t)r * K + c];
            if (v > 0) hpos[(size_t)r * words + (c >> 5)] |= 1u << (c & 31);
            if (v < 0) hneg[(size_t)r * words + (c >> 5)] |= 1u << (c & 31);
        }
    }

    for (int c = 0; c < K; ++c) {
        uint8_t u = (uint8_t)hx[c];
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

    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;
    std::vector<int32_t> got(M);

    auto check_and_time = [&](const char* name,
                              const std::function<void()>& launch,
                              double weight_bytes) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got.data(), dy, (size_t)M * sizeof(int32_t), cudaMemcpyDeviceToHost));

        bool ok = same(ref, got);
        float ms = time_ms(launch, 5, iters);
        CUDA_CHECK(cudaGetLastError());

        double macs = (double)M * K;
        double gmacs = macs / (ms * 1e6);
        double gbps = weight_bytes / (ms * 1e6);

        std::cout << std::left << std::setw(20) << name
                  << " correctness=" << (ok ? "PASS" : "FAIL")
                  << "  " << std::right << std::fixed << std::setprecision(3) << ms << " ms"
                  << "  " << std::setprecision(1) << gmacs << " GMAC/s"
                  << "  weight-read~" << gbps << " GB/s\n";
        if (!ok) std::exit(4);
    };

    check_and_time("I8 DP4A",
                   [&]{ gemv_i8_dp4a<<<blocks, threads>>>(dw, dx, dy, M, K); },
                   (double)hw.size());

    check_and_time("W2 decode+DP4A",
                   [&]{ gemv_w2_dp4a<<<blocks, threads>>>(dw2, dx, dy, M, K); },
                   (double)hw2.size());

    check_and_time("W1x2 POPC",
                   [&]{ gemv_bitplane_popc<<<blocks, threads>>>(dpos, dneg, dxb, dy, M, words); },
                   (double)(hpos.size() + hneg.size()) * sizeof(uint32_t));

    uint64_t* dsink = nullptr;
    CUDA_CHECK(cudaMalloc(&dsink, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(dsink, 0, sizeof(uint64_t)));

    const int sblocks = 4096;
    float stream_ms = time_ms(
        [&]{ stream_w2<<<sblocks, 256>>>(dw2, (uint64_t)hw2.size(), dsink); },
        5, iters);
    CUDA_CHECK(cudaGetLastError());

    std::cout << std::left << std::setw(20) << "W2 raw stream"
              << "                  " << std::right << std::fixed << std::setprecision(3)
              << stream_ms << " ms"
              << "  " << std::setprecision(1)
              << ((double)hw2.size() / (stream_ms * 1e6)) << " GB/s\n";

    CUDA_CHECK(cudaFree(dsink));
    CUDA_CHECK(cudaFree(dw));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dw2));
    CUDA_CHECK(cudaFree(dpos));
    CUDA_CHECK(cudaFree(dneg));
    CUDA_CHECK(cudaFree(dxb));
    CUDA_CHECK(cudaFree(dy));
}

static void run_bmma(const cudaDeviceProp& p, int loops, int iters) {
    std::cout << "\n=== SM86 BMMA b1 Tensor Core probe ===\n";
    if (p.major < 8) {
        std::cout << "BMMA skipped: device is not Ampere-class.\n";
        return;
    }

    const int threads = 128;
    const int blocks = std::max(1, p.multiProcessorCount * 4);
    const int warps = blocks * (threads / 32);

    std::vector<uint32_t> h(4096, 0xA5A5A5A5u);
    uint32_t *da = nullptr, *db = nullptr;
    int32_t* dout = nullptr;

    CUDA_CHECK(cudaMalloc(&da, h.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&db, h.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dout, (size_t)warps * 4 * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(da, h.data(), h.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, h.data(), h.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    auto launch = [&]{ bmma_probe<<<blocks, threads>>>(da, db, dout, loops); };
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = time_ms(launch, 5, iters);
    CUDA_CHECK(cudaGetLastError());

    // m16*n8*k256 = 32768 binary AND contributions per warp-level MMA.
    double binary_ops = (double)warps * loops * 16.0 * 8.0 * 256.0;
    double tops = binary_ops / (ms * 1e9);

    std::cout << "warps             : " << warps << "\n"
              << "BMMA/warp         : " << loops << "\n"
              << "kernel time       : " << std::fixed << std::setprecision(3) << ms << " ms\n"
              << "logical b1 ANDs   : " << std::setprecision(2) << tops << " TOP/s\n"
              << "note              : raw dependent-loop probe; not a vendor peak benchmark\n";

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dout));
}

int main(int argc, char** argv) {
    try {
        Options o = parse_args(argc, argv);
        cudaDeviceProp p = require_device(o.allow_other_gpu);

        if (o.info) print_device(p);
        if (o.ternary) run_ternary(o.m, o.k, o.iters);
        if (o.bmma) run_bmma(p, o.bmma_loops, o.iters);

        std::cout << "\nGA102-ROM completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
