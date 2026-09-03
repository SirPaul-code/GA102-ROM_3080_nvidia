#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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
};

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--m" || a == "--k" || a == "--iters") && i + 1 < argc) {
            int v = std::stoi(argv[++i]);
            if (a == "--m") o.m = v;
            else if (a == "--k") o.k = v;
            else o.iters = v;
        } else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V3 exact BMMA ternary GEMV\n"
                      << "  --m N      output rows, multiple of 16 (default 8192)\n"
                      << "  --k N      input width, multiple of 256 (default 8192)\n"
                      << "  --iters N  timing iterations (default 100)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + a);
        }
    }
    if (o.m <= 0 || o.k <= 0 || o.iters <= 0) throw std::runtime_error("Arguments must be positive");
    if ((o.m & 15) != 0) throw std::runtime_error("V3 requires M to be a multiple of 16");
    if ((o.k & 255) != 0) throw std::runtime_error("V3 requires K to be a multiple of 256");
    return o;
}

static float time_ms(void (*launch)(void*), void* ctx, int warmup, int iters) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < warmup; ++i) launch(ctx);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch(ctx);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms / iters;
}

__device__ __forceinline__ void bmma_and_popc_16x8x256(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int& c0, int& c1, int& c2, int& c3) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// Exact W1.58/A8 GEMV using Ampere binary Tensor Cores.
//
// One warp computes 16 output rows. K is traversed in 256-bit chunks.
// Ternary W is represented by two one-bit matrices: pos=(W==+1), neg=(W==-1).
// Signed int8 activations are represented by eight bitplanes. For each bitplane:
//   dot(W, bitplane) = popc(pos & bitplane) - popc(neg & bitplane)
// and the bitplane results are reconstructed with scales 1,2,...,64,-128.
//
// m16n8k256 necessarily computes eight columns. We replicate the same activation
// bitplane into all eight B columns and consume column 0 only. This is exact but
// intentionally wastes 7/8 of N; future V4 will use those columns for multi-token
// verification / batching instead of throwing them away.
__global__ void gemv_bmma_ternary_a8(
    const uint32_t* __restrict__ pos,
    const uint32_t* __restrict__ neg,
    const uint32_t* __restrict__ xb,
    int32_t* __restrict__ y,
    int M, int K, int words_per_row) {

    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = global_thread >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = warp * 16;
    if (row_base >= M) return;

    const int group = lane >> 2;          // output rows group and group+8
    const int tid4  = lane & 3;           // 32-bit subfragment within a 128-bit half
    const int row0 = row_base + group;
    const int row8 = row0 + 8;
    const int chunks = K >> 8;            // K / 256

    const uint32_t* p0 = pos + (size_t)row0 * words_per_row;
    const uint32_t* p8 = pos + (size_t)row8 * words_per_row;
    const uint32_t* n0 = neg + (size_t)row0 * words_per_row;
    const uint32_t* n8 = neg + (size_t)row8 * words_per_row;

    int result0 = 0;
    int result8 = 0;

#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        int pc0 = 0, pc1 = 0, pc2 = 0, pc3 = 0;
        int nc0 = 0, nc1 = 0, nc2 = 0, nc3 = 0;
        const uint32_t* plane = xb + (size_t)bit * words_per_row;

        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int wbase = chunk * 8;

            // NVIDIA PTX m16n8k256 A fragment layout:
            //   a0: row group,   cols   0..127 selected by tid4
            //   a1: row group+8, cols   0..127 selected by tid4
            //   a2: row group,   cols 128..255 selected by tid4
            //   a3: row group+8, cols 128..255 selected by tid4
            const uint32_t pa0 = p0[wbase + tid4];
            const uint32_t pa1 = p8[wbase + tid4];
            const uint32_t pa2 = p0[wbase + 4 + tid4];
            const uint32_t pa3 = p8[wbase + 4 + tid4];

            const uint32_t na0 = n0[wbase + tid4];
            const uint32_t na1 = n8[wbase + tid4];
            const uint32_t na2 = n0[wbase + 4 + tid4];
            const uint32_t na3 = n8[wbase + 4 + tid4];

            // B is Kx8 column-major for MMA. Because all 8 columns are identical,
            // groupID (the column) does not affect which activation bits are loaded.
            const uint32_t b0 = plane[wbase + tid4];
            const uint32_t b1 = plane[wbase + 4 + tid4];

            bmma_and_popc_16x8x256(pa0, pa1, pa2, pa3, b0, b1, pc0, pc1, pc2, pc3);
            bmma_and_popc_16x8x256(na0, na1, na2, na3, b0, b1, nc0, nc1, nc2, nc3);
        }

        if (tid4 == 0) {
            const int scale = (bit == 7) ? -128 : (1 << bit);
            // For tid4==0: c0 is column 0 of row group; c2 is column 0 of row group+8.
            result0 += (pc0 - nc0) * scale;
            result8 += (pc2 - nc2) * scale;
        }
    }

    if (tid4 == 0) {
        y[row0] = result0;
        y[row8] = result8;
    }
}

// V2-style warp POPC baseline retained in V3 so comparison is from the same binary/run.
__global__ void gemv_popc_warp(
    const uint32_t* __restrict__ pos,
    const uint32_t* __restrict__ neg,
    const uint32_t* __restrict__ xb,
    int32_t* __restrict__ y,
    int M, int words_per_row) {

    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = global_thread >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= M) return;

    const uint32_t* pr = pos + (size_t)warp * words_per_row;
    const uint32_t* nr = neg + (size_t)warp * words_per_row;
    int acc = 0;

#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        const uint32_t* plane = xb + (size_t)bit * words_per_row;
        int count = 0;
        for (int j = lane; j < words_per_row; j += 32) {
            const uint32_t x = plane[j];
            count += __popc(pr[j] & x) - __popc(nr[j] & x);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            count += __shfl_down_sync(0xffffffffu, count, offset);
        if (lane == 0) {
            const int scale = (bit == 7) ? -128 : (1 << bit);
            acc += count * scale;
        }
    }
    if (lane == 0) y[warp] = acc;
}

static void cpu_reference(const std::vector<int8_t>& w,
                          const std::vector<int8_t>& x,
                          std::vector<int32_t>& y,
                          int M, int K) {
    for (int r = 0; r < M; ++r) {
        int32_t acc = 0;
        const int8_t* wr = w.data() + (size_t)r * K;
        for (int c = 0; c < K; ++c) acc += (int32_t)wr[c] * (int32_t)x[c];
        y[r] = acc;
    }
}

static bool compare_exact(const std::vector<int32_t>& ref,
                          const std::vector<int32_t>& got,
                          const char* name) {
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != got[i]) {
            std::cerr << name << " mismatch row " << i << ": ref=" << ref[i]
                      << " got=" << got[i] << "\n";
            return false;
        }
    }
    return true;
}

struct LaunchCtx {
    const uint32_t* pos;
    const uint32_t* neg;
    const uint32_t* xb;
    int32_t* y;
    int M;
    int K;
    int words;
    int blocks;
    int threads;
};

static void launch_bmma(void* p) {
    auto& c = *reinterpret_cast<LaunchCtx*>(p);
    gemv_bmma_ternary_a8<<<c.blocks, c.threads>>>(c.pos, c.neg, c.xb, c.y, c.M, c.K, c.words);
}

static void launch_popc(void* p) {
    auto& c = *reinterpret_cast<LaunchCtx*>(p);
    const int warps_per_block = c.threads / 32;
    const int blocks = (c.M + warps_per_block - 1) / warps_per_block;
    gemv_popc_warp<<<blocks, c.threads>>>(c.pos, c.neg, c.xb, c.y, c.M, c.words);
}

int main(int argc, char** argv) {
    try {
        Options o = parse_args(argc, argv);

        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 8 || prop.minor != 6 || std::string(prop.name).find("RTX 3080") == std::string::npos) {
            std::cerr << "V3 is intentionally restricted to RTX 3080 / SM86; found "
                      << prop.name << " sm_" << prop.major << prop.minor << "\n";
            return 3;
        }

        const int M = o.m;
        const int K = o.k;
        const int words = K / 32;

        std::cout << "GA102-ROM V3: exact ternary BMMA GEMV\n"
                  << "GPU               : " << prop.name << "\n"
                  << "shape             : M=" << M << " K=" << K << "\n"
                  << "BMMA tile         : m16n8k256 b1 x b1 AND.POPC\n"
                  << "mapping           : 1 warp / 16 rows, N=8 replicated activation columns\n";

        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> wd(-1, 1);
        std::uniform_int_distribution<int> xd(-127, 127);
        std::vector<int8_t> hw((size_t)M * K), hx(K);
        for (auto& v : hw) v = (int8_t)wd(rng);
        for (auto& v : hx) v = (int8_t)xd(rng);

        std::vector<int32_t> ref(M), got(M);
        cpu_reference(hw, hx, ref, M, K);

        std::vector<uint32_t> hpos((size_t)M * words, 0u);
        std::vector<uint32_t> hneg((size_t)M * words, 0u);
        std::vector<uint32_t> hxb((size_t)8 * words, 0u);

        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < K; ++c) {
                const int8_t v = hw[(size_t)r * K + c];
                const uint32_t mask = 1u << (c & 31);
                if (v > 0) hpos[(size_t)r * words + (c >> 5)] |= mask;
                else if (v < 0) hneg[(size_t)r * words + (c >> 5)] |= mask;
            }
        }
        for (int c = 0; c < K; ++c) {
            const uint8_t u = (uint8_t)hx[c];
            for (int bit = 0; bit < 8; ++bit)
                if ((u >> bit) & 1u) hxb[(size_t)bit * words + (c >> 5)] |= 1u << (c & 31);
        }

        uint32_t *dpos = nullptr, *dneg = nullptr, *dxb = nullptr;
        int32_t* dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dpos, hpos.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dneg, hneg.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dxb, hxb.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dy, (size_t)M * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(dpos, hpos.data(), hpos.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dneg, hneg.data(), hneg.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dxb, hxb.data(), hxb.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

        const int threads = 128; // 4 warps/block
        const int bmma_warps = M / 16;
        const int bmma_blocks = (bmma_warps + 3) / 4;
        LaunchCtx ctx{dpos, dneg, dxb, dy, M, K, words, bmma_blocks, threads};

        launch_popc(&ctx);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got.data(), dy, (size_t)M * sizeof(int32_t), cudaMemcpyDeviceToHost));
        const bool popc_ok = compare_exact(ref, got, "POPC");
        if (!popc_ok) return 4;
        const float popc_ms = time_ms(launch_popc, &ctx, 10, o.iters);

        CUDA_CHECK(cudaMemset(dy, 0, (size_t)M * sizeof(int32_t)));
        launch_bmma(&ctx);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got.data(), dy, (size_t)M * sizeof(int32_t), cudaMemcpyDeviceToHost));
        const bool bmma_ok = compare_exact(ref, got, "BMMA");
        if (!bmma_ok) {
            std::cerr << "BMMA correctness failed. Fragment packing/layout must be fixed before any performance number is accepted.\n";
            return 5;
        }
        const float bmma_ms = time_ms(launch_bmma, &ctx, 10, o.iters);

        const double logical_macs = (double)M * K;
        const double weight_bytes = (double)(hpos.size() + hneg.size()) * sizeof(uint32_t);
        const auto print_row = [&](const char* name, float ms) {
            const double gmacs = logical_macs / (ms * 1e6);
            const double weight_gbps = weight_bytes / (ms * 1e6);
            std::cout << std::left << std::setw(24) << name
                      << " correctness=PASS  " << std::right << std::fixed << std::setprecision(3)
                      << ms << " ms  " << std::setprecision(1) << gmacs << " GMAC/s  "
                      << "weight-read~" << weight_gbps << " GB/s\n";
        };

        std::cout << "\n=== Exact comparison ===\n";
        print_row("W1x2 CUDA POPC warp", popc_ms);
        print_row("W1x2 BMMA TensorCore", bmma_ms);
        std::cout << "BMMA vs POPC speedup : " << std::fixed << std::setprecision(2)
                  << (double)popc_ms / bmma_ms << "x\n";

        const double bmma_ops_per_warp = (double)(K / 256) * 8.0 * 2.0;
        const double total_bmma = (double)bmma_warps * bmma_ops_per_warp;
        const double raw_bit_contrib = total_bmma * 16.0 * 8.0 * 256.0;
        std::cout << "BMMA instructions/run : " << std::fixed << std::setprecision(0) << total_bmma << "\n"
                  << "raw b1 contributions  : " << std::setprecision(2)
                  << raw_bit_contrib / (bmma_ms * 1e9) << " TOP/s\n"
                  << "useful N columns       : 1 / 8 (V3 intentionally wastes seven)\n";

        CUDA_CHECK(cudaFree(dpos));
        CUDA_CHECK(cudaFree(dneg));
        CUDA_CHECK(cudaFree(dxb));
        CUDA_CHECK(cudaFree(dy));

        std::cout << "\nV3 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
