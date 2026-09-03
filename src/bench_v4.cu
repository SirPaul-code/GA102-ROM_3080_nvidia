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
};

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--m" && i + 1 < argc) o.m = std::stoi(argv[++i]);
        else if (a == "--k" && i + 1 < argc) o.k = std::stoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V4 BMMA-native packed benchmark\n"
                      << "  --m N      output rows, multiple of 16 (default 8192)\n"
                      << "  --k N      input width, multiple of 256 (default 8192)\n"
                      << "  --iters N  timing iterations (default 100)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + a);
        }
    }
    if (o.m <= 0 || o.k <= 0 || o.iters <= 0)
        throw std::runtime_error("Arguments must be positive");
    if ((o.m & 15) != 0)
        throw std::runtime_error("V4 requires M to be a multiple of 16");
    if ((o.k & 255) != 0)
        throw std::runtime_error("V4 requires K to be a multiple of 256");
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

__device__ __forceinline__ int warp_sum_i32(int v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
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

// V2 best single-token baseline: each warp owns one row, loads each ternary
// weight word exactly once, and reuses it across all eight A8 bitplanes.
__global__ void popc_single_warp(
    const uint32_t* __restrict__ pos,
    const uint32_t* __restrict__ neg,
    const uint32_t* __restrict__ xb,
    int32_t* __restrict__ y,
    int M, int words) {

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

// V4 single-token BMMA path.
//
// Difference from V3:
//   * A/weights are prepacked in exact lane-fragment order, so every warp loads
//     a contiguous 512-byte positive fragment and 512-byte negative fragment.
//   * each A fragment is loaded ONCE per 256-K chunk and reused for all 8 A8
//     activation bitplanes;
//   * BMMA accumulators are local to one chunk/bit, removing the long dependent
//     BMMA accumulator chain across all K chunks;
//   * only four lanes load the replicated B fragment and broadcast to the warp.
__global__ void bmma_single_packed(
    const uint4* __restrict__ posp,
    const uint4* __restrict__ negp,
    const uint32_t* __restrict__ xb,
    int32_t* __restrict__ y,
    int tiles, int chunks, int words) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile = tid >> 5;
    const int lane = threadIdx.x & 31;
    if (tile >= tiles) return;

    const int group = lane >> 2;
    const int tid4 = lane & 3;
    int out0 = 0;
    int out8 = 0;

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const size_t frag_idx = ((size_t)tile * chunks + chunk) * 32 + lane;
        const uint4 p = posp[frag_idx];
        const uint4 n = negp[frag_idx];
        const int wbase = chunk * 8;

#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t local_b0 = 0;
            uint32_t local_b1 = 0;
            if (group == 0) {
                const uint32_t* plane = xb + (size_t)bit * words;
                local_b0 = plane[wbase + tid4];
                local_b1 = plane[wbase + 4 + tid4];
            }
            const uint32_t b0 = __shfl_sync(0xffffffffu, local_b0, tid4);
            const uint32_t b1 = __shfl_sync(0xffffffffu, local_b1, tid4);

            int pc0 = 0, pc1 = 0, pc2 = 0, pc3 = 0;
            int nc0 = 0, nc1 = 0, nc2 = 0, nc3 = 0;
            bmma_and_popc_16x8x256(p.x, p.y, p.z, p.w, b0, b1,
                                   pc0, pc1, pc2, pc3);
            bmma_and_popc_16x8x256(n.x, n.y, n.z, n.w, b0, b1,
                                   nc0, nc1, nc2, nc3);

            if (tid4 == 0) {
                const int scale = (bit == 7) ? -128 : (1 << bit);
                out0 += (pc0 - nc0) * scale;
                out8 += (pc2 - nc2) * scale;
            }
        }
    }

    if (tid4 == 0) {
        const int row0 = tile * 16 + group;
        y[row0] = out0;
        y[row0 + 8] = out8;
    }
}

// Batch-8 POPC baseline. Same fixed weights feed eight activation vectors.
// Each warp owns one output row and reuses each loaded weight word across all
// eight columns. This is the fair software baseline for BMMA's native N=8 tile.
__global__ void popc_batch8_warp(
    const uint32_t* __restrict__ pos,
    const uint32_t* __restrict__ neg,
    const uint32_t* __restrict__ xb8,
    int32_t* __restrict__ y,
    int M, int words) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = tid >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const uint32_t* pr = pos + (size_t)row * words;
    const uint32_t* nr = neg + (size_t)row * words;
    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    int acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

    for (int j = lane; j < words; j += 32) {
        const uint32_t p = pr[j];
        const uint32_t n = nr[j];
        int tmp[8] = {0,0,0,0,0,0,0,0};
#pragma unroll
        for (int col = 0; col < 8; ++col) {
#pragma unroll
            for (int bit = 0; bit < 8; ++bit) {
                const uint32_t a = xb8[((size_t)col * 8 + bit) * words + j];
                const int cnt = __popc(p & a) - __popc(n & a);
                const int scale = (bit == 7) ? -128 : (1 << bit);
                tmp[col] += cnt * scale;
            }
        }
        acc0 += tmp[0]; acc1 += tmp[1]; acc2 += tmp[2]; acc3 += tmp[3];
        acc4 += tmp[4]; acc5 += tmp[5]; acc6 += tmp[6]; acc7 += tmp[7];
    }

    acc0 = warp_sum_i32(acc0); acc1 = warp_sum_i32(acc1);
    acc2 = warp_sum_i32(acc2); acc3 = warp_sum_i32(acc3);
    acc4 = warp_sum_i32(acc4); acc5 = warp_sum_i32(acc5);
    acc6 = warp_sum_i32(acc6); acc7 = warp_sum_i32(acc7);

    if (lane == 0) {
        int32_t* out = y + (size_t)row * 8;
        out[0] = acc0; out[1] = acc1; out[2] = acc2; out[3] = acc3;
        out[4] = acc4; out[5] = acc5; out[6] = acc6; out[7] = acc7;
    }
}

// Native N=8 BMMA. A is identical to the single-token packed format. B has
// eight distinct activation columns, prepacked into the exact two-register
// fragment required by each lane. All 16x8 outputs of every BMMA are useful.
__global__ void bmma_batch8_packed(
    const uint4* __restrict__ posp,
    const uint4* __restrict__ negp,
    const uint2* __restrict__ bp8,
    int32_t* __restrict__ y,
    int tiles, int chunks) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile = tid >> 5;
    const int lane = threadIdx.x & 31;
    if (tile >= tiles) return;

    int out0 = 0, out1 = 0, out2 = 0, out3 = 0;

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const size_t frag_idx = ((size_t)tile * chunks + chunk) * 32 + lane;
        const uint4 p = posp[frag_idx];
        const uint4 n = negp[frag_idx];

#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const uint2 b = bp8[((size_t)chunk * 8 + bit) * 32 + lane];
            int pc0 = 0, pc1 = 0, pc2 = 0, pc3 = 0;
            int nc0 = 0, nc1 = 0, nc2 = 0, nc3 = 0;
            bmma_and_popc_16x8x256(p.x, p.y, p.z, p.w, b.x, b.y,
                                   pc0, pc1, pc2, pc3);
            bmma_and_popc_16x8x256(n.x, n.y, n.z, n.w, b.x, b.y,
                                   nc0, nc1, nc2, nc3);
            const int scale = (bit == 7) ? -128 : (1 << bit);
            out0 += (pc0 - nc0) * scale;
            out1 += (pc1 - nc1) * scale;
            out2 += (pc2 - nc2) * scale;
            out3 += (pc3 - nc3) * scale;
        }
    }

    const int group = lane >> 2;
    const int tid4 = lane & 3;
    const int row0 = tile * 16 + group;
    const int row8 = row0 + 8;
    const int col0 = tid4 * 2;
    const int col1 = col0 + 1;

    y[(size_t)row0 * 8 + col0] = out0;
    y[(size_t)row0 * 8 + col1] = out1;
    y[(size_t)row8 * 8 + col0] = out2;
    y[(size_t)row8 * 8 + col1] = out3;
}

static void cpu_reference_batch8(const std::vector<int8_t>& w,
                                 const std::vector<int8_t>& x8,
                                 std::vector<int32_t>& y8,
                                 int M, int K) {
    for (int r = 0; r < M; ++r) {
        int32_t acc[8] = {0,0,0,0,0,0,0,0};
        const int8_t* wr = w.data() + (size_t)r * K;
        for (int c = 0; c < K; ++c) {
            const int wi = (int)wr[c];
            acc[0] += wi * (int)x8[(size_t)0 * K + c];
            acc[1] += wi * (int)x8[(size_t)1 * K + c];
            acc[2] += wi * (int)x8[(size_t)2 * K + c];
            acc[3] += wi * (int)x8[(size_t)3 * K + c];
            acc[4] += wi * (int)x8[(size_t)4 * K + c];
            acc[5] += wi * (int)x8[(size_t)5 * K + c];
            acc[6] += wi * (int)x8[(size_t)6 * K + c];
            acc[7] += wi * (int)x8[(size_t)7 * K + c];
        }
        for (int col = 0; col < 8; ++col)
            y8[(size_t)r * 8 + col] = acc[col];
    }
}

static bool exact_equal(const std::vector<int32_t>& ref,
                        const std::vector<int32_t>& got,
                        const char* label) {
    if (ref.size() != got.size()) return false;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != got[i]) {
            std::cerr << label << " mismatch index " << i
                      << ": ref=" << ref[i] << " got=" << got[i] << "\n";
            return false;
        }
    }
    return true;
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
            std::cerr << "V4 is intentionally restricted to RTX 3080 / SM86; found "
                      << prop.name << " sm_" << prop.major << prop.minor << "\n";
            return 3;
        }

        const int M = o.m;
        const int K = o.k;
        const int words = K / 32;
        const int chunks = K / 256;
        const int tiles = M / 16;

        std::cout << "GA102-ROM V4: BMMA-native weight layout\n"
                  << "GPU               : " << prop.name << "\n"
                  << "shape             : M=" << M << " K=" << K << "\n"
                  << "weight format     : exact PTX A-fragment order (uint4/lane)\n"
                  << "single-token      : packed BMMA vs V2-style POPC\n"
                  << "batch-8           : all BMMA N=8 columns useful\n";

        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> wd(-1, 1);
        std::uniform_int_distribution<int> xd(-127, 127);

        std::vector<int8_t> hw((size_t)M * K);
        std::vector<int8_t> hx8((size_t)8 * K);
        for (auto& v : hw) v = (int8_t)wd(rng);
        for (auto& v : hx8) v = (int8_t)xd(rng);

        // Row-major bit masks for the POPC baseline and as the source for the
        // offline BMMA-native weight compiler below.
        std::vector<uint32_t> hpos((size_t)M * words, 0u);
        std::vector<uint32_t> hneg((size_t)M * words, 0u);
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < K; ++c) {
                const int8_t v = hw[(size_t)r * K + c];
                const uint32_t mask = 1u << (c & 31);
                if (v > 0) hpos[(size_t)r * words + (c >> 5)] |= mask;
                else if (v < 0) hneg[(size_t)r * words + (c >> 5)] |= mask;
            }
        }

        // Dynamic activations represented as eight A8 bitplanes per column.
        std::vector<uint32_t> hxb8((size_t)8 * 8 * words, 0u);
        for (int col = 0; col < 8; ++col) {
            for (int c = 0; c < K; ++c) {
                const uint8_t u = (uint8_t)hx8[(size_t)col * K + c];
                for (int bit = 0; bit < 8; ++bit) {
                    if ((u >> bit) & 1u)
                        hxb8[((size_t)col * 8 + bit) * words + (c >> 5)] |= 1u << (c & 31);
                }
            }
        }

        // Offline fixed-model compiler: row-major W masks -> exact A-fragment
        // layout consumed by one m16n8k256 warp. A warp's 32 uint4 fragments are
        // contiguous in memory (512 B), eliminating V3's strided row loads.
        const size_t packed_frags = (size_t)tiles * chunks * 32;
        std::vector<uint4> hpp(packed_frags);
        std::vector<uint4> hpn(packed_frags);
        for (int tile = 0; tile < tiles; ++tile) {
            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int wbase = chunk * 8;
                for (int lane = 0; lane < 32; ++lane) {
                    const int group = lane >> 2;
                    const int tid4 = lane & 3;
                    const int row0 = tile * 16 + group;
                    const int row8 = row0 + 8;
                    const size_t idx = ((size_t)tile * chunks + chunk) * 32 + lane;
                    hpp[idx] = make_uint4(
                        hpos[(size_t)row0 * words + wbase + tid4],
                        hpos[(size_t)row8 * words + wbase + tid4],
                        hpos[(size_t)row0 * words + wbase + 4 + tid4],
                        hpos[(size_t)row8 * words + wbase + 4 + tid4]);
                    hpn[idx] = make_uint4(
                        hneg[(size_t)row0 * words + wbase + tid4],
                        hneg[(size_t)row8 * words + wbase + tid4],
                        hneg[(size_t)row0 * words + wbase + 4 + tid4],
                        hneg[(size_t)row8 * words + wbase + 4 + tid4]);
                }
            }
        }

        // B/activation native fragment pack for the N=8 experiment. This pack is
        // dynamic in a real runtime; V4 measures the matrix kernel separately from
        // activation quantization/packing, just as V1-V3 did.
        std::vector<uint2> hbp8((size_t)chunks * 8 * 32);
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int wbase = chunk * 8;
            for (int bit = 0; bit < 8; ++bit) {
                for (int lane = 0; lane < 32; ++lane) {
                    const int col = lane >> 2;
                    const int tid4 = lane & 3;
                    const uint32_t* plane = hxb8.data() + ((size_t)col * 8 + bit) * words;
                    hbp8[((size_t)chunk * 8 + bit) * 32 + lane] =
                        make_uint2(plane[wbase + tid4], plane[wbase + 4 + tid4]);
                }
            }
        }

        // Exact CPU reference for all eight columns. Single-token reference is col 0.
        std::vector<int32_t> ref8((size_t)M * 8);
        cpu_reference_batch8(hw, hx8, ref8, M, K);
        std::vector<int32_t> ref1(M);
        for (int r = 0; r < M; ++r) ref1[r] = ref8[(size_t)r * 8];

        uint32_t *dpos = nullptr, *dneg = nullptr, *dxb8 = nullptr;
        uint4 *dpp = nullptr, *dpn = nullptr;
        uint2* dbp8 = nullptr;
        int32_t *dy1 = nullptr, *dy8 = nullptr;

        CUDA_CHECK(cudaMalloc(&dpos, hpos.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dneg, hneg.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dxb8, hxb8.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dpp, hpp.size() * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&dpn, hpn.size() * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&dbp8, hbp8.size() * sizeof(uint2)));
        CUDA_CHECK(cudaMalloc(&dy1, (size_t)M * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&dy8, (size_t)M * 8 * sizeof(int32_t)));

        CUDA_CHECK(cudaMemcpy(dpos, hpos.data(), hpos.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dneg, hneg.data(), hneg.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dxb8, hxb8.data(), hxb8.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpp, hpp.data(), hpp.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpn, hpn.data(), hpn.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dbp8, hbp8.data(), hbp8.size() * sizeof(uint2), cudaMemcpyHostToDevice));

        const int threads = 128;
        const int warps_per_block = threads / 32;
        const int blocks_popc = (M + warps_per_block - 1) / warps_per_block;
        const int blocks_bmma = (tiles + warps_per_block - 1) / warps_per_block;

        auto launch_popc1 = [&] {
            popc_single_warp<<<blocks_popc, threads>>>(dpos, dneg, dxb8, dy1, M, words);
        };
        auto launch_bmma1 = [&] {
            bmma_single_packed<<<blocks_bmma, threads>>>(dpp, dpn, dxb8, dy1, tiles, chunks, words);
        };
        auto launch_popc8 = [&] {
            popc_batch8_warp<<<blocks_popc, threads>>>(dpos, dneg, dxb8, dy8, M, words);
        };
        auto launch_bmma8 = [&] {
            bmma_batch8_packed<<<blocks_bmma, threads>>>(dpp, dpn, dbp8, dy8, tiles, chunks);
        };

        std::vector<int32_t> got1(M);
        std::vector<int32_t> got8((size_t)M * 8);

        launch_popc1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got1.data(), dy1, got1.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (!exact_equal(ref1, got1, "POPC single")) return 4;

        launch_bmma1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got1.data(), dy1, got1.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (!exact_equal(ref1, got1, "BMMA packed single")) return 5;

        launch_popc8(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got8.data(), dy8, got8.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (!exact_equal(ref8, got8, "POPC batch8")) return 6;

        launch_bmma8(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got8.data(), dy8, got8.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (!exact_equal(ref8, got8, "BMMA packed batch8")) return 7;

        const float popc1_ms = time_ms(launch_popc1, 10, o.iters);
        const float bmma1_ms = time_ms(launch_bmma1, 10, o.iters);
        const float popc8_ms = time_ms(launch_popc8, 10, o.iters);
        const float bmma8_ms = time_ms(launch_bmma8, 10, o.iters);

        const double logical1 = (double)M * K;
        const double logical8 = logical1 * 8.0;
        const double weight_bytes = (double)(hpos.size() + hneg.size()) * sizeof(uint32_t);

        auto row = [&](const char* name, float ms, double logical) {
            const double gmacs = logical / (ms * 1e6);
            const double wgb = weight_bytes / (ms * 1e6);
            std::cout << std::left << std::setw(28) << name
                      << " correctness=PASS  " << std::right << std::fixed << std::setprecision(3)
                      << ms << " ms  " << std::setprecision(1) << gmacs << " GMAC/s"
                      << "  physical-W~" << wgb << " GB/s\n";
        };

        std::cout << "\n=== V4 single-token exact ===\n";
        row("W1x2 CUDA POPC", popc1_ms, logical1);
        row("W1x2 BMMA packed", bmma1_ms, logical1);
        std::cout << "BMMA/POPC speed ratio : " << std::fixed << std::setprecision(2)
                  << (double)popc1_ms / bmma1_ms << "x\n";

        std::cout << "\n=== V4 eight-column exact ===\n";
        row("W1x2 CUDA POPC x8", popc8_ms, logical8);
        row("W1x2 BMMA packed N=8", bmma8_ms, logical8);
        std::cout << "BMMA/POPC speed ratio : " << std::fixed << std::setprecision(2)
                  << (double)popc8_ms / bmma8_ms << "x\n";
        std::cout << "per-vector BMMA time  : " << std::setprecision(4) << bmma8_ms / 8.0 << " ms equivalent\n"
                  << "B-fragment pack        : excluded from kernel timing (measured separately in a future runtime pass)\n";

        const double bmma_instr = (double)tiles * chunks * 8.0 * 2.0;
        const double raw_contrib = bmma_instr * 16.0 * 8.0 * 256.0;
        std::cout << "\nBMMA instructions/run : " << std::fixed << std::setprecision(0) << bmma_instr << "\n"
                  << "batch8 raw b1 rate     : " << std::setprecision(2)
                  << raw_contrib / (bmma8_ms * 1e9) << " TOP/s\n";

        CUDA_CHECK(cudaFree(dpos));
        CUDA_CHECK(cudaFree(dneg));
        CUDA_CHECK(cudaFree(dxb8));
        CUDA_CHECK(cudaFree(dpp));
        CUDA_CHECK(cudaFree(dpn));
        CUDA_CHECK(cudaFree(dbp8));
        CUDA_CHECK(cudaFree(dy1));
        CUDA_CHECK(cudaFree(dy8));

        std::cout << "\nV4 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
