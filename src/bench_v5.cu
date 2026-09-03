#include <cuda_runtime.h>

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
    int iters = 200;
};

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--m" && i + 1 < argc) o.m = std::stoi(argv[++i]);
        else if (a == "--k" && i + 1 < argc) o.k = std::stoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V5 end-to-end BMMA pipeline\n"
                      << "  --m N      output rows, multiple of 16 (default 8192)\n"
                      << "  --k N      input width, multiple of 256 (default 8192)\n"
                      << "  --iters N  timing iterations (default 200)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + a);
        }
    }
    if (o.m <= 0 || o.k <= 0 || o.iters <= 0)
        throw std::runtime_error("Arguments must be positive");
    if ((o.m & 15) != 0)
        throw std::runtime_error("V5 requires M to be a multiple of 16");
    if ((o.k & 255) != 0)
        throw std::runtime_error("V5 requires K to be a multiple of 256");
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

// Dynamic A8 x 8 -> exact BMMA B-fragment pack.
// One block owns one 256-K chunk. Its 8 warps each ballot one 32-element word.
// Shared memory holds [column][bitplane][word-within-256], then threads 0..31
// emit the exact uint2 fragment expected by m16n8k256 for all eight columns.
__global__ void pack_a8x8_to_bmma_b(const int8_t* __restrict__ x8,
                                    uint2* __restrict__ bp8,
                                    int K) {
    __shared__ uint32_t planes[8][8][8];

    const int chunk = blockIdx.x;
    const int word = threadIdx.x >> 5; // 0..7
    const int lane = threadIdx.x & 31;
    const int kidx = chunk * 256 + word * 32 + lane;

#pragma unroll
    for (int col = 0; col < 8; ++col) {
        const uint8_t u = (uint8_t)x8[(size_t)col * K + kidx];
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = __ballot_sync(0xffffffffu, ((u >> bit) & 1u) != 0u);
            if (lane == 0)
                planes[col][bit][word] = mask;
        }
    }

    __syncthreads();

    if (threadIdx.x < 32) {
        const int frag_lane = threadIdx.x;
        const int col = frag_lane >> 2;
        const int tid4 = frag_lane & 3;
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            bp8[((size_t)chunk * 8 + bit) * 32 + frag_lane] =
                make_uint2(planes[col][bit][tid4],
                           planes[col][bit][4 + tid4]);
        }
    }
}

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
#pragma unroll
            for (int col = 0; col < 8; ++col)
                acc[col] += wi * (int)x8[(size_t)col * K + c];
        }
        for (int col = 0; col < 8; ++col)
            y8[(size_t)r * 8 + col] = acc[col];
    }
}

static bool exact_equal(const std::vector<int32_t>& ref,
                        const std::vector<int32_t>& got) {
    if (ref.size() != got.size()) return false;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != got[i]) {
            std::cerr << "mismatch index " << i << ": ref=" << ref[i]
                      << " got=" << got[i] << "\n";
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
            std::cerr << "V5 is intentionally restricted to RTX 3080 / SM86; found "
                      << prop.name << " sm_" << prop.major << prop.minor << "\n";
            return 3;
        }

        const int M = o.m;
        const int K = o.k;
        const int words = K / 32;
        const int chunks = K / 256;
        const int tiles = M / 16;

        std::cout << "GA102-ROM V5: end-to-end dynamic activation pack + BMMA N=8\n"
                  << "GPU               : " << prop.name << "\n"
                  << "shape             : M=" << M << " K=" << K << " N=8\n"
                  << "input activation  : raw signed int8 [8,K]\n"
                  << "weight format     : offline BMMA-native uint4/lane\n"
                  << "dynamic B pack    : GPU ballot -> exact uint2/lane fragment\n";

        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> wd(-1, 1);
        std::uniform_int_distribution<int> xd(-127, 127);

        std::vector<int8_t> hw((size_t)M * K);
        std::vector<int8_t> hx8((size_t)8 * K);
        for (auto& v : hw) v = (int8_t)wd(rng);
        for (auto& v : hx8) v = (int8_t)xd(rng);

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

        // Fixed-model/offline compiler: row-major ternary masks -> exact A fragments.
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

        std::vector<int32_t> ref8((size_t)M * 8);
        cpu_reference_batch8(hw, hx8, ref8, M, K);

        int8_t* dx8 = nullptr;
        uint4 *dpp = nullptr, *dpn = nullptr;
        uint2* dbp8 = nullptr;
        int32_t* dy8 = nullptr;

        const size_t b_frag_count = (size_t)chunks * 8 * 32;
        CUDA_CHECK(cudaMalloc(&dx8, hx8.size() * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&dpp, hpp.size() * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&dpn, hpn.size() * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&dbp8, b_frag_count * sizeof(uint2)));
        CUDA_CHECK(cudaMalloc(&dy8, (size_t)M * 8 * sizeof(int32_t)));

        CUDA_CHECK(cudaMemcpy(dx8, hx8.data(), hx8.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpp, hpp.data(), hpp.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpn, hpn.data(), hpn.size() * sizeof(uint4), cudaMemcpyHostToDevice));

        const int pack_threads = 256;
        const int bmma_threads = 128;
        const int warps_per_block = bmma_threads / 32;
        const int bmma_blocks = (tiles + warps_per_block - 1) / warps_per_block;

        auto launch_pack = [&] {
            pack_a8x8_to_bmma_b<<<chunks, pack_threads>>>(dx8, dbp8, K);
        };
        auto launch_bmma = [&] {
            bmma_batch8_packed<<<bmma_blocks, bmma_threads>>>(dpp, dpn, dbp8, dy8, tiles, chunks);
        };
        auto launch_pipeline = [&] {
            pack_a8x8_to_bmma_b<<<chunks, pack_threads>>>(dx8, dbp8, K);
            bmma_batch8_packed<<<bmma_blocks, bmma_threads>>>(dpp, dpn, dbp8, dy8, tiles, chunks);
        };

        launch_pipeline();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int32_t> got8((size_t)M * 8);
        CUDA_CHECK(cudaMemcpy(got8.data(), dy8, got8.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (!exact_equal(ref8, got8)) {
            std::cerr << "V5 exact pipeline correctness failed.\n";
            return 4;
        }

        // Keep a valid packed B in dbp8 before timing BMMA alone.
        launch_pack();
        CUDA_CHECK(cudaDeviceSynchronize());

        const float pack_ms = time_ms(launch_pack, 20, o.iters);
        const float bmma_ms = time_ms(launch_bmma, 20, o.iters);
        const float pipeline_ms = time_ms(launch_pipeline, 20, o.iters);

        const double logical = (double)M * K * 8.0;
        const double raw_x_bytes = (double)hx8.size();
        const double packed_b_bytes = (double)b_frag_count * sizeof(uint2);
        const double pack_io_bytes = raw_x_bytes + packed_b_bytes;
        const double logical_gmacs = logical / (pipeline_ms * 1e6);

        std::cout << "\n=== V5 exact end-to-end N=8 ===\n"
                  << "correctness          : PASS\n"
                  << std::fixed << std::setprecision(4)
                  << "GPU B-fragment pack   : " << pack_ms << " ms\n"
                  << "BMMA matrix kernel    : " << bmma_ms << " ms\n"
                  << "pack + BMMA pipeline  : " << pipeline_ms << " ms\n"
                  << "pipeline throughput   : " << std::setprecision(1) << logical_gmacs << " GMAC/s\n"
                  << std::setprecision(4)
                  << "amortized/vector      : " << pipeline_ms / 8.0 << " ms equivalent\n"
                  << "pack share            : " << std::setprecision(1)
                  << (100.0 * pack_ms / pipeline_ms) << " % of pipeline time\n"
                  << "pack input             : " << (raw_x_bytes / 1024.0) << " KiB\n"
                  << "packed B output        : " << (packed_b_bytes / 1024.0) << " KiB\n"
                  << "pack effective IO      : " << std::setprecision(1)
                  << (pack_io_bytes / (pack_ms * 1e6)) << " GB/s\n";

        std::cout << "\nInterpretation guardrail:\n"
                  << "  amortized/vector is throughput-equivalent, not one-token latency.\n"
                  << "  V5 includes dynamic int8->BMMA B-fragment packing on the GPU.\n";

        CUDA_CHECK(cudaFree(dx8));
        CUDA_CHECK(cudaFree(dpp));
        CUDA_CHECK(cudaFree(dpn));
        CUDA_CHECK(cudaFree(dbp8));
        CUDA_CHECK(cudaFree(dy8));

        std::cout << "\nV5 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
