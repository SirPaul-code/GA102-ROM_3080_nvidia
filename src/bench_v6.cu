#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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

struct Options {
    int iters = 200;
    int ring_mib = 64;
};

struct Shape {
    const char* name;
    int M;
    int K;
};

struct Result {
    std::string name;
    int M = 0;
    int K = 0;
    int copies = 0;
    double weight_mib = 0.0;
    float single_matrix_ms = 0.0f;
    float single_pipeline_ms = 0.0f;
    float n8_matrix_ms = 0.0f;
    float n8_pipeline_ms = 0.0f;
};

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "--ring-mib" && i + 1 < argc) o.ring_mib = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V6 real BitNet 2B projection benchmark\n"
                      << "  --iters N      timing iterations/shape (default 200)\n"
                      << "  --ring-mib N   rotating packed-weight ring per backend (default 64 MiB)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + a);
        }
    }
    if (o.iters <= 0 || o.ring_mib < 16)
        throw std::runtime_error("iters must be positive and ring-mib >= 16");
    return o;
}

static float time_indexed(const std::function<void(int)>& launch,
                          int warmup, int iters, int copies) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));

    for (int i = 0; i < warmup; ++i) launch(i % copies);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch(i % copies);
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

// Raw A8 vector -> eight row-major bitplanes for the single-token POPC path.
__global__ void pack_a8_to_planes(const int8_t* __restrict__ x,
                                  uint32_t* __restrict__ xb,
                                  int words) {
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

__global__ void popc_single_warp(const uint32_t* __restrict__ pos,
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

// Same dynamic N=8 activation pack proven exact in V5.
__global__ void pack_a8x8_to_bmma_b(const int8_t* __restrict__ x8,
                                    uint2* __restrict__ bp8,
                                    int K) {
    __shared__ uint32_t planes[8][8][8];

    const int chunk = blockIdx.x;
    const int word = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int kidx = chunk * 256 + word * 32 + lane;

#pragma unroll
    for (int col = 0; col < 8; ++col) {
        const uint8_t u = (uint8_t)x8[(size_t)col * K + kidx];
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = __ballot_sync(0xffffffffu, ((u >> bit) & 1u) != 0u);
            if (lane == 0) planes[col][bit][word] = mask;
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
                make_uint2(planes[col][bit][tid4], planes[col][bit][4 + tid4]);
        }
    }
}

__global__ void bmma_batch8_packed(const uint4* __restrict__ posp,
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

static inline uint32_t next_u32(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    s ^= s >> 16;
    return s;
}

static void make_synthetic(std::vector<int8_t>& w,
                           std::vector<int8_t>& x8,
                           uint32_t seed) {
    uint32_t s = seed;
    for (auto& v : w) v = (int8_t)((int)(next_u32(s) % 3u) - 1);
    for (auto& v : x8) v = (int8_t)((int)(next_u32(s) % 255u) - 127);
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

static Result benchmark_shape(const Shape& s, const Options& o, int ordinal) {
    const int M = s.M;
    const int K = s.K;
    const int words = K / 32;
    const int chunks = K / 256;
    const int tiles = M / 16;

    std::cout << "\n--- " << s.name << "  M=" << M << " K=" << K << " ---\n";
    std::cout << "preparing deterministic ternary weights and exact reference...\n";

    std::vector<int8_t> hw((size_t)M * K);
    std::vector<int8_t> hx8((size_t)8 * K);
    make_synthetic(hw, hx8, 0x12345678u + (uint32_t)ordinal * 0x9e3779b9u);

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
    std::vector<int32_t> ref1(M);
    for (int r = 0; r < M; ++r) ref1[r] = ref8[(size_t)r * 8];

    const size_t one_weight_bytes =
        (hpos.size() + hneg.size()) * sizeof(uint32_t);
    const size_t ring_target = (size_t)o.ring_mib * 1024u * 1024u;
    int copies = (int)((ring_target + one_weight_bytes - 1) / one_weight_bytes);
    copies = std::max(2, std::min(copies, 64));

    uint32_t *dpos = nullptr, *dneg = nullptr, *dxb1 = nullptr;
    uint4 *dpp = nullptr, *dpn = nullptr;
    uint2* dbp8 = nullptr;
    int8_t* dx8 = nullptr;
    int32_t *dy1 = nullptr, *dy8 = nullptr;

    const size_t pos_words_one = hpos.size();
    const size_t packed_one = hpp.size();
    const size_t b_frag_count = (size_t)chunks * 8 * 32;

    CUDA_CHECK(cudaMalloc(&dpos, pos_words_one * copies * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dneg, pos_words_one * copies * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dpp, packed_one * copies * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&dpn, packed_one * copies * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&dxb1, (size_t)8 * words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dbp8, b_frag_count * sizeof(uint2)));
    CUDA_CHECK(cudaMalloc(&dx8, hx8.size()));
    CUDA_CHECK(cudaMalloc(&dy1, (size_t)M * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dy8, (size_t)M * 8 * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(dx8, hx8.data(), hx8.size(), cudaMemcpyHostToDevice));
    for (int c = 0; c < copies; ++c) {
        CUDA_CHECK(cudaMemcpy(dpos + (size_t)c * pos_words_one,
                              hpos.data(), pos_words_one * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dneg + (size_t)c * pos_words_one,
                              hneg.data(), pos_words_one * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpp + (size_t)c * packed_one,
                              hpp.data(), packed_one * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpn + (size_t)c * packed_one,
                              hpn.data(), packed_one * sizeof(uint4), cudaMemcpyHostToDevice));
    }

    const int threads = 128;
    const int warps_per_block = threads / 32;
    const int popc_blocks = (M + warps_per_block - 1) / warps_per_block;
    const int bmma_blocks = (tiles + warps_per_block - 1) / warps_per_block;
    const int pack1_threads = 256;
    const int pack1_blocks = (words + 7) / 8;
    const int pack8_threads = 256;

    auto pack1 = [&] {
        pack_a8_to_planes<<<pack1_blocks, pack1_threads>>>(dx8, dxb1, words);
    };
    auto popc_for = [&](int copy) {
        popc_single_warp<<<popc_blocks, threads>>>(
            dpos + (size_t)copy * pos_words_one,
            dneg + (size_t)copy * pos_words_one,
            dxb1, dy1, M, words);
    };
    auto single_pipeline_for = [&](int copy) {
        pack1();
        popc_for(copy);
    };

    auto pack8 = [&] {
        pack_a8x8_to_bmma_b<<<chunks, pack8_threads>>>(dx8, dbp8, K);
    };
    auto bmma_for = [&](int copy) {
        bmma_batch8_packed<<<bmma_blocks, threads>>>(
            dpp + (size_t)copy * packed_one,
            dpn + (size_t)copy * packed_one,
            dbp8, dy8, tiles, chunks);
    };
    auto n8_pipeline_for = [&](int copy) {
        pack8();
        bmma_for(copy);
    };

    // Exact correctness on copy 0.
    single_pipeline_for(0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> got1(M);
    CUDA_CHECK(cudaMemcpy(got1.data(), dy1, got1.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (!exact_equal(ref1, got1, "single POPC")) std::exit(4);

    n8_pipeline_for(0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> got8((size_t)M * 8);
    CUDA_CHECK(cudaMemcpy(got8.data(), dy8, got8.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (!exact_equal(ref8, got8, "N8 BMMA")) std::exit(5);

    // Keep valid packed activations before matrix-only timings.
    pack1();
    pack8();
    CUDA_CHECK(cudaDeviceSynchronize());

    const int warmup = std::max(16, copies);
    const float single_matrix_ms = time_indexed(
        [&](int copy) { popc_for(copy); }, warmup, o.iters, copies);
    const float single_pipeline_ms = time_indexed(
        [&](int copy) { single_pipeline_for(copy); }, warmup, o.iters, copies);
    const float n8_matrix_ms = time_indexed(
        [&](int copy) { bmma_for(copy); }, warmup, o.iters, copies);
    const float n8_pipeline_ms = time_indexed(
        [&](int copy) { n8_pipeline_for(copy); }, warmup, o.iters, copies);

    const double macs1 = (double)M * K;
    const double macs8 = macs1 * 8.0;
    const double weight_mib = (double)one_weight_bytes / (1024.0 * 1024.0);
    const double single_gmacs = macs1 / (single_pipeline_ms * 1e6);
    const double n8_gmacs = macs8 / (n8_pipeline_ms * 1e6);
    const double single_w_gbps = one_weight_bytes / (single_matrix_ms * 1e6);
    const double n8_w_gbps = one_weight_bytes / (n8_matrix_ms * 1e6);

    std::cout << "correctness        : PASS (single + N=8)\n"
              << "packed W           : " << std::fixed << std::setprecision(3)
              << weight_mib << " MiB, rotating copies=" << copies << "\n"
              << "single matrix      : " << std::setprecision(4) << single_matrix_ms
              << " ms  W~" << std::setprecision(1) << single_w_gbps << " GB/s\n"
              << "single pack+POPC   : " << std::setprecision(4) << single_pipeline_ms
              << " ms  " << std::setprecision(1) << single_gmacs << " GMAC/s\n"
              << "N8 matrix          : " << std::setprecision(4) << n8_matrix_ms
              << " ms  W~" << std::setprecision(1) << n8_w_gbps << " GB/s\n"
              << "N8 pack+BMMA       : " << std::setprecision(4) << n8_pipeline_ms
              << " ms  " << std::setprecision(1) << n8_gmacs << " GMAC/s\n";

    CUDA_CHECK(cudaFree(dpos));
    CUDA_CHECK(cudaFree(dneg));
    CUDA_CHECK(cudaFree(dpp));
    CUDA_CHECK(cudaFree(dpn));
    CUDA_CHECK(cudaFree(dxb1));
    CUDA_CHECK(cudaFree(dbp8));
    CUDA_CHECK(cudaFree(dx8));
    CUDA_CHECK(cudaFree(dy1));
    CUDA_CHECK(cudaFree(dy8));

    Result r;
    r.name = s.name;
    r.M = M;
    r.K = K;
    r.copies = copies;
    r.weight_mib = weight_mib;
    r.single_matrix_ms = single_matrix_ms;
    r.single_pipeline_ms = single_pipeline_ms;
    r.n8_matrix_ms = n8_matrix_ms;
    r.n8_pipeline_ms = n8_pipeline_ms;
    return r;
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
            std::cerr << "V6 is intentionally restricted to RTX 3080 / SM86; found "
                      << prop.name << " sm_" << prop.major << prop.minor << "\n";
            return 3;
        }

        const std::array<Shape, 4> shapes = {{
            {"fused QKV",     3840, 2560},
            {"attention O",   2560, 2560},
            {"fused gate+up", 13824, 2560},
            {"MLP down",      2560, 6912},
        }};

        std::cout << "GA102-ROM V6: real microsoft/bitnet-b1.58-2B-4T projection shapes\n"
                  << "GPU               : " << prop.name << "\n"
                  << "decoder layers    : 30\n"
                  << "hidden/intermediate: 2560 / 6912\n"
                  << "execution         : fused QKV + O + fused gate/up + down\n"
                  << "weight-cache guard: rotate ~" << o.ring_mib
                  << " MiB per backend/shape\n"
                  << "input             : raw signed A8; dynamic bit packing included\n";

        std::vector<Result> results;
        for (int i = 0; i < (int)shapes.size(); ++i)
            results.push_back(benchmark_shape(shapes[i], o, i));

        double layer_single_ms = 0.0;
        double layer_n8_ms = 0.0;
        double layer_weight_mib = 0.0;
        double layer_macs = 0.0;
        for (const auto& r : results) {
            layer_single_ms += r.single_pipeline_ms;
            layer_n8_ms += r.n8_pipeline_ms;
            layer_weight_mib += r.weight_mib;
            layer_macs += (double)r.M * r.K;
        }

        const double decoder_single_ms = layer_single_ms * 30.0;
        const double decoder_n8_ms = layer_n8_ms * 30.0;
        const double decoder_macs = layer_macs * 30.0;
        const double single_linear_tps = 1000.0 / decoder_single_ms;
        const double n8_state_rate = 8000.0 / decoder_n8_ms;

        std::cout << "\n=== V6 aggregate: 30-layer ternary linears only ===\n"
                  << std::fixed << std::setprecision(3)
                  << "packed W / layer       : " << layer_weight_mib << " MiB\n"
                  << "packed W / 30 layers   : " << layer_weight_mib * 30.0 << " MiB\n"
                  << "logical MAC/token       : " << std::setprecision(0) << decoder_macs << "\n"
                  << std::setprecision(4)
                  << "single-token layer      : " << layer_single_ms << " ms\n"
                  << "single-token 30 layers  : " << decoder_single_ms << " ms\n"
                  << std::setprecision(1)
                  << "single linear-only rate : " << single_linear_tps << " token/s\n"
                  << std::setprecision(4)
                  << "N8 layer                : " << layer_n8_ms << " ms / 8 states\n"
                  << "N8 30 layers            : " << decoder_n8_ms << " ms / 8 states\n"
                  << std::setprecision(1)
                  << "N8 linear-only rate     : " << n8_state_rate << " states/s throughput\n";

        std::cout << "\nGuardrail:\n"
                  << "  This is NOT full-model token/s. It intentionally excludes attention/KV,\n"
                  << "  normalization, RoPE, ReLU2/gating, residuals, LM head, sampling and\n"
                  << "  cross-layer runtime overhead. The rotating weight ring is intended to\n"
                  << "  avoid reporting permanently-L2-hot small projection matrices.\n";

        std::cout << "\nV6 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
