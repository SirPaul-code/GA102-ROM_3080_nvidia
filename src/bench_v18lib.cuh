#pragma once

// Reusable V18 split-K helpers. Include after bench_v17.cu has supplied
// Family17, Stats17, bmma17, launch_popc17, equal_i32_17 and timing helpers.

template<int WARPS>
__global__ void bmma_splitk18(const uint4* posp, const uint4* negp,
                              const uint32_t* xb, int32_t* y,
                              int tiles, int chunks, int words) {
    const int tile = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (tile >= tiles || warp >= WARPS) return;

    const int group = lane >> 2;
    const int tid4 = lane & 3;
    const int col0 = tid4 * 2;
    const int col1 = col0 + 1;
    const int sc0 = (col0 == 7) ? -128 : (1 << col0);
    const int sc1 = (col1 == 7) ? -128 : (1 << col1);

    int part0 = 0;
    int part8 = 0;
    for (int chunk = warp; chunk < chunks; chunk += WARPS) {
        const size_t fi = ((size_t)tile * chunks + chunk) * 32 + lane;
        const uint4 p = posp[fi];
        const uint4 n = negp[fi];
        const uint32_t* plane = xb + (size_t)group * words;
        const int wbase = chunk * 8;
        const uint32_t b0 = plane[wbase + tid4];
        const uint32_t b1 = plane[wbase + 4 + tid4];

        int pc0 = 0, pc1 = 0, pc2 = 0, pc3 = 0;
        int nc0 = 0, nc1 = 0, nc2 = 0, nc3 = 0;
        bmma17(p.x, p.y, p.z, p.w, b0, b1, pc0, pc1, pc2, pc3);
        bmma17(n.x, n.y, n.z, n.w, b0, b1, nc0, nc1, nc2, nc3);
        part0 += (pc0 - nc0) * sc0 + (pc1 - nc1) * sc1;
        part8 += (pc2 - nc2) * sc0 + (pc3 - nc3) * sc1;
    }

    part0 += __shfl_down_sync(0xffffffffu, part0, 2, 4);
    part8 += __shfl_down_sync(0xffffffffu, part8, 2, 4);
    part0 += __shfl_down_sync(0xffffffffu, part0, 1, 4);
    part8 += __shfl_down_sync(0xffffffffu, part8, 1, 4);

    __shared__ int partial[WARPS][16];
    if (tid4 == 0) {
        partial[warp][group] = part0;
        partial[warp][group + 8] = part8;
    }
    __syncthreads();

    if (threadIdx.x < 16) {
        int sum = 0;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) sum += partial[w][threadIdx.x];
        y[tile * 16 + threadIdx.x] = sum;
    }
}

static void launch_split18(const Family17& f, int layer,
                           const uint32_t* planes, int32_t* out,
                           int warps, cudaStream_t s) {
    switch (warps) {
        case 1: bmma_splitk18<1><<<f.tiles, 32, 0, s>>>(f.bp(layer), f.bn(layer), planes, out, f.tiles, f.chunks, f.words); break;
        case 2: bmma_splitk18<2><<<f.tiles, 64, 0, s>>>(f.bp(layer), f.bn(layer), planes, out, f.tiles, f.chunks, f.words); break;
        case 4: bmma_splitk18<4><<<f.tiles, 128, 0, s>>>(f.bp(layer), f.bn(layer), planes, out, f.tiles, f.chunks, f.words); break;
        case 8: bmma_splitk18<8><<<f.tiles, 256, 0, s>>>(f.bp(layer), f.bn(layer), planes, out, f.tiles, f.chunks, f.words); break;
        default: throw std::runtime_error("V18 warps/tile must be 1,2,4,8");
    }
}

static cudaGraphExec_t capture_split_family18(const Family17& f,
                                               const uint32_t* planes,
                                               int32_t* out, int warps,
                                               cudaStream_t s) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr;
    cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int l = 0; l < L17; ++l) launch_split18(f, l, planes, out, warps, s);
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static cudaGraphExec_t capture_split_all18(const Family17& qkv,
                                            const Family17& o,
                                            const Family17& gu,
                                            const Family17& down,
                                            const uint32_t* ph,
                                            const uint32_t* pi,
                                            int32_t* oq, int32_t* oo,
                                            int32_t* og, int32_t* od,
                                            const std::array<int,4>& warps,
                                            cudaStream_t s) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr;
    cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int l = 0; l < L17; ++l) {
        launch_split18(qkv, l, ph, oq, warps[0], s);
        launch_split18(o, l, ph, oo, warps[1], s);
        launch_split18(gu, l, ph, og, warps[2], s);
        launch_split18(down, l, pi, od, warps[3], s);
    }
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static bool check_split18(const Family17& f, const uint32_t* planes,
                          int32_t* ref, int32_t* got, int warps,
                          cudaStream_t s) {
    for (int l = 0; l < L17; ++l) {
        launch_popc17(f, l, planes, ref, 128, s);
        launch_split18(f, l, planes, got, warps, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        int bad = -1;
        if (!equal_i32_17(ref, got, f.M, &bad)) {
            std::cerr << f.name << " split-K correctness FAIL warps=" << warps
                      << " layer=" << l << " row=" << bad << "\n";
            return false;
        }
    }
    return true;
}

struct Choice18 { int warps = 1; Stats17 st; };

static Choice18 sweep_split18(const Family17& f, const uint32_t* planes,
                              int32_t* out, cudaStream_t s,
                              int rounds, int batch) {
    const std::array<int,4> opts{{1,2,4,8}};
    Choice18 best;
    best.st.med = std::numeric_limits<float>::infinity();
    for (int w : opts) {
        auto e = capture_split_family18(f, planes, out, w, s);
        const Stats17 st = measure17(e, s, rounds, batch);
        CUDA_CHECK(cudaGraphExecDestroy(e));
        const double gbps = (double)f.bytes_all() / (st.med * 1e6);
        const double gmac = (double)f.M * f.K * L17 / (st.med * 1e6);
        const float spread = 100.0f * (st.max - st.min) / st.med;
        std::cout << std::fixed << std::setprecision(4)
                  << "  split-K warps/tile " << w << "  " << st.med << " ms/30L"
                  << "  " << std::setprecision(1) << gbps << " GB/s"
                  << "  " << gmac << " GMAC/s  spread=" << spread << "%\n";
        if (st.med < best.st.med) best = {w, st};
    }
    return best;
}
