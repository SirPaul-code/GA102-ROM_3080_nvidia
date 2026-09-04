#define main ga102_rom_v11_embedded_main
#include "bench_v11.cu"
#undef main

#include <limits>

// V16 validates the single-token POPC linear path against the exact decoder
// weight footprint: 30 unique layer matrices for every projection family.
// It also sweeps CUDA block size while keeping the one-warp/output-row mapping
// unchanged. This isolates scheduler/mapping effects from higher-level decoder
// kernels and from the cache-cold-ish ~64 MiB rings used by V11-V15.

__global__ void init_balanced_masks_v16(uint32_t* p, uint32_t* n,
                                        size_t count, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < count; i += stride) {
        uint32_t a = (uint32_t)i * 747796405u + seed;
        a = ((a >> ((a >> 28) + 4)) ^ a) * 277803737u;
        a = (a >> 22) ^ a;
        uint32_t b = a * 1664525u + 1013904223u + (seed ^ 0x9e3779b9u);
        b ^= b >> 16;
        b *= 2246822519u;
        b ^= b >> 13;
        p[i] = a & ~b;
        n[i] = b & ~a;
    }
}

__global__ void init_u32_v16(uint32_t* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        uint32_t z = (uint32_t)i * 747796405u + seed;
        z = ((z >> ((z >> 28) + 4)) ^ z) * 277803737u;
        z = (z >> 22) ^ z;
        p[i] = z;
    }
}

struct ExactRing16 {
    const char* name;
    int M = 0, K = 0, words = 0;
    int copies = LAYERS;
    size_t words_per_copy = 0;
    uint32_t* pos = nullptr;
    uint32_t* neg = nullptr;

    ExactRing16(const char* n, int m, int k) : name(n), M(m), K(k) {}

    void init(uint32_t seed) {
        words = K / 32;
        words_per_copy = (size_t)M * words;
        const size_t total_words = words_per_copy * copies;
        CUDA_CHECK(cudaMalloc(&pos, total_words * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&neg, total_words * sizeof(uint32_t)));
        const int blocks = (int)std::min<size_t>(65535, (total_words + 255) / 256);
        init_balanced_masks_v16<<<blocks, 256>>>(pos, neg, total_words, seed);
        CUDA_CHECK(cudaGetLastError());
    }

    const uint32_t* p(int layer) const {
        return pos + (size_t)layer * words_per_copy;
    }
    const uint32_t* n(int layer) const {
        return neg + (size_t)layer * words_per_copy;
    }
    size_t bytes_per_copy() const {
        return 2 * words_per_copy * sizeof(uint32_t);
    }
    size_t bytes_all() const {
        return bytes_per_copy() * copies;
    }
    double mib_all() const {
        return (double)bytes_all() / 1048576.0;
    }
    void release() {
        if (pos) cudaFree(pos);
        if (neg) cudaFree(neg);
        pos = neg = nullptr;
    }
    ~ExactRing16() { release(); }
};

static void launch_linear16(const ExactRing16& w, int layer,
                            const uint32_t* planes, int32_t* out,
                            int threads, cudaStream_t s) {
    const int blocks = (w.M * 32 + threads - 1) / threads;
    popc_single_warp<<<blocks, threads, 0, s>>>(
        w.p(layer), w.n(layer), planes, out, w.M, w.words);
}

static bool compare_i32_exact16(const int32_t* a, const int32_t* b, int n) {
    std::vector<int32_t> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    return ha == hb;
}

static cudaGraphExec_t capture_family16(const ExactRing16& w,
                                        const uint32_t* planes,
                                        int32_t* out,
                                        int threads,
                                        cudaStream_t s) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr;
    cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int layer = 0; layer < LAYERS; ++layer)
        launch_linear16(w, layer, planes, out, threads, s);
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static cudaGraphExec_t capture_all16(const ExactRing16& qkv,
                                     const ExactRing16& outw,
                                     const ExactRing16& gu,
                                     const ExactRing16& down,
                                     const uint32_t* planes_h,
                                     const uint32_t* planes_i,
                                     int32_t* qkv_out,
                                     int32_t* o_out,
                                     int32_t* gu_out,
                                     int32_t* down_out,
                                     int tq, int to, int tg, int td,
                                     cudaStream_t s) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr;
    cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int layer = 0; layer < LAYERS; ++layer) {
        launch_linear16(qkv, layer, planes_h, qkv_out, tq, s);
        launch_linear16(outw, layer, planes_h, o_out, to, s);
        launch_linear16(gu, layer, planes_h, gu_out, tg, s);
        launch_linear16(down, layer, planes_i, down_out, td, s);
    }
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static void warm_graph16(cudaGraphExec_t e, cudaStream_t s, int n = 5) {
    for (int i = 0; i < n; ++i) CUDA_CHECK(cudaGraphLaunch(e, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

static float time_graph_batch16(cudaGraphExec_t e, cudaStream_t s, int batch) {
    cudaEvent_t a = nullptr, b = nullptr;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < batch; ++i) CUDA_CHECK(cudaGraphLaunch(e, s));
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms / batch;
}

struct Stats16 {
    float med = 0.0f, min = 0.0f, max = 0.0f;
};

static Stats16 stats16(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front(), v.back()};
}

static Stats16 measure_graph16(cudaGraphExec_t e, cudaStream_t s,
                               int rounds, int batch) {
    warm_graph16(e, s, 5);
    std::vector<float> t;
    t.reserve(rounds);
    for (int r = 0; r < rounds; ++r)
        t.push_back(time_graph_batch16(e, s, batch));
    return stats16(t);
}

struct FamilyResult16 {
    int threads = 128;
    float ms = std::numeric_limits<float>::infinity();
    float spread = 0.0f;
    double gbps = 0.0;
    double gmacs = 0.0;
};

static FamilyResult16 benchmark_family16(const ExactRing16& w,
                                         const uint32_t* planes,
                                         int32_t* out,
                                         int32_t* ref,
                                         cudaStream_t s,
                                         int rounds,
                                         int batch) {
    const std::array<int, 5> thread_opts{{32, 64, 128, 256, 512}};

    launch_linear16(w, 0, planes, ref, 128, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    FamilyResult16 best;
    for (const int threads : thread_opts) {
        launch_linear16(w, 0, planes, out, threads, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        const bool exact = compare_i32_exact16(ref, out, w.M);
        if (!exact) {
            std::cout << "  threads " << std::setw(3) << threads
                      << "  correctness=FAIL\n";
            continue;
        }

        cudaGraphExec_t e = capture_family16(w, planes, out, threads, s);
        const Stats16 st = measure_graph16(e, s, rounds, batch);
        CUDA_CHECK(cudaGraphExecDestroy(e));

        const double gbps = (double)w.bytes_all() / (st.med * 1.0e6);
        const double macs = (double)w.M * w.K * LAYERS;
        const double gmacs = macs / (st.med * 1.0e6);
        const float spread = 100.0f * (st.max - st.min) / st.med;
        std::cout << std::fixed << std::setprecision(4)
                  << "  threads " << std::setw(3) << threads
                  << "  " << st.med << " ms/30L"
                  << "  " << std::setprecision(1) << gbps << " GB/s"
                  << "  " << gmacs << " GMAC/s"
                  << "  spread=" << spread << "%"
                  << "  PASS\n";

        if (st.med < best.ms)
            best = {threads, st.med, spread, gbps, gmacs};
    }
    return best;
}

struct Options16 {
    int rounds = 7;
    int batch = 200;
};

static Options16 parse16(int argc, char** argv) {
    Options16 o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rounds" && i + 1 < argc) o.rounds = std::stoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) o.batch = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V16 exact 30-layer ternary linear profiler\n"
                      << "  --rounds N  timing rounds/config (default 7)\n"
                      << "  --batch N   graph replays/round (default 200)\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown or incomplete argument: " + a);
    }
    if (o.rounds < 3 || o.batch < 1)
        throw std::runtime_error("rounds>=3 and batch>=1 required");
    return o;
}

int main(int argc, char** argv) {
    try {
        const Options16 o = parse16(argc, argv);
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 8 || prop.minor != 6 ||
            std::string(prop.name).find("RTX 3080") == std::string::npos) {
            std::cerr << "V16 restricted to RTX 3080 / SM86; found " << prop.name << "\n";
            return 3;
        }

        int memclk_khz = 0, bus_bits = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memclk_khz, cudaDevAttrMemoryClockRate, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
        const double peak_gbps = 2.0 * (double)memclk_khz * 1000.0 *
                                 ((double)bus_bits / 8.0) / 1.0e9;

        std::cout << "GA102-ROM V16: exact 30-layer ternary linear profiler\n"
                  << "GPU               : " << prop.name << "\n"
                  << "mapping           : 1 warp / output row, exact W1x2 POPC\n"
                  << "weight identity   : 30 unique matrices / projection family\n"
                  << "thread sweep      : 32 / 64 / 128 / 256 / 512\n"
                  << "property BW       : " << std::fixed << std::setprecision(1)
                  << peak_gbps << " GB/s\n"
                  << "timing            : CUDA Graph, " << o.rounds
                  << " rounds x " << o.batch << " replays\n";

        ExactRing16 qkv("QKV", QKV_DIM, HIDDEN);
        ExactRing16 outw("O", HIDDEN, HIDDEN);
        ExactRing16 gu("gate+up", GATEUP_DIM, HIDDEN);
        ExactRing16 down("down", HIDDEN, INTER);
        qkv.init(0x1234u);
        outw.init(0x2345u);
        gu.init(0x3456u);
        down.init(0x4567u);
        CUDA_CHECK(cudaDeviceSynchronize());

        const double total_mib = qkv.mib_all() + outw.mib_all() +
                                 gu.mib_all() + down.mib_all();
        std::cout << "\n=== V16 exact decoder linear footprint ===\n"
                  << "QKV       : " << std::setprecision(3) << qkv.mib_all() << " MiB\n"
                  << "O         : " << outw.mib_all() << " MiB\n"
                  << "gate+up   : " << gu.mib_all() << " MiB\n"
                  << "down      : " << down.mib_all() << " MiB\n"
                  << "TOTAL     : " << total_mib << " MiB"
                  << "  (expected 496.875 MiB)\n";

        uint32_t *planes_h = nullptr, *planes_i = nullptr;
        CUDA_CHECK(cudaMalloc(&planes_h, (size_t)8 * (HIDDEN / 32) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&planes_i, (size_t)8 * (INTER / 32) * sizeof(uint32_t)));
        init_u32_v16<<<32, 256>>>(planes_h, (size_t)8 * (HIDDEN / 32), 0x11112222u);
        init_u32_v16<<<32, 256>>>(planes_i, (size_t)8 * (INTER / 32), 0x33334444u);

        int32_t *qkv_out = nullptr, *o_out = nullptr, *gu_out = nullptr, *down_out = nullptr, *ref = nullptr;
        CUDA_CHECK(cudaMalloc(&qkv_out, QKV_DIM * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&o_out, HIDDEN * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&gu_out, GATEUP_DIM * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&down_out, HIDDEN * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&ref, GATEUP_DIM * sizeof(int32_t)));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaStream_t s = nullptr;
        CUDA_CHECK(cudaStreamCreate(&s));

        std::cout << "\n=== V16 projection-family sweep (30 unique layers) ===\n";
        std::cout << "\nQKV 3840x2560, packed=" << qkv.mib_all() << " MiB\n";
        const FamilyResult16 rq = benchmark_family16(qkv, planes_h, qkv_out, ref, s, o.rounds, o.batch);
        std::cout << "  BEST threads=" << rq.threads << "  " << rq.ms << " ms/30L\n";

        std::cout << "\nO 2560x2560, packed=" << outw.mib_all() << " MiB\n";
        const FamilyResult16 ro = benchmark_family16(outw, planes_h, o_out, ref, s, o.rounds, o.batch);
        std::cout << "  BEST threads=" << ro.threads << "  " << ro.ms << " ms/30L\n";

        std::cout << "\ngate+up 13824x2560, packed=" << gu.mib_all() << " MiB\n";
        const FamilyResult16 rg = benchmark_family16(gu, planes_h, gu_out, ref, s, o.rounds, o.batch);
        std::cout << "  BEST threads=" << rg.threads << "  " << rg.ms << " ms/30L\n";

        std::cout << "\ndown 2560x6912, packed=" << down.mib_all() << " MiB\n";
        const FamilyResult16 rd = benchmark_family16(down, planes_i, down_out, ref, s, o.rounds, o.batch);
        std::cout << "  BEST threads=" << rd.threads << "  " << rd.ms << " ms/30L\n";

        std::cout << "\n=== V16 complete 30-layer linear chain ===\n";
        cudaGraphExec_t e128 = capture_all16(qkv, outw, gu, down,
                                             planes_h, planes_i,
                                             qkv_out, o_out, gu_out, down_out,
                                             128, 128, 128, 128, s);
        cudaGraphExec_t etuned = capture_all16(qkv, outw, gu, down,
                                               planes_h, planes_i,
                                               qkv_out, o_out, gu_out, down_out,
                                               rq.threads, ro.threads, rg.threads, rd.threads, s);
        const Stats16 s128 = measure_graph16(e128, s, o.rounds, o.batch);
        const Stats16 stuned = measure_graph16(etuned, s, o.rounds, o.batch);
        CUDA_CHECK(cudaGraphExecDestroy(e128));
        CUDA_CHECK(cudaGraphExecDestroy(etuned));

        const size_t total_bytes = qkv.bytes_all() + outw.bytes_all() + gu.bytes_all() + down.bytes_all();
        const double total_macs = 2084044800.0;
        const double bw128 = (double)total_bytes / (s128.med * 1.0e6);
        const double bwt = (double)total_bytes / (stuned.med * 1.0e6);
        const double mac128 = total_macs / (s128.med * 1.0e6);
        const double mact = total_macs / (stuned.med * 1.0e6);
        const float spread128 = 100.0f * (s128.max - s128.min) / s128.med;
        const float spreadt = 100.0f * (stuned.max - stuned.min) / stuned.med;

        std::cout << std::fixed << std::setprecision(4)
                  << "current 128-thread mapping : " << s128.med << " ms/token"
                  << "  " << std::setprecision(1) << bw128 << " GB/s"
                  << "  " << mac128 << " GMAC/s"
                  << "  spread=" << spread128 << "%\n"
                  << std::setprecision(4)
                  << "family-tuned mapping       : " << stuned.med << " ms/token"
                  << "  " << std::setprecision(1) << bwt << " GB/s"
                  << "  " << mact << " GMAC/s"
                  << "  spread=" << spreadt << "%\n"
                  << "mapping speedup             : " << std::setprecision(3)
                  << (s128.med / stuned.med) << "x\n"
                  << "property-BW equivalent      : " << std::setprecision(1)
                  << (100.0 * bwt / peak_gbps) << "%\n";

        std::cout << "\nInterpretation guardrails:\n"
                  << "  * Every family owns exactly 30 unique packed matrices; total is 496.875 MiB.\n"
                  << "  * The sweep changes only CUDA block size. One warp still computes one output row.\n"
                  << "  * GB/s counts packed P+N weight bytes; activation/output traffic is not included.\n"
                  << "  * The complete-chain time contains only the 120 ternary projection launches, not norms, attention or LM head.\n"
                  << "  * If block-size tuning is small, the next linear optimization must change data layout/loading or warp mapping, not just scheduling.\n";

        CUDA_CHECK(cudaStreamDestroy(s));
        cudaFree(planes_h); cudaFree(planes_i);
        cudaFree(qkv_out); cudaFree(o_out); cudaFree(gu_out); cudaFree(down_out); cudaFree(ref);
        std::cout << "\nV16 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
