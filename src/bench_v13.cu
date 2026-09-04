#define main ga102_rom_v12_embedded_main
#include "bench_v12.cu"
#undef main

#include <cstring>

// V13 hardens the V12 experiment before we accept any performance conclusion.
// It addresses two problems observed in the first V12 run:
//   1) the synthetic V11 mask generator had a +1 bias (P~50%, N~25%), which can
//      make the 30-layer synthetic state numerically pathological;
//   2) one-shot A-then-B timings are too sensitive to boost/thermal/order state.
//
// V13 therefore:
//   * overwrites every packed weight ring with balanced, disjoint +/- masks;
//   * uses independent Runtime objects for baseline and fused paths;
//   * validates the three quant->bitplane fusion primitives exactly;
//   * alternates A/B and B/A timing batches and reports medians/spread;
//   * compares the FULL 128256-logit output and rejects non-finite results.

__global__ void init_balanced_masks(uint32_t* p, uint32_t* n,
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

        // Equal expected +1/-1 density (25% each), 50% zero, and never P&N.
        // Density does not change POPC work; zero mean keeps synthetic state sane.
        p[i] = a & ~b;
        n[i] = b & ~a;
    }
}

__global__ void init_i32_v13(int32_t* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        uint32_t z = (uint32_t)i * 747796405u + seed;
        z = ((z >> ((z >> 28) + 4)) ^ z) * 277803737u;
        z = (z >> 22) ^ z;
        p[i] = (int32_t)(z & 0x1ffffu) - 0xffff;
    }
}

static void rebalance_ring(MaskRing& w, uint32_t seed) {
    const size_t count = w.words_per_copy * (size_t)w.copies;
    const int blocks = (int)std::min<size_t>(65535, (count + 255) / 256);
    init_balanced_masks<<<blocks, 256>>>(w.pos, w.neg, count, seed);
    CUDA_CHECK(cudaGetLastError());
}

static void rebalance_runtime(Runtime& r) {
    rebalance_ring(r.qkv, 0x1234u);
    rebalance_ring(r.out, 0x2345u);
    rebalance_ring(r.gu, 0x3456u);
    rebalance_ring(r.down, 0x4567u);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static bool equal_u32_device(const uint32_t* a, const uint32_t* b,
                             size_t n, const char* label) {
    std::vector<uint32_t> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        if (ha[i] != hb[i]) {
            std::cerr << label << " plane mismatch at word " << i
                      << ": baseline=0x" << std::hex << ha[i]
                      << " fused=0x" << hb[i] << std::dec << "\n";
            return false;
        }
    }
    return true;
}

static bool equal_bf16_device(const bf16* a, const bf16* b,
                              size_t n, const char* label) {
    std::vector<bf16> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(bf16), cudaMemcpyDeviceToHost));
    if (std::memcmp(ha.data(), hb.data(), n * sizeof(bf16)) != 0) {
        std::cerr << label << " BF16 intermediate mismatch\n";
        return false;
    }
    return true;
}

struct PrimitiveCheck {
    bool input = false;
    bool o = false;
    bool gateup = false;
};

static PrimitiveCheck check_fusion_primitives(Runtime& base, Runtime& fused,
                                              cudaStream_t sb, cudaStream_t sf) {
    constexpr int t = 256;
    PrimitiveCheck c;

    // 1) input RMSNorm + A8 quant + standalone pack vs direct bitplanes.
    rmsnorm_quant_a8<<<1, t, 0, sb>>>(base.embed, base.gamma_h,
                                      base.qh, base.scale_h, HIDDEN, 1e-6f);
    base.pack_hidden(sb);
    rmsnorm_quant_planes<<<1, t, 0, sf>>>(fused.embed, fused.gamma_h,
                                          fused.planes_h, fused.scale_h,
                                          HIDDEN, 1e-6f);
    CUDA_CHECK(cudaStreamSynchronize(sb));
    CUDA_CHECK(cudaStreamSynchronize(sf));
    c.input = equal_u32_device(base.planes_h, fused.planes_h,
                               (size_t)8 * (HIDDEN / 32), "input RMSNorm");

    // 2) O epilogue + residual + norm + quant + pack.
    init_i32_v13<<<64, 256, 0, sb>>>(base.o_acc, HIDDEN, 0xa11ce001u);
    init_i32_v13<<<64, 256, 0, sf>>>(fused.o_acc, HIDDEN, 0xa11ce001u);
    o_resid_norm_quant<<<1, t, 0, sb>>>(base.o_acc, base.embed, base.gamma_h,
                                        base.mid, base.qh, base.scale_h,
                                        0.0001f, 1e-6f);
    base.pack_hidden(sb);
    o_resid_norm_planes<<<1, t, 0, sf>>>(fused.o_acc, fused.embed, fused.gamma_h,
                                         fused.mid, fused.planes_h, fused.scale_h,
                                         0.0001f, 1e-6f);
    CUDA_CHECK(cudaStreamSynchronize(sb));
    CUDA_CHECK(cudaStreamSynchronize(sf));
    c.o = equal_bf16_device(base.mid, fused.mid, HIDDEN, "O residual") &&
          equal_u32_device(base.planes_h, fused.planes_h,
                           (size_t)8 * (HIDDEN / 32), "O residual");

    // 3) gate/up + ReLU2 + norm + quant + pack.
    init_i32_v13<<<128, 256, 0, sb>>>(base.gu_acc, GATEUP_DIM, 0xa11ce002u);
    init_i32_v13<<<128, 256, 0, sf>>>(fused.gu_acc, GATEUP_DIM, 0xa11ce002u);
    gateup_relu2_norm_quant<<<1, t, 0, sb>>>(base.gu_acc, base.gamma_i,
                                             base.tmpi, base.qi, base.scale_i,
                                             0.00005f, 0.00005f, 1e-6f);
    base.pack_inter(sb);
    gateup_relu2_norm_planes<<<1, t, 0, sf>>>(fused.gu_acc, fused.gamma_i,
                                               fused.tmpi, fused.planes_i,
                                               fused.scale_i,
                                               0.00005f, 0.00005f, 1e-6f);
    CUDA_CHECK(cudaStreamSynchronize(sb));
    CUDA_CHECK(cudaStreamSynchronize(sf));
    c.gateup = equal_bf16_device(base.tmpi, fused.tmpi, INTER, "gate/up") &&
               equal_u32_device(base.planes_i, fused.planes_i,
                                (size_t)8 * (INTER / 32), "gate/up");
    return c;
}

static cudaGraphExec_t capture_v11(Runtime& r, cudaStream_t s, int L) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr;
    cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    r.launch(s, L);
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static cudaGraphExec_t capture_v12(Runtime& r, cudaStream_t s, int L) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr;
    cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    launch_v12(r, s, L);
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static void launch_n(cudaGraphExec_t e, cudaStream_t s, int n) {
    for (int i = 0; i < n; ++i) CUDA_CHECK(cudaGraphLaunch(e, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

static float time_exec_batch(cudaGraphExec_t e, cudaStream_t s, int n) {
    cudaEvent_t a = nullptr, b = nullptr;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < n; ++i) CUDA_CHECK(cudaGraphLaunch(e, s));
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return total / n;
}

struct Stats {
    float median = 0.0f;
    float min = 0.0f;
    float max = 0.0f;
};

static Stats summarize(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    Stats s;
    s.min = v.front();
    s.max = v.back();
    s.median = v[v.size() / 2];
    return s;
}

struct LogitCheck {
    float max_abs_diff = 0.0f;
    size_t nonfinite_base = 0;
    size_t nonfinite_fused = 0;
};

static LogitCheck compare_full_logits(Runtime& base, Runtime& fused) {
    std::vector<float> a(VOCAB), b(VOCAB);
    CUDA_CHECK(cudaMemcpy(a.data(), base.logits, a.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), fused.logits, b.size() * sizeof(float), cudaMemcpyDeviceToHost));
    LogitCheck c;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) ++c.nonfinite_base;
        if (!std::isfinite(b[i])) ++c.nonfinite_fused;
        if (std::isfinite(a[i]) && std::isfinite(b[i]))
            c.max_abs_diff = std::max(c.max_abs_diff, std::fabs(a[i] - b[i]));
    }
    return c;
}

static Options parse_v13_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "--ring-mib" && i + 1 < argc) o.ring_mib = std::stoi(argv[++i]);
        else if (a == "--max-context" && i + 1 < argc) o.max_context = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V13 hardened V11-vs-V12 fusion benchmark\n"
                      << "  --iters N        target timed graph launches/path/context (default 50)\n"
                      << "  --ring-mib N     packed-weight ring per projection (default 64 MiB)\n"
                      << "  --max-context N  128..4096 (default 4096)\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown or incomplete argument: " + a);
    }
    if (o.iters <= 0 || o.ring_mib < 16 || o.max_context < 128 || o.max_context > 4096)
        throw std::runtime_error("iters>0, ring-mib>=16, max-context=128..4096 required");
    return o;
}

int main(int argc, char** argv) {
    try {
        const Options o = parse_v13_args(argc, argv);
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 8 || prop.minor != 6 ||
            std::string(prop.name).find("RTX 3080") == std::string::npos) {
            std::cerr << "V13 restricted to RTX 3080 / SM86; found " << prop.name << "\n";
            return 3;
        }

        std::cout << "GA102-ROM V13: hardened quant->bitplane fusion A/B\n"
                  << "GPU               : " << prop.name << "\n"
                  << "baseline          : V11 graph schedule\n"
                  << "candidate         : V12 fused-bitplane graph schedule\n"
                  << "weights           : balanced disjoint +/- synthetic masks (zero mean)\n"
                  << "isolation         : separate Runtime objects, identical deterministic data\n"
                  << "timing            : alternating A/B batches, median of 7 rounds\n"
                  << "correctness       : primitive bitplanes + all 128256 final logits\n";

        Runtime base(o);
        Runtime fused(o);
        rebalance_runtime(base);
        rebalance_runtime(fused);

        cudaStream_t sb = nullptr, sf = nullptr;
        CUDA_CHECK(cudaStreamCreate(&sb));
        CUDA_CHECK(cudaStreamCreate(&sf));

        const PrimitiveCheck pc = check_fusion_primitives(base, fused, sb, sf);
        std::cout << "\n=== V13 primitive fusion exactness ===\n"
                  << "input RMSNorm->planes : " << (pc.input ? "PASS" : "FAIL") << "\n"
                  << "O resid/norm->planes  : " << (pc.o ? "PASS" : "FAIL") << "\n"
                  << "gate/up norm->planes  : " << (pc.gateup ? "PASS" : "FAIL") << "\n";
        if (!(pc.input && pc.o && pc.gateup)) {
            std::cerr << "V13 stopped: fused primitive is not bit-exact.\n";
            return 4;
        }

        constexpr int rounds = 7;
        const int batch = std::max(1, (o.iters + rounds - 1) / rounds);
        const std::array<int, 5> ctxs{{128, 512, 1024, 2048, 4096}};

        std::cout << "\n=== V13 stabilized full-decoder graph A/B ===\n";
        for (const int L : ctxs) {
            if (L > o.max_context) continue;

            cudaGraphExec_t eb = capture_v11(base, sb, L);
            cudaGraphExec_t ef = capture_v12(fused, sf, L);

            // Equal warm-up counts on independent state before any timed round.
            launch_n(eb, sb, 5);
            launch_n(ef, sf, 5);

            std::vector<float> tb, tf;
            tb.reserve(rounds);
            tf.reserve(rounds);
            for (int r = 0; r < rounds; ++r) {
                if ((r & 1) == 0) {
                    tb.push_back(time_exec_batch(eb, sb, batch));
                    tf.push_back(time_exec_batch(ef, sf, batch));
                } else {
                    tf.push_back(time_exec_batch(ef, sf, batch));
                    tb.push_back(time_exec_batch(eb, sb, batch));
                }
            }

            // Equal final launch count; compare the entire logits vector.
            launch_n(eb, sb, 1);
            launch_n(ef, sf, 1);
            const LogitCheck lc = compare_full_logits(base, fused);

            const Stats bs = summarize(tb);
            const Stats fs = summarize(tf);
            const float bspread = 100.0f * (bs.max - bs.min) / bs.median;
            const float fspread = 100.0f * (fs.max - fs.min) / fs.median;
            const bool exact = lc.nonfinite_base == 0 && lc.nonfinite_fused == 0 &&
                               lc.max_abs_diff == 0.0f;

            std::cout << std::fixed << std::setprecision(4)
                      << "ctx " << std::setw(4) << L
                      << "  V11-med=" << bs.median << " ms"
                      << "  V12-med=" << fs.median << " ms"
                      << "  speedup=" << std::setprecision(3) << (bs.median / fs.median) << "x"
                      << "  V12-rate=" << std::setprecision(1) << (1000.0f / fs.median) << " tok/s\n"
                      << "          spread V11=" << std::setprecision(1) << bspread << "%"
                      << " V12=" << fspread << "%"
                      << "  full-logit-diff=" << std::setprecision(6) << lc.max_abs_diff
                      << "  nonfinite=" << lc.nonfinite_base << "/" << lc.nonfinite_fused
                      << "  " << (exact ? "PASS" : "FAIL") << "\n";

            CUDA_CHECK(cudaGraphExecDestroy(eb));
            CUDA_CHECK(cudaGraphExecDestroy(ef));
        }

        std::cout << "\nInterpretation guardrails:\n"
                  << "  * V12's first run is treated as inconclusive because ctx2048 failed output equality and timing order was noisy.\n"
                  << "  * V13 changes only synthetic weight statistics and benchmark isolation/methodology; matrix/attention algorithms are unchanged.\n"
                  << "  * Balanced masks are zero-mean and disjoint; POPC instruction count and packed bytes are unchanged.\n"
                  << "  * Accept fusion performance only on rows with PASS and reasonable A/B spread.\n"
                  << "  * This is still a synthetic GPU-runtime ceiling, not checkpoint-validated text generation.\n";

        CUDA_CHECK(cudaStreamDestroy(sb));
        CUDA_CHECK(cudaStreamDestroy(sf));
        std::cout << "\nV13 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
