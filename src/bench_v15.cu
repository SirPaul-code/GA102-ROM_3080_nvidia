#define main ga102_rom_v11_embedded_main
#include "bench_v11.cu"
#undef main

#include <cstring>

// V15 probes the cleanest matrix->epilogue fusion in the decoder:
// MLP down POPC + second residual add. Unlike O/gate-up epilogues this stage
// has no cross-row reduction, so the fused kernel can preserve the exact
// arithmetic while removing the int32 down_acc write/read and one kernel node.

__global__ void init_balanced_masks_v15(uint32_t* p, uint32_t* n,
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

static void rebalance_ring_v15(MaskRing& w, uint32_t seed) {
    const size_t count = w.words_per_copy * (size_t)w.copies;
    const int blocks = (int)std::min<size_t>(65535, (count + 255) / 256);
    init_balanced_masks_v15<<<blocks, 256>>>(w.pos, w.neg, count, seed);
    CUDA_CHECK(cudaGetLastError());
}

static void rebalance_runtime_v15(Runtime& r) {
    rebalance_ring_v15(r.qkv, 0x1234u);
    rebalance_ring_v15(r.out, 0x2345u);
    rebalance_ring_v15(r.gu, 0x3456u);
    rebalance_ring_v15(r.down, 0x4567u);
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void popc_down_resid_fused(const uint32_t* pos,
                                      const uint32_t* neg,
                                      const uint32_t* xb,
                                      const bf16* residual,
                                      bf16* out,
                                      int M,
                                      int words,
                                      float sc) {
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
    if (lane == 0)
        out[row] = b16(bf(residual[row]) + (float)acc * sc);
}

static void launch_down_fused(Runtime& r, cudaStream_t s, int layer,
                              const bf16* residual, bf16* out) {
    constexpr int threads = 128;
    const int blocks = (HIDDEN * 32 + threads - 1) / threads;
    popc_down_resid_fused<<<blocks, threads, 0, s>>>(
        r.down.p(layer), r.down.n(layer), r.planes_i,
        residual, out, HIDDEN, r.down.words, 0.001f);
}

static void launch_v15(Runtime& r, cudaStream_t s, int L) {
    constexpr int t = 256;
    copy_bf16<<<(HIDDEN + t - 1) / t, t, 0, s>>>(r.embed, r.hidden, HIDDEN);

    for (int layer = 0; layer < LAYERS; ++layer) {
        rmsnorm_quant_a8<<<1, t, 0, s>>>(
            r.hidden, r.gamma_h, r.qh, r.scale_h, HIDDEN, 1e-6f);
        r.pack_hidden(s);
        r.linear(s, r.qkv, layer, r.planes_h, r.qkv_acc);

        bf16* kl = r.kcache + (size_t)layer * r.layer_kv_stride;
        bf16* vl = r.vcache + (size_t)layer * r.layer_kv_stride;
        qkv_epilogue_rope_kv<<<(QKV_DIM + t - 1) / t, t, 0, s>>>(
            r.qkv_acc, r.q, kl, vl, r.cs, r.sn, L - 1,
            0.001f, 0.001f, 0.001f);

        r.attention(s, layer, L);

        rmsnorm_quant_a8<<<1, t, 0, s>>>(
            r.attn, r.gamma_h, r.qh, r.scale_h, HIDDEN, 1e-6f);
        r.pack_hidden(s);
        r.linear(s, r.out, layer, r.planes_h, r.o_acc);

        o_resid_norm_quant<<<1, t, 0, s>>>(
            r.o_acc, r.hidden, r.gamma_h, r.mid,
            r.qh, r.scale_h, 0.001f, 1e-6f);
        r.pack_hidden(s);
        r.linear(s, r.gu, layer, r.planes_h, r.gu_acc);

        gateup_relu2_norm_quant<<<1, t, 0, s>>>(
            r.gu_acc, r.gamma_i, r.tmpi,
            r.qi, r.scale_i, 0.0005f, 0.0005f, 1e-6f);
        r.pack_inter(s);

        // V15 change: POPC down and residual are one kernel. No down_acc.
        launch_down_fused(r, s, layer, r.mid, r.hidden);
    }

    rmsnorm_bf16<<<1, t, 0, s>>>(
        r.hidden, r.gamma_h, r.finalnorm, HIDDEN, 1e-6f);
    constexpr int lm_threads = 128;
    const int lm_blocks = (VOCAB * 32 + lm_threads - 1) / lm_threads;
    lmhead_bf16_warp<<<lm_blocks, lm_threads, 0, s>>>(
        r.lmw, r.finalnorm, r.logits, VOCAB, HIDDEN);
}

static cudaGraphExec_t capture_graph_v15(Runtime& r, cudaStream_t s,
                                         int L, bool fused) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr;
    cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    if (fused) launch_v15(r, s, L); else r.launch(s, L);
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static void launch_n_v15(cudaGraphExec_t e, cudaStream_t s, int n) {
    for (int i = 0; i < n; ++i) CUDA_CHECK(cudaGraphLaunch(e, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

static float time_batch_v15(cudaGraphExec_t e, cudaStream_t s, int n) {
    cudaEvent_t a = nullptr, b = nullptr;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < n; ++i) CUDA_CHECK(cudaGraphLaunch(e, s));
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms / n;
}

struct Stats15 {
    float med = 0.0f, min = 0.0f, max = 0.0f;
};

static Stats15 stats15(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front(), v.back()};
}

struct LogitDiff15 {
    float max_abs = 0.0f;
    size_t nonfinite_a = 0, nonfinite_b = 0;
};

static LogitDiff15 compare_logits_v15(Runtime& a, Runtime& b) {
    std::vector<float> ha(VOCAB), hb(VOCAB);
    CUDA_CHECK(cudaMemcpy(ha.data(), a.logits, VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b.logits, VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
    LogitDiff15 d;
    for (int i = 0; i < VOCAB; ++i) {
        if (!std::isfinite(ha[i])) ++d.nonfinite_a;
        if (!std::isfinite(hb[i])) ++d.nonfinite_b;
        if (std::isfinite(ha[i]) && std::isfinite(hb[i]))
            d.max_abs = std::max(d.max_abs, std::fabs(ha[i] - hb[i]));
    }
    return d;
}

static bool compare_bf16_exact_v15(const bf16* a, const bf16* b, int n,
                                   float* max_abs) {
    std::vector<bf16> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(bf16), cudaMemcpyDeviceToHost));
    *max_abs = 0.0f;
    bool same = true;
    for (int i = 0; i < n; ++i) {
        uint16_t ua = 0, ub = 0;
        std::memcpy(&ua, &ha[i], sizeof(uint16_t));
        std::memcpy(&ub, &hb[i], sizeof(uint16_t));
        if (ua != ub) same = false;
        uint32_t fa_bits = (uint32_t)ua << 16;
        uint32_t fb_bits = (uint32_t)ub << 16;
        float fa = 0.0f, fb = 0.0f;
        std::memcpy(&fa, &fa_bits, sizeof(float));
        std::memcpy(&fb, &fb_bits, sizeof(float));
        *max_abs = std::max(*max_abs, std::fabs(fa - fb));
    }
    return same;
}

static cudaGraphExec_t capture_down_chain(Runtime& r, cudaStream_t s, bool fused) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr;
    cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int layer = 0; layer < LAYERS; ++layer) {
        if (fused) {
            launch_down_fused(r, s, layer, r.mid, r.hidden);
        } else {
            r.linear(s, r.down, layer, r.planes_i, r.down_acc);
            down_resid<<<(HIDDEN + 255) / 256, 256, 0, s>>>(
                r.down_acc, r.mid, r.hidden, 0.001f);
        }
    }
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static Options parse_v15(int argc, char** argv) {
    Options o;
    o.iters = 140;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "--ring-mib" && i + 1 < argc) o.ring_mib = std::stoi(argv[++i]);
        else if (a == "--max-context" && i + 1 < argc) o.max_context = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V15 exact down POPC+residual fusion\n"
                      << "  --iters N        total target launches/path/context (default 140)\n"
                      << "  --ring-mib N     packed-weight ring/projection (default 64)\n"
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
        const Options o = parse_v15(argc, argv);
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 8 || prop.minor != 6 ||
            std::string(prop.name).find("RTX 3080") == std::string::npos) {
            std::cerr << "V15 restricted to RTX 3080 / SM86; found " << prop.name << "\n";
            return 3;
        }

        std::cout << "GA102-ROM V15: exact down projection + residual fusion\n"
                  << "GPU               : " << prop.name << "\n"
                  << "baseline          : V11 Graph schedule\n"
                  << "candidate         : fuse POPC down + residual add\n"
                  << "removed/token     : 30 down_resid kernel nodes + ~600 KiB int32 write/read\n"
                  << "weights           : balanced disjoint +/- synthetic masks\n"
                  << "correctness       : bit-exact BF16 down output + all final logits\n";

        Runtime base(o), fused(o);
        rebalance_runtime_v15(base);
        rebalance_runtime_v15(fused);
        cudaStream_t sb = nullptr, sf = nullptr;
        CUDA_CHECK(cudaStreamCreate(&sb));
        CUDA_CHECK(cudaStreamCreate(&sf));

        // Primitive exactness with identical residual, bitplanes and layer weights.
        init_bf16<<<32, 256, 0, sb>>>(base.mid, HIDDEN, 0x51515151u);
        init_bf16<<<32, 256, 0, sf>>>(fused.mid, HIDDEN, 0x51515151u);
        CUDA_CHECK(cudaMemsetAsync(base.planes_i, 0xa5,
            (size_t)8 * (INTER / 32) * sizeof(uint32_t), sb));
        CUDA_CHECK(cudaMemsetAsync(fused.planes_i, 0xa5,
            (size_t)8 * (INTER / 32) * sizeof(uint32_t), sf));
        base.linear(sb, base.down, 7, base.planes_i, base.down_acc);
        down_resid<<<(HIDDEN + 255) / 256, 256, 0, sb>>>(
            base.down_acc, base.mid, base.hidden, 0.001f);
        launch_down_fused(fused, sf, 7, fused.mid, fused.hidden);
        CUDA_CHECK(cudaStreamSynchronize(sb));
        CUDA_CHECK(cudaStreamSynchronize(sf));
        float primitive_err = 0.0f;
        const bool primitive_exact = compare_bf16_exact_v15(
            base.hidden, fused.hidden, HIDDEN, &primitive_err);

        std::cout << "\n=== V15 primitive exactness ===\n"
                  << "down POPC+residual : " << (primitive_exact ? "PASS" : "FAIL")
                  << "  max_abs=" << std::fixed << std::setprecision(6) << primitive_err << "\n";
        if (!primitive_exact) {
            std::cerr << "V15 stopped: local fused epilogue is not bit-exact.\n";
            return 4;
        }

        // 30-layer down-only ring probe: same 67.5 MiB rotating weight family.
        cudaGraphExec_t db = capture_down_chain(base, sb, false);
        cudaGraphExec_t df = capture_down_chain(fused, sf, true);
        launch_n_v15(db, sb, 5);
        launch_n_v15(df, sf, 5);
        std::vector<float> tdb, tdf;
        for (int r = 0; r < 7; ++r) {
            if ((r & 1) == 0) {
                tdb.push_back(time_batch_v15(db, sb, 40));
                tdf.push_back(time_batch_v15(df, sf, 40));
            } else {
                tdf.push_back(time_batch_v15(df, sf, 40));
                tdb.push_back(time_batch_v15(db, sb, 40));
            }
        }
        const Stats15 dsb = stats15(tdb), dsf = stats15(tdf);
        std::cout << "\n=== V15 30-layer down-chain Graph probe ===\n"
                  << "baseline  : " << std::setprecision(4) << dsb.med << " ms\n"
                  << "fused     : " << dsf.med << " ms\n"
                  << "speedup   : " << std::setprecision(3) << (dsb.med / dsf.med) << "x\n"
                  << "spread    : base=" << std::setprecision(1)
                  << 100.0f * (dsb.max - dsb.min) / dsb.med << "% fused="
                  << 100.0f * (dsf.max - dsf.min) / dsf.med << "%\n";
        CUDA_CHECK(cudaGraphExecDestroy(db));
        CUDA_CHECK(cudaGraphExecDestroy(df));

        constexpr int rounds = 7;
        const int batch = std::max(1, (o.iters + rounds - 1) / rounds);
        const std::array<int, 5> ctxs{{128, 512, 1024, 2048, 4096}};
        std::cout << "\n=== V15 stabilized full-decoder Graph A/B ===\n";
        for (const int L : ctxs) {
            if (L > o.max_context) continue;
            cudaGraphExec_t eb = capture_graph_v15(base, sb, L, false);
            cudaGraphExec_t ef = capture_graph_v15(fused, sf, L, true);
            launch_n_v15(eb, sb, 5);
            launch_n_v15(ef, sf, 5);

            std::vector<float> tb, tf;
            for (int r = 0; r < rounds; ++r) {
                if ((r & 1) == 0) {
                    tb.push_back(time_batch_v15(eb, sb, batch));
                    tf.push_back(time_batch_v15(ef, sf, batch));
                } else {
                    tf.push_back(time_batch_v15(ef, sf, batch));
                    tb.push_back(time_batch_v15(eb, sb, batch));
                }
            }

            launch_n_v15(eb, sb, 1);
            launch_n_v15(ef, sf, 1);
            const LogitDiff15 ld = compare_logits_v15(base, fused);
            const Stats15 bs = stats15(tb), fs = stats15(tf);
            const float bspread = 100.0f * (bs.max - bs.min) / bs.med;
            const float fspread = 100.0f * (fs.max - fs.min) / fs.med;
            const bool pass = ld.nonfinite_a == 0 && ld.nonfinite_b == 0 && ld.max_abs == 0.0f;

            std::cout << std::fixed << std::setprecision(4)
                      << "ctx " << std::setw(4) << L
                      << "  V11-med=" << bs.med << " ms"
                      << "  V15-med=" << fs.med << " ms"
                      << "  speedup=" << std::setprecision(3) << (bs.med / fs.med) << "x"
                      << "  V15-rate=" << std::setprecision(1) << (1000.0f / fs.med) << " tok/s\n"
                      << "          spread V11=" << bspread << "% V15=" << fspread << "%"
                      << "  full-logit-diff=" << std::setprecision(6) << ld.max_abs
                      << "  nonfinite=" << ld.nonfinite_a << "/" << ld.nonfinite_b
                      << "  " << (pass ? "PASS" : "FAIL") << "\n";

            CUDA_CHECK(cudaGraphExecDestroy(eb));
            CUDA_CHECK(cudaGraphExecDestroy(ef));
        }

        std::cout << "\nInterpretation guardrails:\n"
                  << "  * V14 proved the V11 and V12 arithmetic paths are bit-exact in one full 4096-token trace.\n"
                  << "  * V15 changes only down POPC + residual; all normalization, attention and LM-head code is unchanged.\n"
                  << "  * The down-chain probe rotates through the full down weight ring; it is not a single L2-hot matrix.\n"
                  << "  * A small or zero full-schedule gain means int32 epilogue traffic is not worth broader fusion complexity.\n"
                  << "  * This remains a synthetic GPU-runtime ceiling, not checkpoint-validated text generation.\n";

        CUDA_CHECK(cudaStreamDestroy(sb));
        CUDA_CHECK(cudaStreamDestroy(sf));
        std::cout << "\nV15 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
