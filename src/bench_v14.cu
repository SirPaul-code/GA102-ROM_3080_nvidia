#define main ga102_rom_v11_embedded_main
#include "bench_v11.cu"
#undef main
#include "bench_v12lib.cuh"

#include <cstring>
#include <limits>

// V14 is diagnostic-only. It executes the V11 baseline and V12 fused-bitplane
// path in lockstep and stops at the first stage that differs. This isolates the
// ctx=4096 mismatch observed in V13 before any further optimization work.

__global__ void init_balanced_masks_v14(uint32_t* p, uint32_t* n,
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

static void rebalance_ring_v14(MaskRing& w, uint32_t seed) {
    const size_t count = w.words_per_copy * (size_t)w.copies;
    const int blocks = (int)std::min<size_t>(65535, (count + 255) / 256);
    init_balanced_masks_v14<<<blocks, 256>>>(w.pos, w.neg, count, seed);
    CUDA_CHECK(cudaGetLastError());
}

static void rebalance_runtime_v14(Runtime& r) {
    rebalance_ring_v14(r.qkv, 0x1234u);
    rebalance_ring_v14(r.out, 0x2345u);
    rebalance_ring_v14(r.gu, 0x3456u);
    rebalance_ring_v14(r.down, 0x4567u);
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct DiffInfo {
    bool equal = true;
    size_t index = 0;
    double max_abs = 0.0;
};

static float host_bf16_v14(const bf16& v) {
    uint16_t hi = 0;
    std::memcpy(&hi, &v, sizeof(hi));
    uint32_t bits = (uint32_t)hi << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static DiffInfo compare_u32(const uint32_t* a, const uint32_t* b, size_t n) {
    std::vector<uint32_t> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    DiffInfo d;
    for (size_t i = 0; i < n; ++i) {
        if (ha[i] != hb[i]) { d.equal = false; d.index = i; break; }
    }
    return d;
}

static DiffInfo compare_i32(const int32_t* a, const int32_t* b, size_t n) {
    std::vector<int32_t> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    DiffInfo d;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs((double)ha[i] - (double)hb[i]);
        if (e > d.max_abs) { d.max_abs = e; d.index = i; }
        if (ha[i] != hb[i]) d.equal = false;
    }
    return d;
}

static DiffInfo compare_bf16(const bf16* a, const bf16* b, size_t n) {
    std::vector<bf16> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(bf16), cudaMemcpyDeviceToHost));
    DiffInfo d;
    for (size_t i = 0; i < n; ++i) {
        const float fa = host_bf16_v14(ha[i]);
        const float fb = host_bf16_v14(hb[i]);
        const double e = std::fabs((double)fa - (double)fb);
        if (e > d.max_abs) { d.max_abs = e; d.index = i; }
        uint16_t ua = 0, ub = 0;
        std::memcpy(&ua, &ha[i], sizeof(uint16_t));
        std::memcpy(&ub, &hb[i], sizeof(uint16_t));
        if (ua != ub) d.equal = false;
    }
    return d;
}

static DiffInfo compare_f32(const float* a, const float* b, size_t n) {
    std::vector<float> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost));
    DiffInfo d;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs((double)ha[i] - (double)hb[i]);
        if (e > d.max_abs) { d.max_abs = e; d.index = i; }
        if (ha[i] != hb[i]) d.equal = false;
    }
    return d;
}

static bool report(const char* stage, int layer, const DiffInfo& d) {
    if (d.equal) {
        std::cout << "layer " << std::setw(2) << layer << "  " << std::left << std::setw(30)
                  << stage << " PASS" << std::right << "\n";
        return true;
    }
    std::cout << "layer " << std::setw(2) << layer << "  " << std::left << std::setw(30)
              << stage << " FAIL" << std::right
              << "  first/max-index=" << d.index << "  max_abs=" << d.max_abs << "\n";
    return false;
}

static bool sync2(cudaStream_t a, cudaStream_t b) {
    CUDA_CHECK(cudaStreamSynchronize(a));
    CUDA_CHECK(cudaStreamSynchronize(b));
    return true;
}

struct Options14 {
    int context = 4096;
    int ring_mib = 64;
};

static Options14 parse14(int argc, char** argv) {
    Options14 o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--context" && i + 1 < argc) o.context = std::stoi(argv[++i]);
        else if (a == "--ring-mib" && i + 1 < argc) o.ring_mib = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V14 stage divergence diagnostic\n"
                      << "  --context N   context to trace (default 4096)\n"
                      << "  --ring-mib N  packed-weight ring/projection (default 64)\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown or incomplete argument: " + a);
    }
    if (o.context < 128 || o.context > 4096 || o.ring_mib < 16)
        throw std::runtime_error("context must be 128..4096 and ring-mib >= 16");
    return o;
}

int main(int argc, char** argv) {
    try {
        const Options14 dopt = parse14(argc, argv);
        Options o;
        o.iters = 1;
        o.ring_mib = dopt.ring_mib;
        o.max_context = dopt.context;

        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 8 || prop.minor != 6 ||
            std::string(prop.name).find("RTX 3080") == std::string::npos) {
            std::cerr << "V14 restricted to RTX 3080 / SM86; found " << prop.name << "\n";
            return 3;
        }

        std::cout << "GA102-ROM V14: first-divergence trace\n"
                  << "GPU               : " << prop.name << "\n"
                  << "context           : " << dopt.context << "\n"
                  << "baseline          : V11 quant->INT8->pack\n"
                  << "candidate         : V12 direct quant->bitplanes\n"
                  << "weights           : balanced disjoint +/- masks\n"
                  << "mode              : lockstep, stage-by-stage, exact comparisons\n\n";

        Runtime base(o), fused(o);
        rebalance_runtime_v14(base);
        rebalance_runtime_v14(fused);

        cudaStream_t sb = nullptr, sf = nullptr;
        CUDA_CHECK(cudaStreamCreate(&sb));
        CUDA_CHECK(cudaStreamCreate(&sf));
        constexpr int t = 256;
        const int L = dopt.context;

        copy_bf16<<<(HIDDEN + t - 1) / t, t, 0, sb>>>(base.embed, base.hidden, HIDDEN);
        copy_bf16<<<(HIDDEN + t - 1) / t, t, 0, sf>>>(fused.embed, fused.hidden, HIDDEN);
        sync2(sb, sf);
        if (!report("initial hidden", -1, compare_bf16(base.hidden, fused.hidden, HIDDEN))) return 10;

        for (int layer = 0; layer < LAYERS; ++layer) {
            rmsnorm_quant_a8<<<1, t, 0, sb>>>(base.hidden, base.gamma_h, base.qh, base.scale_h, HIDDEN, 1e-6f);
            base.pack_hidden(sb);
            rmsnorm_quant_planes<<<1, t, 0, sf>>>(fused.hidden, fused.gamma_h, fused.planes_h, fused.scale_h, HIDDEN, 1e-6f);
            sync2(sb, sf);
            if (!report("input quant planes", layer, compare_u32(base.planes_h, fused.planes_h, (size_t)8 * (HIDDEN / 32)))) return 11;

            base.linear(sb, base.qkv, layer, base.planes_h, base.qkv_acc);
            fused.linear(sf, fused.qkv, layer, fused.planes_h, fused.qkv_acc);
            sync2(sb, sf);
            if (!report("QKV int32", layer, compare_i32(base.qkv_acc, fused.qkv_acc, QKV_DIM))) return 12;

            bf16* bkl = base.kcache + (size_t)layer * base.layer_kv_stride;
            bf16* bvl = base.vcache + (size_t)layer * base.layer_kv_stride;
            bf16* fkl = fused.kcache + (size_t)layer * fused.layer_kv_stride;
            bf16* fvl = fused.vcache + (size_t)layer * fused.layer_kv_stride;
            qkv_epilogue_rope_kv<<<(QKV_DIM + t - 1) / t, t, 0, sb>>>(base.qkv_acc, base.q, bkl, bvl, base.cs, base.sn, L - 1, 0.001f, 0.001f, 0.001f);
            qkv_epilogue_rope_kv<<<(QKV_DIM + t - 1) / t, t, 0, sf>>>(fused.qkv_acc, fused.q, fkl, fvl, fused.cs, fused.sn, L - 1, 0.001f, 0.001f, 0.001f);
            sync2(sb, sf);
            if (!report("Q after RoPE", layer, compare_bf16(base.q, fused.q, Q_DIM))) return 13;
            if (!report("K current token", layer, compare_bf16(bkl + (size_t)(L - 1) * KV_DIM, fkl + (size_t)(L - 1) * KV_DIM, KV_DIM))) return 14;
            if (!report("V current token", layer, compare_bf16(bvl + (size_t)(L - 1) * KV_DIM, fvl + (size_t)(L - 1) * KV_DIM, KV_DIM))) return 15;

            base.attention(sb, layer, L);
            fused.attention(sf, layer, L);
            sync2(sb, sf);
            if (!report("attention BF16", layer, compare_bf16(base.attn, fused.attn, HIDDEN))) return 16;

            rmsnorm_quant_a8<<<1, t, 0, sb>>>(base.attn, base.gamma_h, base.qh, base.scale_h, HIDDEN, 1e-6f);
            base.pack_hidden(sb);
            rmsnorm_quant_planes<<<1, t, 0, sf>>>(fused.attn, fused.gamma_h, fused.planes_h, fused.scale_h, HIDDEN, 1e-6f);
            sync2(sb, sf);
            if (!report("attn norm planes", layer, compare_u32(base.planes_h, fused.planes_h, (size_t)8 * (HIDDEN / 32)))) return 17;

            base.linear(sb, base.out, layer, base.planes_h, base.o_acc);
            fused.linear(sf, fused.out, layer, fused.planes_h, fused.o_acc);
            sync2(sb, sf);
            if (!report("O int32", layer, compare_i32(base.o_acc, fused.o_acc, HIDDEN))) return 18;

            o_resid_norm_quant<<<1, t, 0, sb>>>(base.o_acc, base.hidden, base.gamma_h, base.mid, base.qh, base.scale_h, 0.001f, 1e-6f);
            base.pack_hidden(sb);
            o_resid_norm_planes<<<1, t, 0, sf>>>(fused.o_acc, fused.hidden, fused.gamma_h, fused.mid, fused.planes_h, fused.scale_h, 0.001f, 1e-6f);
            sync2(sb, sf);
            if (!report("O residual BF16", layer, compare_bf16(base.mid, fused.mid, HIDDEN))) return 19;
            if (!report("O->gate planes", layer, compare_u32(base.planes_h, fused.planes_h, (size_t)8 * (HIDDEN / 32)))) return 20;

            base.linear(sb, base.gu, layer, base.planes_h, base.gu_acc);
            fused.linear(sf, fused.gu, layer, fused.planes_h, fused.gu_acc);
            sync2(sb, sf);
            if (!report("gate/up int32", layer, compare_i32(base.gu_acc, fused.gu_acc, GATEUP_DIM))) return 21;

            gateup_relu2_norm_quant<<<1, t, 0, sb>>>(base.gu_acc, base.gamma_i, base.tmpi, base.qi, base.scale_i, 0.0005f, 0.0005f, 1e-6f);
            base.pack_inter(sb);
            gateup_relu2_norm_planes<<<1, t, 0, sf>>>(fused.gu_acc, fused.gamma_i, fused.tmpi, fused.planes_i, fused.scale_i, 0.0005f, 0.0005f, 1e-6f);
            sync2(sb, sf);
            if (!report("gate/up BF16", layer, compare_bf16(base.tmpi, fused.tmpi, INTER))) return 22;
            if (!report("gate/up->down planes", layer, compare_u32(base.planes_i, fused.planes_i, (size_t)8 * (INTER / 32)))) return 23;

            base.linear(sb, base.down, layer, base.planes_i, base.down_acc);
            fused.linear(sf, fused.down, layer, fused.planes_i, fused.down_acc);
            sync2(sb, sf);
            if (!report("down int32", layer, compare_i32(base.down_acc, fused.down_acc, HIDDEN))) return 24;

            down_resid<<<(HIDDEN + t - 1) / t, t, 0, sb>>>(base.down_acc, base.mid, base.hidden, 0.001f);
            down_resid<<<(HIDDEN + t - 1) / t, t, 0, sf>>>(fused.down_acc, fused.mid, fused.hidden, 0.001f);
            sync2(sb, sf);
            if (!report("layer hidden", layer, compare_bf16(base.hidden, fused.hidden, HIDDEN))) return 25;
        }

        rmsnorm_bf16<<<1, t, 0, sb>>>(base.hidden, base.gamma_h, base.finalnorm, HIDDEN, 1e-6f);
        rmsnorm_bf16<<<1, t, 0, sf>>>(fused.hidden, fused.gamma_h, fused.finalnorm, HIDDEN, 1e-6f);
        sync2(sb, sf);
        if (!report("final RMSNorm", LAYERS, compare_bf16(base.finalnorm, fused.finalnorm, HIDDEN))) return 26;

        const int lm_threads = 128;
        const int lm_blocks = (VOCAB * 32 + lm_threads - 1) / lm_threads;
        lmhead_bf16_warp<<<lm_blocks, lm_threads, 0, sb>>>(base.lmw, base.finalnorm, base.logits, VOCAB, HIDDEN);
        lmhead_bf16_warp<<<lm_blocks, lm_threads, 0, sf>>>(fused.lmw, fused.finalnorm, fused.logits, VOCAB, HIDDEN);
        sync2(sb, sf);
        const DiffInfo ld = compare_f32(base.logits, fused.logits, VOCAB);
        if (!report("full LM logits", LAYERS, ld)) return 27;

        std::cout << "\nV14 result: PASS - no divergence found at context " << L << ".\n";
        CUDA_CHECK(cudaStreamDestroy(sb));
        CUDA_CHECK(cudaStreamDestroy(sf));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
