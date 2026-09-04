#define main ga102_rom_v11_embedded_main
#include "bench_v11.cu"
#undef main

// V12 removes the standalone INT8 materialization + pack_a8_to_planes launch.
// The normalization/epilogue kernel quantizes and directly emits the eight
// row-major A8 bitplanes consumed by the exact POPC linear kernel.

__global__ void rmsnorm_quant_planes(const bf16* x,
                                     const bf16* gamma,
                                     uint32_t* planes,
                                     float* qscale,
                                     int n,
                                     float eps) {
    __shared__ float s[8], invr, scale;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = bf(x[i]);
        ss += v * v;
    }
    const float sum = block_sum(ss, s);
    if (threadIdx.x == 0) invr = rsqrtf(sum / (float)n + eps);
    __syncthreads();

    float lm = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        lm = fmaxf(lm, fabsf(bf(x[i]) * invr * bf(gamma[i])));
    const float mx = block_max(lm, s);
    if (threadIdx.x == 0) {
        scale = 127.0f / fmaxf(mx, 1e-5f);
        qscale[0] = scale;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int words = n >> 5;
    for (int word = warp; word < words; word += warps) {
        const int i = word * 32 + lane;
        int z = __float2int_rn(bf(x[i]) * invr * bf(gamma[i]) * scale);
        z = max(-128, min(127, z));
        const uint8_t u = (uint8_t)(int8_t)z;
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = __ballot_sync(0xffffffffu, ((u >> bit) & 1u) != 0u);
            if (lane == 0) planes[(size_t)bit * words + word] = mask;
        }
    }
}

__global__ void o_resid_norm_planes(const int32_t* oacc,
                                    const bf16* residual,
                                    const bf16* gamma,
                                    bf16* mid,
                                    uint32_t* planes,
                                    float* qscale,
                                    float os,
                                    float eps) {
    __shared__ float s[8], invr, scale;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        const float v = bf(residual[i]) + (float)oacc[i] * os;
        mid[i] = b16(v);
        ss += v * v;
    }
    const float sum = block_sum(ss, s);
    if (threadIdx.x == 0) invr = rsqrtf(sum / (float)HIDDEN + eps);
    __syncthreads();

    float lm = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x)
        lm = fmaxf(lm, fabsf(bf(mid[i]) * invr * bf(gamma[i])));
    const float mx = block_max(lm, s);
    if (threadIdx.x == 0) {
        scale = 127.0f / fmaxf(mx, 1e-5f);
        qscale[0] = scale;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    constexpr int words = HIDDEN / 32;
    for (int word = warp; word < words; word += warps) {
        const int i = word * 32 + lane;
        int z = __float2int_rn(bf(mid[i]) * invr * bf(gamma[i]) * scale);
        z = max(-128, min(127, z));
        const uint8_t u = (uint8_t)(int8_t)z;
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = __ballot_sync(0xffffffffu, ((u >> bit) & 1u) != 0u);
            if (lane == 0) planes[(size_t)bit * words + word] = mask;
        }
    }
}

__global__ void gateup_relu2_norm_planes(const int32_t* gu,
                                         const bf16* gamma,
                                         bf16* tmp,
                                         uint32_t* planes,
                                         float* qscale,
                                         float gs,
                                         float us,
                                         float eps) {
    __shared__ float s[8], invr, scale;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < INTER; i += blockDim.x) {
        const float g = (float)gu[i] * gs;
        const float u = (float)gu[INTER + i] * us;
        const float r = fmaxf(g, 0.0f);
        const float z = r * r * u;
        tmp[i] = b16(z);
        ss += z * z;
    }
    const float sum = block_sum(ss, s);
    if (threadIdx.x == 0) invr = rsqrtf(sum / (float)INTER + eps);
    __syncthreads();

    float lm = 0.0f;
    for (int i = threadIdx.x; i < INTER; i += blockDim.x)
        lm = fmaxf(lm, fabsf(bf(tmp[i]) * invr * bf(gamma[i])));
    const float mx = block_max(lm, s);
    if (threadIdx.x == 0) {
        scale = 127.0f / fmaxf(mx, 1e-5f);
        qscale[0] = scale;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    constexpr int words = INTER / 32;
    for (int word = warp; word < words; word += warps) {
        const int i = word * 32 + lane;
        int z = __float2int_rn(bf(tmp[i]) * invr * bf(gamma[i]) * scale);
        z = max(-128, min(127, z));
        const uint8_t u = (uint8_t)(int8_t)z;
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = __ballot_sync(0xffffffffu, ((u >> bit) & 1u) != 0u);
            if (lane == 0) planes[(size_t)bit * words + word] = mask;
        }
    }
}

static void launch_v12(Runtime& r, cudaStream_t s, int L) {
    constexpr int t = 256;
    copy_bf16<<<(HIDDEN + t - 1) / t, t, 0, s>>>(r.embed, r.hidden, HIDDEN);

    for (int layer = 0; layer < LAYERS; ++layer) {
        rmsnorm_quant_planes<<<1, t, 0, s>>>(
            r.hidden, r.gamma_h, r.planes_h, r.scale_h, HIDDEN, 1e-6f);
        r.linear(s, r.qkv, layer, r.planes_h, r.qkv_acc);

        bf16* kl = r.kcache + (size_t)layer * r.layer_kv_stride;
        bf16* vl = r.vcache + (size_t)layer * r.layer_kv_stride;
        qkv_epilogue_rope_kv<<<(QKV_DIM + t - 1) / t, t, 0, s>>>(
            r.qkv_acc, r.q, kl, vl, r.cs, r.sn, L - 1, 0.001f, 0.001f, 0.001f);

        r.attention(s, layer, L);

        rmsnorm_quant_planes<<<1, t, 0, s>>>(
            r.attn, r.gamma_h, r.planes_h, r.scale_h, HIDDEN, 1e-6f);
        r.linear(s, r.out, layer, r.planes_h, r.o_acc);

        o_resid_norm_planes<<<1, t, 0, s>>>(
            r.o_acc, r.hidden, r.gamma_h, r.mid,
            r.planes_h, r.scale_h, 0.001f, 1e-6f);
        r.linear(s, r.gu, layer, r.planes_h, r.gu_acc);

        gateup_relu2_norm_planes<<<1, t, 0, s>>>(
            r.gu_acc, r.gamma_i, r.tmpi,
            r.planes_i, r.scale_i, 0.0005f, 0.0005f, 1e-6f);
        r.linear(s, r.down, layer, r.planes_i, r.down_acc);

        down_resid<<<(HIDDEN + t - 1) / t, t, 0, s>>>(
            r.down_acc, r.mid, r.hidden, 0.001f);
    }

    rmsnorm_bf16<<<1, t, 0, s>>>(r.hidden, r.gamma_h, r.finalnorm, HIDDEN, 1e-6f);
    constexpr int lm_threads = 128;
    const int lm_blocks = (VOCAB * 32 + lm_threads - 1) / lm_threads;
    lmhead_bf16_warp<<<lm_blocks, lm_threads, 0, s>>>(
        r.lmw, r.finalnorm, r.logits, VOCAB, HIDDEN);
}

static float time_direct_v12(Runtime& r, cudaStream_t s, int L, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch_v12(r, s, L);
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < iters; ++i) launch_v12(r, s, L);
    CUDA_CHECK(cudaEventRecord(b, s)); CUDA_CHECK(cudaEventSynchronize(b));
    float total = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&total, a, b));
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    return total / iters;
}

static float time_graph_v12(Runtime& r, cudaStream_t s, int L, int warmup, int iters,
                            cudaGraphExec_t* out_exec = nullptr) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr;
    cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    launch_v12(r, s, L);
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
    for (int i = 0; i < warmup; ++i) CUDA_CHECK(cudaGraphLaunch(e, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < iters; ++i) CUDA_CHECK(cudaGraphLaunch(e, s));
    CUDA_CHECK(cudaEventRecord(b, s)); CUDA_CHECK(cudaEventSynchronize(b));
    float total = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&total, a, b));
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    if (out_exec) *out_exec = e; else CUDA_CHECK(cudaGraphExecDestroy(e));
    CUDA_CHECK(cudaGraphDestroy(g));
    return total / iters;
}

static Options parse_v12_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "--ring-mib" && i + 1 < argc) o.ring_mib = std::stoi(argv[++i]);
        else if (a == "--max-context" && i + 1 < argc) o.max_context = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V12 fused quant-to-bitplane decoder benchmark\n"
                      << "  --iters N        iterations/context (default 50)\n"
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
        const Options o = parse_v12_args(argc, argv);
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 8 || prop.minor != 6 ||
            std::string(prop.name).find("RTX 3080") == std::string::npos) {
            std::cerr << "V12 restricted to RTX 3080 / SM86; found " << prop.name << "\n";
            return 3;
        }

        std::cout << "GA102-ROM V12: fuse A8 quantization directly into POPC bitplanes\n"
                  << "GPU               : " << prop.name << "\n"
                  << "baseline          : V11 full 30-layer CUDA Graph schedule\n"
                  << "fusion            : remove INT8 q buffers + 4 pack kernels/layer\n"
                  << "removed launches  : 120 pack kernels/token\n"
                  << "comparison        : V11 Graph vs V12 fused Graph, identical weights/dependencies\n";

        Runtime r(o);
        cudaStream_t s = nullptr;
        CUDA_CHECK(cudaStreamCreate(&s));

        const std::array<int, 5> ctxs{{128, 512, 1024, 2048, 4096}};
        std::cout << "\n=== V12 full decoder fusion benchmark ===\n";
        for (const int L : ctxs) {
            if (L > o.max_context) continue;

            cudaGraphExec_t old_exec = nullptr;
            const float old_graph = time_graph(r, s, L, 3, o.iters, &old_exec);
            CUDA_CHECK(cudaGraphLaunch(old_exec, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            std::vector<float> old_logits(64);
            CUDA_CHECK(cudaMemcpy(old_logits.data(), r.logits,
                                  old_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));

            cudaGraphExec_t new_exec = nullptr;
            const float fused_graph = time_graph_v12(r, s, L, 3, o.iters, &new_exec);
            CUDA_CHECK(cudaGraphLaunch(new_exec, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            std::vector<float> new_logits(64);
            CUDA_CHECK(cudaMemcpy(new_logits.data(), r.logits,
                                  new_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));

            float diff = 0.0f;
            for (size_t i = 0; i < old_logits.size(); ++i)
                diff = std::max(diff, std::fabs(old_logits[i] - new_logits[i]));

            const float fused_direct = time_direct_v12(r, s, L, 3, o.iters);
            CUDA_CHECK(cudaGraphExecDestroy(old_exec));
            CUDA_CHECK(cudaGraphExecDestroy(new_exec));

            std::cout << std::fixed << std::setprecision(4)
                      << "ctx " << std::setw(4) << L
                      << "  V11-graph=" << old_graph << " ms"
                      << "  V12-graph=" << fused_graph << " ms"
                      << "  fusion-speedup=" << std::setprecision(2) << (old_graph / fused_graph) << "x"
                      << "  V12-direct=" << std::setprecision(4) << fused_direct << " ms"
                      << "  graph-rate=" << std::setprecision(1) << (1000.0f / fused_graph) << " tok/s"
                      << "  output-diff=" << std::setprecision(6) << diff << "\n";
        }

        std::cout << "\nGuardrails:\n"
                  << "  * V12 keeps V11 POPC matrices, V9 attention, final norm and full BF16 LM head unchanged.\n"
                  << "  * Only the A8 quantize->INT8->pack path is replaced by direct ballot bitplane emission.\n"
                  << "  * V11 and V12 timings use the same Runtime allocations and synthetic weights.\n"
                  << "  * output-diff validates first 64 final logits; zero is expected because quantization arithmetic is identical.\n"
                  << "  * This remains a synthetic GPU-runtime ceiling, not validated checkpoint generation throughput.\n";

        CUDA_CHECK(cudaStreamDestroy(s));
        std::cout << "\nV12 completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
