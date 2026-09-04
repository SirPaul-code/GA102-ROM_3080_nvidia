// Reusable V12 fused quant-to-bitplane kernels.
// This header assumes bench_v11.cu has already been included in the translation unit.

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
