#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

static constexpr int H17 = 2560;
static constexpr int I17 = 6912;
static constexpr int QKV17 = 3840;
static constexpr int GU17 = 13824;
static constexpr int L17 = 30;

__device__ __forceinline__ int warp_sum17(int v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ void bmma17(
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

__global__ void init_masks17(uint32_t* p, uint32_t* n, size_t count, uint32_t seed) {
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

__global__ void init_u32_17(uint32_t* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        uint32_t z = (uint32_t)i * 747796405u + seed;
        z = ((z >> ((z >> 28) + 4)) ^ z) * 277803737u;
        z = (z >> 22) ^ z;
        p[i] = z;
    }
}

__global__ void popc17(const uint32_t* pos, const uint32_t* neg,
                       const uint32_t* xb, int32_t* y, int M, int words) {
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
            const int sc = (bit == 7) ? -128 : (1 << bit);
            acc += cnt * sc;
        }
    }
    acc = warp_sum17(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void pack_bmma17(const uint32_t* pos, const uint32_t* neg,
                            uint4* pp, uint4* np,
                            int M, int words, int chunks, int tiles, int layers) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)layers * tiles * chunks * 32;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        size_t x = idx;
        const int lane = (int)(x % 32); x /= 32;
        const int chunk = (int)(x % chunks); x /= chunks;
        const int tile = (int)(x % tiles); x /= tiles;
        const int layer = (int)x;

        const int group = lane >> 2;
        const int tid4 = lane & 3;
        const int r0 = tile * 16 + group;
        const int r8 = r0 + 8;
        const int w0 = chunk * 8 + tid4;
        const int w1 = w0 + 4;
        const size_t lb = (size_t)layer * M * words;

        pp[idx] = make_uint4(
            pos[lb + (size_t)r0 * words + w0],
            pos[lb + (size_t)r8 * words + w0],
            pos[lb + (size_t)r0 * words + w1],
            pos[lb + (size_t)r8 * words + w1]);
        np[idx] = make_uint4(
            neg[lb + (size_t)r0 * words + w0],
            neg[lb + (size_t)r8 * words + w0],
            neg[lb + (size_t)r0 * words + w1],
            neg[lb + (size_t)r8 * words + w1]);
    }
}

// Key V17 idea: BMMA's native N=8 columns are the eight A8 bitplanes of ONE
// token, not eight different token states. Each K=256 chunk therefore needs
// only two BMMA instructions (positive and negative masks), rather than the
// old single-token path's 8 bitplanes x 2 BMMA instructions.
__global__ void bmma_bitplanes17(const uint4* posp, const uint4* negp,
                                 const uint32_t* xb, int32_t* y,
                                 int tiles, int chunks, int words) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile = tid >> 5;
    const int lane = threadIdx.x & 31;
    if (tile >= tiles) return;

    const int group = lane >> 2;   // B column = activation bitplane 0..7
    const int tid4 = lane & 3;
    const int col0 = tid4 * 2;
    const int col1 = col0 + 1;
    const int sc0 = (col0 == 7) ? -128 : (1 << col0);
    const int sc1 = (col1 == 7) ? -128 : (1 << col1);

    int lane_row0 = 0;
    int lane_row8 = 0;

    for (int chunk = 0; chunk < chunks; ++chunk) {
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

        lane_row0 += (pc0 - nc0) * sc0 + (pc1 - nc1) * sc1;
        lane_row8 += (pc2 - nc2) * sc0 + (pc3 - nc3) * sc1;
    }

    lane_row0 += __shfl_down_sync(0xffffffffu, lane_row0, 2, 4);
    lane_row8 += __shfl_down_sync(0xffffffffu, lane_row8, 2, 4);
    lane_row0 += __shfl_down_sync(0xffffffffu, lane_row0, 1, 4);
    lane_row8 += __shfl_down_sync(0xffffffffu, lane_row8, 1, 4);

    if (tid4 == 0) {
        const int r0 = tile * 16 + group;
        y[r0] = lane_row0;
        y[r0 + 8] = lane_row8;
    }
}

struct Family17 {
    const char* name;
    int M, K, words, chunks, tiles;
    uint32_t *pos = nullptr, *neg = nullptr;
    uint4 *pp = nullptr, *np = nullptr;

    Family17(const char* n, int m, int k) : name(n), M(m), K(k) {
        words = K / 32;
        chunks = K / 256;
        tiles = M / 16;
    }
    void init(uint32_t seed) {
        const size_t rw = (size_t)L17 * M * words;
        CUDA_CHECK(cudaMalloc(&pos, rw * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&neg, rw * sizeof(uint32_t)));
        int b = (int)std::min<size_t>(65535, (rw + 255) / 256);
        init_masks17<<<b,256>>>(pos, neg, rw, seed);
        CUDA_CHECK(cudaGetLastError());

        const size_t pf = (size_t)L17 * tiles * chunks * 32;
        CUDA_CHECK(cudaMalloc(&pp, pf * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&np, pf * sizeof(uint4)));
        b = (int)std::min<size_t>(65535, (pf + 255) / 256);
        pack_bmma17<<<b,256>>>(pos, neg, pp, np, M, words, chunks, tiles, L17);
        CUDA_CHECK(cudaGetLastError());
    }
    const uint32_t* p(int l) const { return pos + (size_t)l * M * words; }
    const uint32_t* n(int l) const { return neg + (size_t)l * M * words; }
    const uint4* bp(int l) const { return pp + (size_t)l * tiles * chunks * 32; }
    const uint4* bn(int l) const { return np + (size_t)l * tiles * chunks * 32; }
    size_t bytes_all() const { return (size_t)L17 * 2 * M * words * sizeof(uint32_t); }
    double mib_all() const { return (double)bytes_all() / 1048576.0; }
    ~Family17() { cudaFree(pos); cudaFree(neg); cudaFree(pp); cudaFree(np); }
};

static void launch_popc17(const Family17& f, int layer, const uint32_t* planes,
                          int32_t* out, int threads, cudaStream_t s) {
    const int blocks = (f.M * 32 + threads - 1) / threads;
    popc17<<<blocks,threads,0,s>>>(f.p(layer), f.n(layer), planes, out, f.M, f.words);
}

static void launch_bmma17(const Family17& f, int layer, const uint32_t* planes,
                          int32_t* out, int threads, cudaStream_t s) {
    const int blocks = (f.tiles * 32 + threads - 1) / threads;
    bmma_bitplanes17<<<blocks,threads,0,s>>>(f.bp(layer), f.bn(layer), planes,
                                            out, f.tiles, f.chunks, f.words);
}

static bool equal_i32_17(const int32_t* a, const int32_t* b, int n, int* bad = nullptr) {
    std::vector<int32_t> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (ha[i] != hb[i]) { if (bad) *bad = i; return false; }
    }
    return true;
}

static bool check_family17(const Family17& f, const uint32_t* planes,
                           int32_t* a, int32_t* b, cudaStream_t s) {
    for (int l = 0; l < L17; ++l) {
        launch_popc17(f, l, planes, a, 128, s);
        launch_bmma17(f, l, planes, b, 128, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        int bad = -1;
        if (!equal_i32_17(a, b, f.M, &bad)) {
            std::cerr << f.name << " correctness FAIL layer=" << l
                      << " row=" << bad << "\n";
            return false;
        }
    }
    return true;
}

static cudaGraphExec_t capture_family17(const Family17& f, const uint32_t* planes,
                                        int32_t* out, int threads, bool bmma,
                                        cudaStream_t s) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr; cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int l = 0; l < L17; ++l) {
        if (bmma) launch_bmma17(f,l,planes,out,threads,s);
        else launch_popc17(f,l,planes,out,threads,s);
    }
    CUDA_CHECK(cudaStreamEndCapture(s,&g));
    CUDA_CHECK(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static cudaGraphExec_t capture_all17(const Family17& qkv, const Family17& o,
                                     const Family17& gu, const Family17& down,
                                     const uint32_t* ph, const uint32_t* pi,
                                     int32_t* oq, int32_t* oo, int32_t* og, int32_t* od,
                                     const std::array<int,4>& th, bool bmma,
                                     cudaStream_t s) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g = nullptr; cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int l = 0; l < L17; ++l) {
        if (bmma) {
            launch_bmma17(qkv,l,ph,oq,th[0],s); launch_bmma17(o,l,ph,oo,th[1],s);
            launch_bmma17(gu,l,ph,og,th[2],s); launch_bmma17(down,l,pi,od,th[3],s);
        } else {
            launch_popc17(qkv,l,ph,oq,th[0],s); launch_popc17(o,l,ph,oo,th[1],s);
            launch_popc17(gu,l,ph,og,th[2],s); launch_popc17(down,l,pi,od,th[3],s);
        }
    }
    CUDA_CHECK(cudaStreamEndCapture(s,&g));
    CUDA_CHECK(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static float time_batch17(cudaGraphExec_t e, cudaStream_t s, int batch) {
    cudaEvent_t a=nullptr,b=nullptr;
    CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a,s));
    for(int i=0;i<batch;++i) CUDA_CHECK(cudaGraphLaunch(e,s));
    CUDA_CHECK(cudaEventRecord(b,s)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    return ms / batch;
}

struct Stats17 { float med=0,min=0,max=0; };
static Stats17 stats17(std::vector<float> v) {
    std::sort(v.begin(),v.end());
    return {v[v.size()/2],v.front(),v.back()};
}

static Stats17 measure17(cudaGraphExec_t e, cudaStream_t s, int rounds, int batch) {
    for(int i=0;i<5;++i) CUDA_CHECK(cudaGraphLaunch(e,s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<float> v; v.reserve(rounds);
    for(int r=0;r<rounds;++r) v.push_back(time_batch17(e,s,batch));
    return stats17(v);
}

struct Best17 { int threads=128; Stats17 st; };
static Best17 sweep_bmma17(const Family17& f, const uint32_t* planes,
                           int32_t* out, cudaStream_t s, int rounds, int batch) {
    const std::array<int,5> opts{{32,64,128,256,512}};
    Best17 best; best.st.med = std::numeric_limits<float>::infinity();
    for(int t:opts) {
        auto e=capture_family17(f,planes,out,t,true,s);
        auto st=measure17(e,s,rounds,batch);
        CUDA_CHECK(cudaGraphExecDestroy(e));
        const double gbps=(double)f.bytes_all()/(st.med*1e6);
        const double gmac=(double)f.M*f.K*L17/(st.med*1e6);
        const float spread=100.0f*(st.max-st.min)/st.med;
        std::cout<<std::fixed<<std::setprecision(4)
                 <<"  BMMA threads "<<std::setw(3)<<t<<"  "<<st.med<<" ms/30L"
                 <<"  "<<std::setprecision(1)<<gbps<<" GB/s"
                 <<"  "<<gmac<<" GMAC/s  spread="<<spread<<"%\n";
        if(st.med<best.st.med) best={t,st};
    }
    return best;
}

struct Options17 { int rounds=7; int batch=100; };
static Options17 parse17(int argc,char**argv) {
    Options17 o;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--rounds"&&i+1<argc)o.rounds=std::stoi(argv[++i]);
        else if(a=="--batch"&&i+1<argc)o.batch=std::stoi(argv[++i]);
        else if(a=="-h"||a=="--help") {
            std::cout<<"GA102-ROM V17 single-token BMMA bitplane-columns\n"
                     <<"  --rounds N  timing rounds (default 7)\n"
                     <<"  --batch N   graph replays/round (default 100)\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown or incomplete argument: "+a);
    }
    if(o.rounds<3||o.batch<1) throw std::runtime_error("rounds>=3 and batch>=1 required");
    return o;
}

int main(int argc,char**argv) {
    try {
        const Options17 opt=parse17(argc,argv);
        int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
        if(prop.major!=8||prop.minor!=6||std::string(prop.name).find("RTX 3080")==std::string::npos) {
            std::cerr<<"V17 restricted to RTX 3080 / SM86; found "<<prop.name<<"\n"; return 3;
        }
        int memclk=0,bus=0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memclk,cudaDevAttrMemoryClockRate,dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&bus,cudaDevAttrGlobalMemoryBusWidth,dev));
        const double peak=2.0*(double)memclk*1000.0*((double)bus/8.0)/1e9;

        std::cout<<"GA102-ROM V17: single-token BMMA with N=8 activation bitplanes\n"
                 <<"GPU               : "<<prop.name<<"\n"
                 <<"old single BMMA   : 8 bitplanes x (P BMMA + N BMMA) = 16 BMMA/chunk\n"
                 <<"V17 single BMMA   : N=8 columns ARE bitplanes, so 2 BMMA/chunk\n"
                 <<"weight footprint  : 30 unique matrices/family, exact same 2-bit P/N bytes\n"
                 <<"property BW       : "<<std::fixed<<std::setprecision(1)<<peak<<" GB/s\n";

        Family17 qkv("QKV",QKV17,H17), o("O",H17,H17), gu("gate+up",GU17,H17), down("down",H17,I17);
        qkv.init(0x1234u); o.init(0x2345u); gu.init(0x3456u); down.init(0x4567u);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout<<"\n=== V17 footprint ===\n"
                 <<"QKV="<<qkv.mib_all()<<" MiB  O="<<o.mib_all()<<" MiB  gate+up="<<gu.mib_all()
                 <<" MiB  down="<<down.mib_all()<<" MiB  TOTAL="
                 <<(qkv.mib_all()+o.mib_all()+gu.mib_all()+down.mib_all())<<" MiB\n";

        uint32_t *ph=nullptr,*pi=nullptr;
        CUDA_CHECK(cudaMalloc(&ph,(size_t)8*(H17/32)*sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&pi,(size_t)8*(I17/32)*sizeof(uint32_t)));
        init_u32_17<<<32,256>>>(ph,(size_t)8*(H17/32),0x11112222u);
        init_u32_17<<<32,256>>>(pi,(size_t)8*(I17/32),0x33334444u);

        int32_t *a=nullptr,*b=nullptr,*oq=nullptr,*oo=nullptr,*og=nullptr,*od=nullptr;
        CUDA_CHECK(cudaMalloc(&a,GU17*sizeof(int32_t))); CUDA_CHECK(cudaMalloc(&b,GU17*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&oq,QKV17*sizeof(int32_t))); CUDA_CHECK(cudaMalloc(&oo,H17*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&og,GU17*sizeof(int32_t))); CUDA_CHECK(cudaMalloc(&od,H17*sizeof(int32_t)));
        cudaStream_t s=nullptr; CUDA_CHECK(cudaStreamCreate(&s));

        std::cout<<"\n=== V17 exactness: POPC vs bitplane-column BMMA, all 30 layers ===\n";
        bool ok=true;
        ok &= check_family17(qkv,ph,a,b,s); std::cout<<"QKV       : "<<(ok?"PASS":"FAIL")<<"\n";
        bool x=check_family17(o,ph,a,b,s); ok&=x; std::cout<<"O         : "<<(x?"PASS":"FAIL")<<"\n";
        x=check_family17(gu,ph,a,b,s); ok&=x; std::cout<<"gate+up   : "<<(x?"PASS":"FAIL")<<"\n";
        x=check_family17(down,pi,a,b,s); ok&=x; std::cout<<"down      : "<<(x?"PASS":"FAIL")<<"\n";
        if(!ok) { std::cerr<<"V17 stopped: BMMA bitplane-column mapping is not exact.\n"; return 4; }

        const std::array<int,4> popc_threads{{64,128,64,128}};
        std::cout<<"\n=== V17 BMMA block-size sweep ===\n";
        std::cout<<"\nQKV\n"; const Best17 bq=sweep_bmma17(qkv,ph,oq,s,opt.rounds,opt.batch);
        std::cout<<"\nO\n"; const Best17 bo=sweep_bmma17(o,ph,oo,s,opt.rounds,opt.batch);
        std::cout<<"\ngate+up\n"; const Best17 bg=sweep_bmma17(gu,ph,og,s,opt.rounds,opt.batch);
        std::cout<<"\ndown\n"; const Best17 bd=sweep_bmma17(down,pi,od,s,opt.rounds,opt.batch);
        const std::array<int,4> bt{{bq.threads,bo.threads,bg.threads,bd.threads}};

        auto ep=capture_all17(qkv,o,gu,down,ph,pi,oq,oo,og,od,popc_threads,false,s);
        auto eb=capture_all17(qkv,o,gu,down,ph,pi,oq,oo,og,od,bt,true,s);
        for(int i=0;i<5;++i){CUDA_CHECK(cudaGraphLaunch(ep,s));CUDA_CHECK(cudaGraphLaunch(eb,s));}
        CUDA_CHECK(cudaStreamSynchronize(s));
        std::vector<float> tp,tb;
        for(int r=0;r<opt.rounds;++r) {
            if((r&1)==0) { tp.push_back(time_batch17(ep,s,opt.batch)); tb.push_back(time_batch17(eb,s,opt.batch)); }
            else { tb.push_back(time_batch17(eb,s,opt.batch)); tp.push_back(time_batch17(ep,s,opt.batch)); }
        }
        const Stats17 sp=stats17(tp), sb=stats17(tb);
        CUDA_CHECK(cudaGraphExecDestroy(ep)); CUDA_CHECK(cudaGraphExecDestroy(eb));
        const size_t bytes=qkv.bytes_all()+o.bytes_all()+gu.bytes_all()+down.bytes_all();
        const double macs=2084044800.0;
        const double bwp=(double)bytes/(sp.med*1e6), bwb=(double)bytes/(sb.med*1e6);
        const double gp=macs/(sp.med*1e6), gb=macs/(sb.med*1e6);

        std::cout<<"\n=== V17 complete 30-layer linear chain ===\n"
                 <<std::fixed<<std::setprecision(4)
                 <<"V16 tuned POPC : "<<sp.med<<" ms/token  "<<std::setprecision(1)<<bwp<<" GB/s  "<<gp<<" GMAC/s"
                 <<"  spread="<<(100.0f*(sp.max-sp.min)/sp.med)<<"%\n"
                 <<std::setprecision(4)
                 <<"V17 BMMA N8bit : "<<sb.med<<" ms/token  "<<std::setprecision(1)<<bwb<<" GB/s  "<<gb<<" GMAC/s"
                 <<"  spread="<<(100.0f*(sb.max-sb.min)/sb.med)<<"%\n"
                 <<"speedup         : "<<std::setprecision(3)<<(sp.med/sb.med)<<"x\n"
                 <<"property-BW eq  : "<<std::setprecision(1)<<(100.0*bwb/peak)<<"%\n"
                 <<"BMMA threads    : QKV="<<bt[0]<<" O="<<bt[1]<<" gate+up="<<bt[2]<<" down="<<bt[3]<<"\n";

        std::cout<<"\nGuardrails:\n"
                 <<"  * PASS means exact int32 equality against the current bitplane POPC path for every one of the 30 matrices/family.\n"
                 <<"  * V17 changes the interpretation of BMMA N=8: columns are the eight signed-A8 bitplanes of one token.\n"
                 <<"  * Runtime storage is still 496.875 MiB for decoder ternary weights; row-major + BMMA copies coexist only in this benchmark for A/B correctness.\n"
                 <<"  * GB/s counts packed ternary weight bytes only. If V17 approaches property BW, single-token linears have become memory-streaming limited.\n";

        CUDA_CHECK(cudaStreamDestroy(s));
        cudaFree(ph);cudaFree(pi);cudaFree(a);cudaFree(b);cudaFree(oq);cudaFree(oo);cudaFree(og);cudaFree(od);
        std::cout<<"\nV17 completed.\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"error: "<<e.what()<<"\n";
        return 1;
    }
}
