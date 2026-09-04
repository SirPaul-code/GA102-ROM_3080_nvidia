#define main ga102_rom_v17_embedded_main
#include "bench_v17.cu"
#undef main

// V18 attacks the remaining single-token BMMA under-occupancy visible in V17.
// A V17 warp owns one 16-row tile and walks every K=256 chunk serially.  That
// gives only 160 tiles for O/down and 240 for QKV, versus 864 for gate+up.
// V18 keeps the exact same BMMA N=8=activation-bitplanes mapping, but assigns
// multiple warps to one 16-row tile.  Each warp handles a disjoint subset of K
// chunks and the block reduces exact int32 partials in shared memory.

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

    // Four lanes own the eight N columns for one output row.  Collapse the
    // bitplane-weighted columns inside each 4-lane subgroup first.
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

    // Any first 16 threads can now reduce one output row each.  This is exact
    // integer addition; no floating point/reassociation is introduced.
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
        case 1: bmma_splitk18<1><<<f.tiles,  32, 0, s>>>(f.bp(layer),f.bn(layer),planes,out,f.tiles,f.chunks,f.words); break;
        case 2: bmma_splitk18<2><<<f.tiles,  64, 0, s>>>(f.bp(layer),f.bn(layer),planes,out,f.tiles,f.chunks,f.words); break;
        case 4: bmma_splitk18<4><<<f.tiles, 128, 0, s>>>(f.bp(layer),f.bn(layer),planes,out,f.tiles,f.chunks,f.words); break;
        case 8: bmma_splitk18<8><<<f.tiles, 256, 0, s>>>(f.bp(layer),f.bn(layer),planes,out,f.tiles,f.chunks,f.words); break;
        default: throw std::runtime_error("V18 warps/tile must be 1,2,4,8");
    }
}

static cudaGraphExec_t capture_split_family18(const Family17& f,
                                               const uint32_t* planes,
                                               int32_t* out, int warps,
                                               cudaStream_t s) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g=nullptr; cudaGraphExec_t e=nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    for(int l=0;l<L17;++l) launch_split18(f,l,planes,out,warps,s);
    CUDA_CHECK(cudaStreamEndCapture(s,&g));
    CUDA_CHECK(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static cudaGraphExec_t capture_split_all18(const Family17& qkv, const Family17& o,
                                            const Family17& gu, const Family17& down,
                                            const uint32_t* ph, const uint32_t* pi,
                                            int32_t* oq, int32_t* oo,
                                            int32_t* og, int32_t* od,
                                            const std::array<int,4>& warps,
                                            cudaStream_t s) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g=nullptr; cudaGraphExec_t e=nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    for(int l=0;l<L17;++l) {
        launch_split18(qkv,l,ph,oq,warps[0],s);
        launch_split18(o,l,ph,oo,warps[1],s);
        launch_split18(gu,l,ph,og,warps[2],s);
        launch_split18(down,l,pi,od,warps[3],s);
    }
    CUDA_CHECK(cudaStreamEndCapture(s,&g));
    CUDA_CHECK(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
}

static bool check_split18(const Family17& f, const uint32_t* planes,
                          int32_t* ref, int32_t* got, int warps,
                          cudaStream_t s) {
    for(int l=0;l<L17;++l) {
        launch_popc17(f,l,planes,ref,128,s);
        launch_split18(f,l,planes,got,warps,s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        int bad=-1;
        if(!equal_i32_17(ref,got,f.M,&bad)) {
            std::cerr<<f.name<<" split-K correctness FAIL warps="<<warps
                     <<" layer="<<l<<" row="<<bad<<"\n";
            return false;
        }
    }
    return true;
}

struct Choice18 { int warps=1; Stats17 st; };
static Choice18 sweep_split18(const Family17& f, const uint32_t* planes,
                              int32_t* out, cudaStream_t s,
                              int rounds, int batch) {
    const std::array<int,4> opts{{1,2,4,8}};
    Choice18 best; best.st.med=std::numeric_limits<float>::infinity();
    for(int w:opts) {
        auto e=capture_split_family18(f,planes,out,w,s);
        const Stats17 st=measure17(e,s,rounds,batch);
        CUDA_CHECK(cudaGraphExecDestroy(e));
        const double gbps=(double)f.bytes_all()/(st.med*1e6);
        const double gmac=(double)f.M*f.K*L17/(st.med*1e6);
        const float spread=100.0f*(st.max-st.min)/st.med;
        std::cout<<std::fixed<<std::setprecision(4)
                 <<"  split-K warps/tile "<<w<<"  "<<st.med<<" ms/30L"
                 <<"  "<<std::setprecision(1)<<gbps<<" GB/s"
                 <<"  "<<gmac<<" GMAC/s  spread="<<spread<<"%\n";
        if(st.med<best.st.med) best={w,st};
    }
    return best;
}

static Stats17 paired_family18(const Family17& f, const uint32_t* planes,
                               int32_t* out, int popc_threads,
                               cudaStream_t s, int rounds, int batch) {
    auto e=capture_family17(f,planes,out,popc_threads,false,s);
    const Stats17 st=measure17(e,s,rounds,batch);
    CUDA_CHECK(cudaGraphExecDestroy(e));
    return st;
}

struct Options18 { int rounds=9; int batch=100; };
static Options18 parse18(int argc,char**argv) {
    Options18 o;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--rounds"&&i+1<argc)o.rounds=std::stoi(argv[++i]);
        else if(a=="--batch"&&i+1<argc)o.batch=std::stoi(argv[++i]);
        else if(a=="-h"||a=="--help") {
            std::cout<<"GA102-ROM V18 split-K BMMA saturation benchmark\n"
                     <<"  --rounds N  timing rounds (default 9)\n"
                     <<"  --batch N   graph replays/round (default 100)\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown or incomplete argument: "+a);
    }
    if(o.rounds<3||o.batch<1) throw std::runtime_error("rounds>=3 and batch>=1 required");
    return o;
}

static void print_func18(const char* name, const void* fn) {
    cudaFuncAttributes a{};
    CUDA_CHECK(cudaFuncGetAttributes(&a,fn));
    std::cout<<name<<": regs/thread="<<a.numRegs
             <<" static-smem="<<a.sharedSizeBytes
             <<" local/thread="<<a.localSizeBytes
             <<" maxThreads="<<a.maxThreadsPerBlock<<"\n";
}

int main(int argc,char**argv) {
    try {
        const Options18 opt=parse18(argc,argv);
        int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
        if(prop.major!=8||prop.minor!=6||std::string(prop.name).find("RTX 3080")==std::string::npos) {
            std::cerr<<"V18 restricted to RTX 3080 / SM86; found "<<prop.name<<"\n"; return 3;
        }
        int memclk=0,bus=0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memclk,cudaDevAttrMemoryClockRate,dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&bus,cudaDevAttrGlobalMemoryBusWidth,dev));
        const double peak=2.0*(double)memclk*1000.0*((double)bus/8.0)/1e9;

        std::cout<<"GA102-ROM V18: split-K single-token BMMA saturation\n"
                 <<"GPU               : "<<prop.name<<"\n"
                 <<"idea              : 1/2/4/8 warps cooperate on each 16-row BMMA tile\n"
                 <<"reason            : V17 had only 160 O/down tiles, 240 QKV tiles, but 864 gate+up tiles\n"
                 <<"math              : exact N=8 activation-bitplane BMMA + int32 shared reduction\n"
                 <<"property BW       : "<<std::fixed<<std::setprecision(1)<<peak<<" GB/s\n";

        std::cout<<"\n=== V18 kernel resources ===\n";
        print_func18("POPC",(const void*)popc17);
        print_func18("V17 BMMA",(const void*)bmma_bitplanes17);
        print_func18("split-K x1",(const void*)bmma_splitk18<1>);
        print_func18("split-K x2",(const void*)bmma_splitk18<2>);
        print_func18("split-K x4",(const void*)bmma_splitk18<4>);
        print_func18("split-K x8",(const void*)bmma_splitk18<8>);

        Family17 qkv("QKV",QKV17,H17), o("O",H17,H17), gu("gate+up",GU17,H17), down("down",H17,I17);
        qkv.init(0x1234u); o.init(0x2345u); gu.init(0x3456u); down.init(0x4567u);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t *ph=nullptr,*pi=nullptr;
        CUDA_CHECK(cudaMalloc(&ph,(size_t)8*(H17/32)*sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&pi,(size_t)8*(I17/32)*sizeof(uint32_t)));
        init_u32_17<<<32,256>>>(ph,(size_t)8*(H17/32),0x11112222u);
        init_u32_17<<<32,256>>>(pi,(size_t)8*(I17/32),0x33334444u);

        int32_t *ref=nullptr,*got=nullptr,*oq=nullptr,*oo=nullptr,*og=nullptr,*od=nullptr;
        CUDA_CHECK(cudaMalloc(&ref,GU17*sizeof(int32_t))); CUDA_CHECK(cudaMalloc(&got,GU17*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&oq,QKV17*sizeof(int32_t))); CUDA_CHECK(cudaMalloc(&oo,H17*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&og,GU17*sizeof(int32_t))); CUDA_CHECK(cudaMalloc(&od,H17*sizeof(int32_t)));
        cudaStream_t s=nullptr; CUDA_CHECK(cudaStreamCreate(&s));

        // First expose the V16/V17 baseline discrepancy inside this exact TU.
        const std::array<int,4> pt{{64,128,64,128}};
        std::cout<<"\n=== V18 POPC family timings in same executable ===\n";
        const Stats17 pq=paired_family18(qkv,ph,oq,pt[0],s,opt.rounds,opt.batch);
        const Stats17 po=paired_family18(o,ph,oo,pt[1],s,opt.rounds,opt.batch);
        const Stats17 pg=paired_family18(gu,ph,og,pt[2],s,opt.rounds,opt.batch);
        const Stats17 pd=paired_family18(down,pi,od,pt[3],s,opt.rounds,opt.batch);
        const std::array<Stats17,4> pst{{pq,po,pg,pd}};
        const std::array<const Family17*,4> fs{{&qkv,&o,&gu,&down}};
        for(int i=0;i<4;++i) {
            const double bw=(double)fs[i]->bytes_all()/(pst[i].med*1e6);
            std::cout<<std::fixed<<std::setprecision(4)<<"  "<<std::setw(7)<<fs[i]->name
                     <<"  "<<pst[i].med<<" ms/30L  "<<std::setprecision(1)<<bw<<" GB/s"
                     <<"  spread="<<(100.0f*(pst[i].max-pst[i].min)/pst[i].med)<<"%\n";
        }
        std::cout<<std::setprecision(4)<<"  family-sum = "<<(pq.med+po.med+pg.med+pd.med)<<" ms\n";

        std::cout<<"\n=== V18 split-K sweep ===\n";
        std::cout<<"\nQKV\n"; const Choice18 cq=sweep_split18(qkv,ph,oq,s,opt.rounds,opt.batch);
        std::cout<<"\nO\n"; const Choice18 co=sweep_split18(o,ph,oo,s,opt.rounds,opt.batch);
        std::cout<<"\ngate+up\n"; const Choice18 cg=sweep_split18(gu,ph,og,s,opt.rounds,opt.batch);
        std::cout<<"\ndown\n"; const Choice18 cd=sweep_split18(down,pi,od,s,opt.rounds,opt.batch);
        const std::array<int,4> sw{{cq.warps,co.warps,cg.warps,cd.warps}};

        std::cout<<"\n=== V18 exactness of selected split-K mappings, all 30 layers ===\n";
        bool ok=true;
        bool x=check_split18(qkv,ph,ref,got,sw[0],s); ok&=x; std::cout<<"QKV       : "<<(x?"PASS":"FAIL")<<"\n";
        x=check_split18(o,ph,ref,got,sw[1],s); ok&=x; std::cout<<"O         : "<<(x?"PASS":"FAIL")<<"\n";
        x=check_split18(gu,ph,ref,got,sw[2],s); ok&=x; std::cout<<"gate+up   : "<<(x?"PASS":"FAIL")<<"\n";
        x=check_split18(down,pi,ref,got,sw[3],s); ok&=x; std::cout<<"down      : "<<(x?"PASS":"FAIL")<<"\n";
        if(!ok) { std::cerr<<"V18 stopped: selected split-K mapping is not exact.\n"; return 4; }

        // Fresh V17 reference sweep in the same executable, avoiding historical
        // clock/build differences when we compare the complete chain.
        std::cout<<"\n=== V18 fresh V17 BMMA reference sweep ===\n";
        std::cout<<"\nQKV\n"; const Best17 bq=sweep_bmma17(qkv,ph,oq,s,opt.rounds,opt.batch);
        std::cout<<"\nO\n"; const Best17 bo=sweep_bmma17(o,ph,oo,s,opt.rounds,opt.batch);
        std::cout<<"\ngate+up\n"; const Best17 bg=sweep_bmma17(gu,ph,og,s,opt.rounds,opt.batch);
        std::cout<<"\ndown\n"; const Best17 bd=sweep_bmma17(down,pi,od,s,opt.rounds,opt.batch);
        const std::array<int,4> bt{{bq.threads,bo.threads,bg.threads,bd.threads}};

        auto ep=capture_all17(qkv,o,gu,down,ph,pi,oq,oo,og,od,pt,false,s);
        auto eb=capture_all17(qkv,o,gu,down,ph,pi,oq,oo,og,od,bt,true,s);
        auto es=capture_split_all18(qkv,o,gu,down,ph,pi,oq,oo,og,od,sw,s);
        for(int i=0;i<4;++i){CUDA_CHECK(cudaGraphLaunch(ep,s));CUDA_CHECK(cudaGraphLaunch(eb,s));CUDA_CHECK(cudaGraphLaunch(es,s));}
        CUDA_CHECK(cudaStreamSynchronize(s));

        std::vector<float> vp,vb,vs; vp.reserve(opt.rounds);vb.reserve(opt.rounds);vs.reserve(opt.rounds);
        for(int r=0;r<opt.rounds;++r) {
            // Rotate order to reduce boost/temperature/order bias.
            if(r%3==0){vp.push_back(time_batch17(ep,s,opt.batch));vb.push_back(time_batch17(eb,s,opt.batch));vs.push_back(time_batch17(es,s,opt.batch));}
            else if(r%3==1){vb.push_back(time_batch17(eb,s,opt.batch));vs.push_back(time_batch17(es,s,opt.batch));vp.push_back(time_batch17(ep,s,opt.batch));}
            else {vs.push_back(time_batch17(es,s,opt.batch));vp.push_back(time_batch17(ep,s,opt.batch));vb.push_back(time_batch17(eb,s,opt.batch));}
        }
        const Stats17 sp=stats17(vp), sb=stats17(vb), ss=stats17(vs);
        CUDA_CHECK(cudaGraphExecDestroy(ep));CUDA_CHECK(cudaGraphExecDestroy(eb));CUDA_CHECK(cudaGraphExecDestroy(es));

        const size_t bytes=qkv.bytes_all()+o.bytes_all()+gu.bytes_all()+down.bytes_all();
        const double macs=2084044800.0;
        auto bw=[&](float ms){return (double)bytes/(ms*1e6);};
        auto gm=[&](float ms){return macs/(ms*1e6);};
        auto spread=[](const Stats17& z){return 100.0f*(z.max-z.min)/z.med;};

        std::cout<<"\n=== V18 complete 30-layer linear chain ===\n"
                 <<std::fixed<<std::setprecision(4)
                 <<"POPC tuned       : "<<sp.med<<" ms  "<<std::setprecision(1)<<bw(sp.med)<<" GB/s  "<<gm(sp.med)<<" GMAC/s  spread="<<spread(sp)<<"%\n"
                 <<std::setprecision(4)
                 <<"V17 BMMA         : "<<sb.med<<" ms  "<<std::setprecision(1)<<bw(sb.med)<<" GB/s  "<<gm(sb.med)<<" GMAC/s  spread="<<spread(sb)<<"%\n"
                 <<std::setprecision(4)
                 <<"V18 split-K BMMA : "<<ss.med<<" ms  "<<std::setprecision(1)<<bw(ss.med)<<" GB/s  "<<gm(ss.med)<<" GMAC/s  spread="<<spread(ss)<<"%\n"
                 <<"split/V17 speedup: "<<std::setprecision(3)<<(sb.med/ss.med)<<"x\n"
                 <<"split/POPC speedup: "<<(sp.med/ss.med)<<"x\n"
                 <<"property-BW eq   : "<<std::setprecision(1)<<(100.0*bw(ss.med)/peak)<<"%\n"
                 <<"V17 threads      : QKV="<<bt[0]<<" O="<<bt[1]<<" gate+up="<<bt[2]<<" down="<<bt[3]<<"\n"
                 <<"split warps/tile : QKV="<<sw[0]<<" O="<<sw[1]<<" gate+up="<<sw[2]<<" down="<<sw[3]<<"\n";

        std::cout<<"\nInterpretation guardrails:\n"
                 <<"  * Selected split-K mappings must PASS exact int32 equality on all 30 matrices/family.\n"
                 <<"  * POPC, V17 BMMA and V18 split-K are timed in the same executable and rotated order.\n"
                 <<"  * Family-sum vs full-chain POPC exposes whether the old V16 -> V17 baseline jump was a build/timing artifact.\n"
                 <<"  * GB/s counts only the invariant 496.875 MiB ternary P+N decoder weights.\n"
                 <<"  * Split-K adds only on-chip shared int32 reduction; it does not duplicate weight reads.\n";

        CUDA_CHECK(cudaStreamDestroy(s));
        cudaFree(ph);cudaFree(pi);cudaFree(ref);cudaFree(got);cudaFree(oq);cudaFree(oo);cudaFree(og);cudaFree(od);
        std::cout<<"\nV18 completed.\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"error: "<<e.what()<<"\n";
        return 1;
    }
}
