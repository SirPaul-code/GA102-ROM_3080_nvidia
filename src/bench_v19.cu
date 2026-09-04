#define main ga102_rom_v17_embedded_main
#include "bench_v17.cu"
#undef main

// V19 follows directly from V18: the split-K mapping raised the 30-layer
// decoder-linear stream to 531.7 GB/s equivalent, so the remaining lever is
// physical weight traffic.  V17/V18 store exact ternary weights as two one-bit
// masks (P/N) = 2.000 bits/weight.  V19 stores 10 trits in one uint16 because
// 3^10 = 59049 < 65536.  A native BMMA lane fragment contains 128 trits, so
// 13 uint16 codes hold it exactly: 26 B instead of 32 B = 1.625 bits/weight.
// Packing is offline.  The timed kernel expands the compressed ROM in registers
// and feeds the same exact SM86 b1 BMMA instructions.

static constexpr int CODES19 = 13;
static constexpr int LUT19 = 243; // 3^5
__device__ __align__(16) uint16_t g_lut19[LUT19];

__global__ void pack_trits19(const uint4* __restrict__ pp,
                             const uint4* __restrict__ np,
                             uint16_t* __restrict__ rom,
                             size_t lane_frags) {
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    const size_t stride=(size_t)gridDim.x*blockDim.x;
    for(;idx<lane_frags;idx+=stride) {
        const uint4 p=pp[idx], n=np[idx];
        const uint32_t pv[4]={p.x,p.y,p.z,p.w};
        const uint32_t nv[4]={n.x,n.y,n.z,n.w};
        const size_t frag=idx>>5;
        const int lane=(int)(idx&31u);
#pragma unroll
        for(int g=0;g<CODES19;++g) {
            unsigned code=0, mul=1;
#pragma unroll
            for(int j=0;j<10;++j) {
                const int bit=g*10+j;
                unsigned d=0;
                if(bit<128) {
                    const int wi=bit>>5, sh=bit&31;
                    const unsigned pb=(pv[wi]>>sh)&1u;
                    const unsigned nb=(nv[wi]>>sh)&1u;
                    d=pb?1u:(nb?2u:0u);
                }
                code += d*mul;
                mul *= 3u;
            }
            rom[(frag*CODES19+g)*32+lane]=(uint16_t)code;
        }
    }
}

__device__ __forceinline__ void put5_19(uint32_t& w0,uint32_t& w1,
                                        uint32_t& w2,uint32_t& w3,
                                        int pos,uint32_t bits) {
    bits&=31u;
    const int wi=pos>>5, sh=pos&31;
    const uint32_t lo=bits<<sh;
    if(wi==0)w0|=lo; else if(wi==1)w1|=lo; else if(wi==2)w2|=lo; else if(wi==3)w3|=lo;
    if(sh>27 && wi<3) {
        const uint32_t hi=bits>>(32-sh);
        if(wi==0)w1|=hi; else if(wi==1)w2|=hi; else if(wi==2)w3|=hi;
    }
}

__device__ __forceinline__ void decode19(const uint16_t* __restrict__ rom,
                                         size_t frag,int lane,
                                         uint32_t& p0,uint32_t& p1,uint32_t& p2,uint32_t& p3,
                                         uint32_t& n0,uint32_t& n1,uint32_t& n2,uint32_t& n3) {
    p0=p1=p2=p3=0u; n0=n1=n2=n3=0u;
#pragma unroll
    for(int g=0;g<CODES19;++g) {
        const uint16_t code=rom[(frag*CODES19+g)*32+lane];
        const unsigned lo=(unsigned)code%243u;
        const unsigned hi=(unsigned)code/243u;
        const uint16_t a=__ldg(&g_lut19[lo]);
        const uint16_t b=__ldg(&g_lut19[hi]);
        put5_19(p0,p1,p2,p3,g*10,    (uint32_t)(a&31u));
        put5_19(n0,n1,n2,n3,g*10,    (uint32_t)((a>>5)&31u));
        put5_19(p0,p1,p2,p3,g*10+5,  (uint32_t)(b&31u));
        put5_19(n0,n1,n2,n3,g*10+5,  (uint32_t)((b>>5)&31u));
    }
}

template<int WARPS>
__global__ void bmma_tritrom19(const uint16_t* __restrict__ rom,
                               const uint32_t* __restrict__ xb,
                               int32_t* __restrict__ y,
                               int tiles,int chunks,int words) {
    const int tile=blockIdx.x;
    const int warp=threadIdx.x>>5;
    const int lane=threadIdx.x&31;
    if(tile>=tiles||warp>=WARPS)return;

    const int group=lane>>2, tid4=lane&3;
    const int col0=tid4*2, col1=col0+1;
    const int sc0=(col0==7)?-128:(1<<col0);
    const int sc1=(col1==7)?-128:(1<<col1);
    int part0=0,part8=0;

    for(int chunk=warp;chunk<chunks;chunk+=WARPS) {
        const size_t frag=(size_t)tile*chunks+chunk;
        uint32_t p0,p1,p2,p3,n0,n1,n2,n3;
        decode19(rom,frag,lane,p0,p1,p2,p3,n0,n1,n2,n3);

        const uint32_t* plane=xb+(size_t)group*words;
        const int wbase=chunk*8;
        const uint32_t b0=plane[wbase+tid4];
        const uint32_t b1=plane[wbase+4+tid4];
        int pc0=0,pc1=0,pc2=0,pc3=0,nc0=0,nc1=0,nc2=0,nc3=0;
        bmma17(p0,p1,p2,p3,b0,b1,pc0,pc1,pc2,pc3);
        bmma17(n0,n1,n2,n3,b0,b1,nc0,nc1,nc2,nc3);
        part0+=(pc0-nc0)*sc0+(pc1-nc1)*sc1;
        part8+=(pc2-nc2)*sc0+(pc3-nc3)*sc1;
    }

    part0+=__shfl_down_sync(0xffffffffu,part0,2,4);
    part8+=__shfl_down_sync(0xffffffffu,part8,2,4);
    part0+=__shfl_down_sync(0xffffffffu,part0,1,4);
    part8+=__shfl_down_sync(0xffffffffu,part8,1,4);

    __shared__ int partial[WARPS][16];
    if(tid4==0){partial[warp][group]=part0;partial[warp][group+8]=part8;}
    __syncthreads();
    if(threadIdx.x<16) {
        int sum=0;
#pragma unroll
        for(int w=0;w<WARPS;++w)sum+=partial[w][threadIdx.x];
        y[tile*16+threadIdx.x]=sum;
    }
}

struct TritFamily19 {
    const Family17& f;
    uint16_t* rom=nullptr;
    size_t count=0;
    explicit TritFamily19(const Family17& ff):f(ff){}
    void init(){
        const size_t lane_frags=(size_t)L17*f.tiles*f.chunks*32;
        count=(size_t)L17*f.tiles*f.chunks*CODES19*32;
        CUDA_CHECK(cudaMalloc(&rom,count*sizeof(uint16_t)));
        const int blocks=(int)std::min<size_t>(65535,(lane_frags+255)/256);
        pack_trits19<<<blocks,256>>>(f.pp,f.np,rom,lane_frags);
        CUDA_CHECK(cudaGetLastError());
    }
    const uint16_t* layer(int l)const{return rom+(size_t)l*f.tiles*f.chunks*CODES19*32;}
    size_t bytes_all()const{return count*sizeof(uint16_t);}
    ~TritFamily19(){cudaFree(rom);}
};

static void launch19(const TritFamily19& t,int l,const uint32_t* planes,
                     int32_t* out,int warps,cudaStream_t s){
    const Family17& f=t.f; const uint16_t* r=t.layer(l);
    switch(warps){
        case 1:bmma_tritrom19<1><<<f.tiles,32,0,s>>>(r,planes,out,f.tiles,f.chunks,f.words);break;
        case 2:bmma_tritrom19<2><<<f.tiles,64,0,s>>>(r,planes,out,f.tiles,f.chunks,f.words);break;
        case 4:bmma_tritrom19<4><<<f.tiles,128,0,s>>>(r,planes,out,f.tiles,f.chunks,f.words);break;
        case 8:bmma_tritrom19<8><<<f.tiles,256,0,s>>>(r,planes,out,f.tiles,f.chunks,f.words);break;
        default:throw std::runtime_error("V19 warps/tile must be 1,2,4,8");
    }
}

static cudaGraphExec_t capture_family19(const TritFamily19& t,const uint32_t* planes,
                                        int32_t* out,int warps,cudaStream_t s){
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    for(int l=0;l<L17;++l)launch19(t,l,planes,out,warps,s);
    CUDA_CHECK(cudaStreamEndCapture(s,&g));
    CUDA_CHECK(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
    CUDA_CHECK(cudaGraphDestroy(g));return e;
}

static cudaGraphExec_t capture_all19(const TritFamily19& q,const TritFamily19& o,
                                     const TritFamily19& g,const TritFamily19& d,
                                     const uint32_t* ph,const uint32_t* pi,
                                     int32_t* oq,int32_t* oo,int32_t* og,int32_t* od,
                                     const std::array<int,4>& w,cudaStream_t s){
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t gr=nullptr;cudaGraphExec_t e=nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    for(int l=0;l<L17;++l){
        launch19(q,l,ph,oq,w[0],s);launch19(o,l,ph,oo,w[1],s);
        launch19(g,l,ph,og,w[2],s);launch19(d,l,pi,od,w[3],s);
    }
    CUDA_CHECK(cudaStreamEndCapture(s,&gr));
    CUDA_CHECK(cudaGraphInstantiate(&e,gr,nullptr,nullptr,0));
    CUDA_CHECK(cudaGraphDestroy(gr));return e;
}

struct Choice19{int warps=1;Stats17 st;};
static Choice19 sweep19(const TritFamily19& t,const uint32_t* planes,int32_t* out,
                        cudaStream_t s,int rounds,int batch){
    Choice19 best;best.st.med=std::numeric_limits<float>::infinity();
    for(int w:std::array<int,4>{{1,2,4,8}}){
        auto e=capture_family19(t,planes,out,w,s);const Stats17 st=measure17(e,s,rounds,batch);
        CUDA_CHECK(cudaGraphExecDestroy(e));
        const double physical=(double)t.bytes_all()/(st.med*1e6);
        const double eq=(double)t.f.bytes_all()/(st.med*1e6);
        const double gmac=(double)t.f.M*t.f.K*L17/(st.med*1e6);
        std::cout<<std::fixed<<std::setprecision(4)<<"  trit-ROM warps/tile "<<w<<"  "<<st.med<<" ms/30L  "
                 <<std::setprecision(1)<<physical<<" physical GB/s  "<<eq<<" 2bit-eq GB/s  "<<gmac
                 <<" GMAC/s  spread="<<(100.0f*(st.max-st.min)/st.med)<<"%\n";
        if(st.med<best.st.med)best={w,st};
    }return best;
}

static bool check19(const TritFamily19& t,const uint32_t* planes,int32_t* ref,
                    int32_t* got,int warps,cudaStream_t s){
    for(int l=0;l<L17;++l){
        launch_popc17(t.f,l,planes,ref,128,s);launch19(t,l,planes,got,warps,s);
        CUDA_CHECK(cudaStreamSynchronize(s));int bad=-1;
        if(!equal_i32_17(ref,got,t.f.M,&bad)){
            std::cerr<<t.f.name<<" V19 correctness FAIL layer="<<l<<" row="<<bad<<"\n";return false;
        }
    }return true;
}

struct Options19{int rounds=9;int batch=100;};
static Options19 parse19(int argc,char**argv){
    Options19 o;for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--rounds"&&i+1<argc)o.rounds=std::stoi(argv[++i]);
        else if(a=="--batch"&&i+1<argc)o.batch=std::stoi(argv[++i]);
        else if(a=="-h"||a=="--help"){std::cout<<"GA102-ROM V19 exact 1.625-bit ternary ROM\n  --rounds N\n  --batch N\n";std::exit(0);}
        else throw std::runtime_error("Unknown or incomplete argument: "+a);
    }if(o.rounds<3||o.batch<1)throw std::runtime_error("rounds>=3 and batch>=1 required");return o;
}

static void print_func19(const char* n,const void* fn){cudaFuncAttributes a{};CUDA_CHECK(cudaFuncGetAttributes(&a,fn));
    std::cout<<n<<": regs/thread="<<a.numRegs<<" static-smem="<<a.sharedSizeBytes<<" local/thread="<<a.localSizeBytes<<"\n";}

int main(int argc,char**argv){
    try{
        const Options19 opt=parse19(argc,argv);int dev=0;CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
        if(prop.major!=8||prop.minor!=6||std::string(prop.name).find("RTX 3080")==std::string::npos){std::cerr<<"V19 restricted to RTX 3080 / SM86; found "<<prop.name<<"\n";return 3;}
        int memclk=0,bus=0;CUDA_CHECK(cudaDeviceGetAttribute(&memclk,cudaDevAttrMemoryClockRate,dev));CUDA_CHECK(cudaDeviceGetAttribute(&bus,cudaDevAttrGlobalMemoryBusWidth,dev));
        const double peak=2.0*(double)memclk*1000.0*((double)bus/8.0)/1e9;

        std::array<uint16_t,LUT19> lut{};for(int x=0;x<LUT19;++x){int z=x;unsigned pm=0,nm=0;for(int j=0;j<5;++j){int d=z%3;z/=3;if(d==1)pm|=1u<<j;else if(d==2)nm|=1u<<j;}lut[x]=(uint16_t)(pm|(nm<<5));}
        CUDA_CHECK(cudaMemcpyToSymbol(g_lut19,lut.data(),lut.size()*sizeof(uint16_t)));

        std::cout<<"GA102-ROM V19: exact near-entropy ternary ROM\nGPU               : "<<prop.name
                 <<"\nV18 measured      : 0.9799 ms/30L, 531.7 GB/s 2-bit weight stream"
                 <<"\nV19 physical pack : 1.625 bits/weight (26 B per 128-trit lane fragment)"
                 <<"\nproperty BW       : "<<std::fixed<<std::setprecision(1)<<peak<<" GB/s\n";
        std::cout<<"\n=== V19 kernel resources ===\n";print_func19("x1",(const void*)bmma_tritrom19<1>);print_func19("x2",(const void*)bmma_tritrom19<2>);print_func19("x4",(const void*)bmma_tritrom19<4>);print_func19("x8",(const void*)bmma_tritrom19<8>);

        Family17 qkv("QKV",QKV17,H17),o("O",H17,H17),gu("gate+up",GU17,H17),down("down",H17,I17);
        qkv.init(0x1234u);o.init(0x2345u);gu.init(0x3456u);down.init(0x4567u);
        TritFamily19 tq(qkv),to(o),tg(gu),td(down);tq.init();to.init();tg.init();td.init();CUDA_CHECK(cudaDeviceSynchronize());
        const size_t oldbytes=qkv.bytes_all()+o.bytes_all()+gu.bytes_all()+down.bytes_all();
        const size_t newbytes=tq.bytes_all()+to.bytes_all()+tg.bytes_all()+td.bytes_all();
        std::cout<<"\n=== V19 footprint ===\nold P/N 2-bit : "<<(double)oldbytes/1048576.0<<" MiB\ntrit ROM       : "<<(double)newbytes/1048576.0<<" MiB\nsaved          : "<<std::setprecision(2)<<(100.0*(1.0-(double)newbytes/oldbytes))<<"%\n";

        uint32_t *ph=nullptr,*pi=nullptr;CUDA_CHECK(cudaMalloc(&ph,(size_t)8*(H17/32)*sizeof(uint32_t)));CUDA_CHECK(cudaMalloc(&pi,(size_t)8*(I17/32)*sizeof(uint32_t)));
        init_u32_17<<<32,256>>>(ph,(size_t)8*(H17/32),0x11112222u);init_u32_17<<<32,256>>>(pi,(size_t)8*(I17/32),0x33334444u);
        int32_t *ref=nullptr,*got=nullptr,*oq=nullptr,*oo=nullptr,*og=nullptr,*od=nullptr;CUDA_CHECK(cudaMalloc(&ref,GU17*sizeof(int32_t)));CUDA_CHECK(cudaMalloc(&got,GU17*sizeof(int32_t)));CUDA_CHECK(cudaMalloc(&oq,QKV17*sizeof(int32_t)));CUDA_CHECK(cudaMalloc(&oo,H17*sizeof(int32_t)));CUDA_CHECK(cudaMalloc(&og,GU17*sizeof(int32_t)));CUDA_CHECK(cudaMalloc(&od,H17*sizeof(int32_t)));
        cudaStream_t s=nullptr;CUDA_CHECK(cudaStreamCreate(&s));

        std::cout<<"\n=== V19 trit-ROM split-K sweep ===\n\nQKV\n";const Choice19 cq=sweep19(tq,ph,oq,s,opt.rounds,opt.batch);
        std::cout<<"\nO\n";const Choice19 co=sweep19(to,ph,oo,s,opt.rounds,opt.batch);
        std::cout<<"\ngate+up\n";const Choice19 cg=sweep19(tg,ph,og,s,opt.rounds,opt.batch);
        std::cout<<"\ndown\n";const Choice19 cd=sweep19(td,pi,od,s,opt.rounds,opt.batch);
        const std::array<int,4> w{{cq.warps,co.warps,cg.warps,cd.warps}};

        std::cout<<"\n=== V19 exactness, selected mappings, all 30 layers ===\n";bool ok=true,x;
        x=check19(tq,ph,ref,got,w[0],s);ok&=x;std::cout<<"QKV       : "<<(x?"PASS":"FAIL")<<"\n";
        x=check19(to,ph,ref,got,w[1],s);ok&=x;std::cout<<"O         : "<<(x?"PASS":"FAIL")<<"\n";
        x=check19(tg,ph,ref,got,w[2],s);ok&=x;std::cout<<"gate+up   : "<<(x?"PASS":"FAIL")<<"\n";
        x=check19(td,pi,ref,got,w[3],s);ok&=x;std::cout<<"down      : "<<(x?"PASS":"FAIL")<<"\n";
        if(!ok){std::cerr<<"V19 stopped: compressed decode is not exact.\n";return 4;}

        auto e=capture_all19(tq,to,tg,td,ph,pi,oq,oo,og,od,w,s);const Stats17 st=measure17(e,s,opt.rounds,opt.batch);CUDA_CHECK(cudaGraphExecDestroy(e));
        const double macs=2084044800.0;const double physical=(double)newbytes/(st.med*1e6);const double eq=(double)oldbytes/(st.med*1e6);const double gmac=macs/(st.med*1e6);
        const double v18=0.9799;const double ideal=v18*((double)newbytes/oldbytes);
        std::cout<<"\n=== V19 complete 30-layer linear chain ===\n"<<std::fixed<<std::setprecision(4)
                 <<"V18 measured       : "<<v18<<" ms\nV19 trit ROM       : "<<st.med<<" ms  "<<std::setprecision(1)<<physical<<" physical GB/s  "<<eq<<" 2bit-eq GB/s  "<<gmac<<" GMAC/s\n"
                 <<"speedup vs V18     : "<<std::setprecision(3)<<(v18/st.med)<<"x\n"
                 <<"decode-free ideal  : "<<std::setprecision(4)<<ideal<<" ms (same physical BW as V18)\n"
                 <<"physical BW / peak : "<<std::setprecision(1)<<(100.0*physical/peak)<<"%\n"
                 <<"selected warps     : QKV="<<w[0]<<" O="<<w[1]<<" gate+up="<<w[2]<<" down="<<w[3]<<"\n";
        std::cout<<"\nGuardrails:\n  * PASS = exact int32 equality against POPC for every matrix in all 30 layers.\n  * 1.625 bits/weight is physical storage; log2(3)=1.585 is the information floor.\n  * Offline packing is excluded because the fixed checkpoint is compiled once.\n  * LUT/decode cost is fully inside the timed V19 kernel.\n  * 2bit-eq GB/s is only a comparison metric; physical GB/s uses actual compressed bytes.\n";

        CUDA_CHECK(cudaStreamDestroy(s));cudaFree(ph);cudaFree(pi);cudaFree(ref);cudaFree(got);cudaFree(oq);cudaFree(oo);cudaFree(og);cudaFree(od);
        std::cout<<"\nV19 completed.\n";return 0;
    }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
