#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
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

static constexpr int HIDDEN = 2560;
static constexpr int INTER = 6912;
static constexpr int Q_HEADS = 20;
static constexpr int KV_HEADS = 5;
static constexpr int HEAD_DIM = 128;
static constexpr int Q_DIM = Q_HEADS * HEAD_DIM;
static constexpr int KV_DIM = KV_HEADS * HEAD_DIM;
static constexpr int QKV_DIM = Q_DIM + 2 * KV_DIM;
static constexpr int GATEUP_DIM = 2 * INTER;
static constexpr int LAYERS = 30;
static constexpr float V6_LINEAR_MS = 2.1265f;
static constexpr float V7_LM_HEAD_MS = 0.9166f;

struct Options { int iters = 300; int max_context = 4096; };

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) o.iters = std::stoi(argv[++i]);
        else if (a == "--max-context" && i + 1 < argc) o.max_context = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "GA102-ROM V8 fused decoder support + exact decode attention\n"
                      << "  --iters N        timing iterations (default 300)\n"
                      << "  --max-context N  max KV context (default 4096)\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown or incomplete argument: " + a);
    }
    if (o.iters <= 0 || o.max_context < 128 || o.max_context > 8192)
        throw std::runtime_error("iters must be positive; max-context must be 128..8192");
    return o;
}

static float time_ms(const std::function<void()>& launch, int warmup, int iters) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float total = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&total, a, b));
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    return total / iters;
}

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 b16(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffffu,v,off);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off=16; off>0; off>>=1) v = fmaxf(v,__shfl_down_sync(0xffffffffu,v,off));
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* s) {
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    v=warp_sum(v); if(lane==0)s[warp]=v; __syncthreads();
    float x=(threadIdx.x<(blockDim.x>>5))?s[lane]:0.0f;
    if(warp==0)x=warp_sum(x); __syncthreads();
    if(threadIdx.x==0)s[0]=x; __syncthreads(); return s[0];
}
__device__ __forceinline__ float block_max(float v, float* s) {
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    v=warp_max(v); if(lane==0)s[warp]=v; __syncthreads();
    float x=(threadIdx.x<(blockDim.x>>5))?s[lane]:-CUDART_INF_F;
    if(warp==0)x=warp_max(x); __syncthreads();
    if(threadIdx.x==0)s[0]=x; __syncthreads(); return s[0];
}

// RMSNorm + exact BitNet symmetric per-token A8 activation quantization.
__global__ void rmsnorm_quant_a8(const __nv_bfloat16* x, const __nv_bfloat16* gamma,
                                 int8_t* q, float* qscale, int n, float eps) {
    __shared__ float s[8], invr, scale;
    float ss=0.0f;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){float v=bf(x[i]);ss+=v*v;}
    float sum=block_sum(ss,s); if(threadIdx.x==0)invr=rsqrtf(sum/(float)n+eps); __syncthreads();
    float lm=0.0f;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)lm=fmaxf(lm,fabsf(bf(x[i])*invr*bf(gamma[i])));
    float mx=block_max(lm,s);
    if(threadIdx.x==0){scale=127.0f/fmaxf(mx,1e-5f);qscale[0]=scale;} __syncthreads();
    for(int i=threadIdx.x;i<n;i+=blockDim.x){
        int z=__float2int_rn(bf(x[i])*invr*bf(gamma[i])*scale); z=max(-128,min(127,z)); q[i]=(int8_t)z;
    }
}

__global__ void rmsnorm_bf16(const __nv_bfloat16* x,const __nv_bfloat16* gamma,
                             __nv_bfloat16* y,int n,float eps){
    __shared__ float s[8],invr; float ss=0.0f;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){float v=bf(x[i]);ss+=v*v;}
    float sum=block_sum(ss,s); if(threadIdx.x==0)invr=rsqrtf(sum/(float)n+eps); __syncthreads();
    for(int i=threadIdx.x;i<n;i+=blockDim.x)y[i]=b16(bf(x[i])*invr*bf(gamma[i]));
}

__global__ void qkv_epilogue_rope_kv(const int32_t* qkv,__nv_bfloat16* qout,
                                     __nv_bfloat16* kc,__nv_bfloat16* vc,
                                     const float* cs,const float* sn,int pos,
                                     float qs,float ks,float vs){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=QKV_DIM)return;
    if(i<Q_DIM){
        int d=i&127,base=i-d,od=d<64?d+64:d-64; float x=(float)qkv[i]*qs,xo=(float)qkv[base+od]*qs;
        qout[i]=b16(x*cs[d]+(d<64?-xo:xo)*sn[d]);
    }else if(i<Q_DIM+KV_DIM){
        int j=i-Q_DIM,d=j&127,base=i-d,od=d<64?d+64:d-64; float x=(float)qkv[i]*ks,xo=(float)qkv[base+od]*ks;
        kc[(size_t)pos*KV_DIM+j]=b16(x*cs[d]+(d<64?-xo:xo)*sn[d]);
    }else{
        int j=i-Q_DIM-KV_DIM; vc[(size_t)pos*KV_DIM+j]=b16((float)qkv[i]*vs);
    }
}

// 20 blocks: one block per Q head. Higher parallelism, repeated GQA KV traversal.
__global__ void attention_qhead(const __nv_bfloat16* q,const __nv_bfloat16* kc,
                                const __nv_bfloat16* vc,__nv_bfloat16* out,int L){
    extern __shared__ float score[]; __shared__ float s[8],invsum;
    int qh=blockIdx.x,kvh=qh>>2,warp=threadIdx.x>>5,lane=threadIdx.x&31,nw=blockDim.x>>5;
    constexpr float sc=0.08838834764831845f;
    for(int p=warp;p<L;p+=nw){float a=0.0f;for(int d=lane;d<128;d+=32)a+=bf(q[qh*128+d])*bf(kc[(size_t)p*KV_DIM+kvh*128+d]);a=warp_sum(a);if(lane==0)score[p]=a*sc;}
    __syncthreads();
    float lm=-CUDART_INF_F;for(int p=threadIdx.x;p<L;p+=blockDim.x)lm=fmaxf(lm,score[p]);float mx=block_max(lm,s);
    float ls=0.0f;for(int p=threadIdx.x;p<L;p+=blockDim.x){float e=expf(score[p]-mx);score[p]=e;ls+=e;}float sm=block_sum(ls,s);
    if(threadIdx.x==0)invsum=1.0f/sm;__syncthreads();
    if(threadIdx.x<128){int d=threadIdx.x;float a=0.0f;for(int p=0;p<L;++p)a+=score[p]*invsum*bf(vc[(size_t)p*KV_DIM+kvh*128+d]);out[qh*128+d]=b16(a);}
}

// 5 blocks: one per KV head, all four associated Q heads together. K/V each loaded once per group.
__global__ void attention_gqa4(const __nv_bfloat16* q,const __nv_bfloat16* kc,
                               const __nv_bfloat16* vc,__nv_bfloat16* out,int L){
    extern __shared__ float score[]; __shared__ float s[8],inv[4];
    int kvh=blockIdx.x,qb=kvh*4,warp=threadIdx.x>>5,lane=threadIdx.x&31,nw=blockDim.x>>5;
    constexpr float sc=0.08838834764831845f;
    for(int p=warp;p<L;p+=nw){
        float a0=0,a1=0,a2=0,a3=0;
        for(int d=lane;d<128;d+=32){float k=bf(kc[(size_t)p*KV_DIM+kvh*128+d]);a0+=bf(q[(qb+0)*128+d])*k;a1+=bf(q[(qb+1)*128+d])*k;a2+=bf(q[(qb+2)*128+d])*k;a3+=bf(q[(qb+3)*128+d])*k;}
        a0=warp_sum(a0);a1=warp_sum(a1);a2=warp_sum(a2);a3=warp_sum(a3);
        if(lane==0){score[p]=a0*sc;score[L+p]=a1*sc;score[2*L+p]=a2*sc;score[3*L+p]=a3*sc;}
    }__syncthreads();
    for(int h=0;h<4;++h){float* z=score+(size_t)h*L;float lm=-CUDART_INF_F;for(int p=threadIdx.x;p<L;p+=blockDim.x)lm=fmaxf(lm,z[p]);float mx=block_max(lm,s);float ls=0;for(int p=threadIdx.x;p<L;p+=blockDim.x){float e=expf(z[p]-mx);z[p]=e;ls+=e;}float sm=block_sum(ls,s);if(threadIdx.x==0)inv[h]=1.0f/sm;__syncthreads();}
    if(threadIdx.x<128){int d=threadIdx.x;float a0=0,a1=0,a2=0,a3=0;for(int p=0;p<L;++p){float v=bf(vc[(size_t)p*KV_DIM+kvh*128+d]);a0+=score[p]*inv[0]*v;a1+=score[L+p]*inv[1]*v;a2+=score[2*L+p]*inv[2]*v;a3+=score[3*L+p]*inv[3]*v;}out[(qb+0)*128+d]=b16(a0);out[(qb+1)*128+d]=b16(a1);out[(qb+2)*128+d]=b16(a2);out[(qb+3)*128+d]=b16(a3);}
}

__global__ void o_resid_norm_quant(const int32_t* oacc,const __nv_bfloat16* residual,
                                   const __nv_bfloat16* gamma,__nv_bfloat16* mid,
                                   int8_t* q,float* qscale,float os,float eps){
    __shared__ float s[8],invr,scale;float ss=0;
    for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x){float v=bf(residual[i])+(float)oacc[i]*os;mid[i]=b16(v);ss+=v*v;}
    float sum=block_sum(ss,s);if(threadIdx.x==0)invr=rsqrtf(sum/(float)HIDDEN+eps);__syncthreads();
    float lm=0;for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x)lm=fmaxf(lm,fabsf(bf(mid[i])*invr*bf(gamma[i])));float mx=block_max(lm,s);
    if(threadIdx.x==0){scale=127.0f/fmaxf(mx,1e-5f);qscale[0]=scale;}__syncthreads();
    for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x){int z=__float2int_rn(bf(mid[i])*invr*bf(gamma[i])*scale);z=max(-128,min(127,z));q[i]=(int8_t)z;}
}

__global__ void gateup_relu2_norm_quant(const int32_t* gu,const __nv_bfloat16* gamma,
                                        __nv_bfloat16* tmp,int8_t* q,float* qscale,
                                        float gs,float us,float eps){
    __shared__ float s[8],invr,scale;float ss=0;
    for(int i=threadIdx.x;i<INTER;i+=blockDim.x){float g=(float)gu[i]*gs,u=(float)gu[INTER+i]*us,r=fmaxf(g,0.0f),z=r*r*u;tmp[i]=b16(z);ss+=z*z;}
    float sum=block_sum(ss,s);if(threadIdx.x==0)invr=rsqrtf(sum/(float)INTER+eps);__syncthreads();
    float lm=0;for(int i=threadIdx.x;i<INTER;i+=blockDim.x)lm=fmaxf(lm,fabsf(bf(tmp[i])*invr*bf(gamma[i])));float mx=block_max(lm,s);
    if(threadIdx.x==0){scale=127.0f/fmaxf(mx,1e-5f);qscale[0]=scale;}__syncthreads();
    for(int i=threadIdx.x;i<INTER;i+=blockDim.x){int z=__float2int_rn(bf(tmp[i])*invr*bf(gamma[i])*scale);z=max(-128,min(127,z));q[i]=(int8_t)z;}
}

__global__ void down_resid(const int32_t* down,const __nv_bfloat16* residual,__nv_bfloat16* out,float sc){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<HIDDEN)out[i]=b16(bf(residual[i])+(float)down[i]*sc);}

static void cpu_attn0(const std::vector<__nv_bfloat16>& q,const std::vector<__nv_bfloat16>& k,
                      const std::vector<__nv_bfloat16>& v,std::vector<float>& out,int L){
    std::vector<float>s(L);float mx=-std::numeric_limits<float>::infinity();constexpr float sc=0.08838834764831845f;
    for(int p=0;p<L;++p){float a=0;for(int d=0;d<128;++d)a+=__bfloat162float(q[d])*__bfloat162float(k[(size_t)p*KV_DIM+d]);s[p]=a*sc;mx=std::max(mx,s[p]);}
    float sm=0;for(float&x:s){x=std::exp(x-mx);sm+=x;}for(float&x:s)x/=sm;out.assign(128,0);
    for(int d=0;d<128;++d){float a=0;for(int p=0;p<L;++p)a+=s[p]*__bfloat162float(v[(size_t)p*KV_DIM+d]);out[d]=a;}
}

int main(int argc,char**argv){
    try{
        Options o=parse_args(argc,argv);int dev=0;CUDA_CHECK(cudaGetDevice(&dev));cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
        if(prop.major!=8||prop.minor!=6||std::string(prop.name).find("RTX 3080")==std::string::npos){std::cerr<<"V8 restricted to RTX 3080 / SM86; found "<<prop.name<<"\n";return 3;}
        std::cout<<"GA102-ROM V8: fused support kernels + exact GQA decode attention\nGPU               : "<<prop.name
                 <<"\nmodel             : microsoft/bitnet-b1.58-2B-4T shapes\nheads             : Q=20 KV=5 dim=128\nV6 linears        : "<<V6_LINEAR_MS
                 <<" ms/token (30 layers)\nV7 BF16 LM head   : "<<V7_LM_HEAD_MS<<" ms/token\n";

        std::vector<__nv_bfloat16> hh(HIDDEN),gh(HIDDEN),gi(INTER),hq(Q_DIM),hk((size_t)o.max_context*KV_DIM),hv((size_t)o.max_context*KV_DIM);
        for(int i=0;i<HIDDEN;++i){hh[i]=__float2bfloat16((float)((i%97)-48)/64.0f);gh[i]=__float2bfloat16(1.0f+0.001f*(i%11));}
        for(int i=0;i<INTER;++i)gi[i]=__float2bfloat16(1.0f+0.001f*(i%13));
        for(int h=0;h<Q_HEADS;++h)for(int d=0;d<128;++d)hq[h*128+d]=__float2bfloat16((float)(((h*131+d*17)%101)-50)/80.0f);
        for(int p=0;p<o.max_context;++p)for(int h=0;h<KV_HEADS;++h)for(int d=0;d<128;++d){int z=(p*29+h*37+d*11)%127,w=(p*17+h*19+d*23)%113;hk[(size_t)p*KV_DIM+h*128+d]=__float2bfloat16((float)(z-63)/96.0f);hv[(size_t)p*KV_DIM+h*128+d]=__float2bfloat16((float)(w-56)/88.0f);}

        __nv_bfloat16 *dh=nullptr,*dgh=nullptr,*dgi=nullptr,*dq=nullptr,*dk=nullptr,*dv=nullptr,*da=nullptr,*da2=nullptr,*dmid=nullptr,*dtmp=nullptr,*df=nullptr,*dfn=nullptr;
        int8_t *dqh=nullptr,*dqi=nullptr;float *dsh=nullptr,*dsi=nullptr,*dcs=nullptr,*dsn=nullptr;int32_t *dqkv=nullptr,*doo=nullptr,*dgu=nullptr,*ddown=nullptr;
        CUDA_CHECK(cudaMalloc(&dh,HIDDEN*2));CUDA_CHECK(cudaMalloc(&dgh,HIDDEN*2));CUDA_CHECK(cudaMalloc(&dgi,INTER*2));CUDA_CHECK(cudaMalloc(&dq,Q_DIM*2));CUDA_CHECK(cudaMalloc(&dk,(size_t)o.max_context*KV_DIM*2));CUDA_CHECK(cudaMalloc(&dv,(size_t)o.max_context*KV_DIM*2));CUDA_CHECK(cudaMalloc(&da,Q_DIM*2));CUDA_CHECK(cudaMalloc(&da2,Q_DIM*2));CUDA_CHECK(cudaMalloc(&dmid,HIDDEN*2));CUDA_CHECK(cudaMalloc(&dtmp,INTER*2));CUDA_CHECK(cudaMalloc(&df,HIDDEN*2));CUDA_CHECK(cudaMalloc(&dfn,HIDDEN*2));
        CUDA_CHECK(cudaMalloc(&dqh,HIDDEN));CUDA_CHECK(cudaMalloc(&dqi,INTER));CUDA_CHECK(cudaMalloc(&dsh,sizeof(float)));CUDA_CHECK(cudaMalloc(&dsi,sizeof(float)));CUDA_CHECK(cudaMalloc(&dqkv,QKV_DIM*4));CUDA_CHECK(cudaMalloc(&doo,HIDDEN*4));CUDA_CHECK(cudaMalloc(&dgu,GATEUP_DIM*4));CUDA_CHECK(cudaMalloc(&ddown,HIDDEN*4));CUDA_CHECK(cudaMalloc(&dcs,HEAD_DIM*4));CUDA_CHECK(cudaMalloc(&dsn,HEAD_DIM*4));
        CUDA_CHECK(cudaMemcpy(dh,hh.data(),HIDDEN*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dgh,gh.data(),HIDDEN*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dgi,gi.data(),INTER*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dq,hq.data(),Q_DIM*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dk,hk.data(),hk.size()*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dv,hv.data(),hv.size()*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(da,hq.data(),Q_DIM*2,cudaMemcpyHostToDevice));

        std::vector<int32_t> qkv(QKV_DIM),oo(HIDDEN),gu(GATEUP_DIM),down(HIDDEN);for(int i=0;i<QKV_DIM;++i)qkv[i]=(i%257)-128;for(int i=0;i<HIDDEN;++i){oo[i]=(i%193)-96;down[i]=(i%181)-90;}for(int i=0;i<GATEUP_DIM;++i)gu[i]=(i%149)-74;
        CUDA_CHECK(cudaMemcpy(dqkv,qkv.data(),qkv.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(doo,oo.data(),oo.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dgu,gu.data(),gu.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(ddown,down.data(),down.size()*4,cudaMemcpyHostToDevice));
        std::vector<float>cs(128),sn(128);for(int d=0;d<128;++d){float a=0.0007f*(d+1);cs[d]=std::cos(a);sn[d]=std::sin(a);}CUDA_CHECK(cudaMemcpy(dcs,cs.data(),512,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dsn,sn.data(),512,cudaMemcpyHostToDevice));

        int t=256;
        auto kin=[&]{rmsnorm_quant_a8<<<1,t>>>(dh,dgh,dqh,dsh,HIDDEN,1e-6f);};
        auto kqkv=[&]{qkv_epilogue_rope_kv<<<(QKV_DIM+t-1)/t,t>>>(dqkv,dq,dk,dv,dcs,dsn,0,0.001f,0.001f,0.001f);};
        auto kattnq=[&]{rmsnorm_quant_a8<<<1,t>>>(da,dgh,dqh,dsh,HIDDEN,1e-6f);};
        auto ko=[&]{o_resid_norm_quant<<<1,t>>>(doo,dh,dgh,dmid,dqh,dsh,0.001f,1e-6f);};
        auto kgu=[&]{gateup_relu2_norm_quant<<<1,t>>>(dgu,dgi,dtmp,dqi,dsi,0.0005f,0.0005f,1e-6f);};
        auto kd=[&]{down_resid<<<(HIDDEN+t-1)/t,t>>>(ddown,dmid,df,0.001f);};
        auto kfn=[&]{rmsnorm_bf16<<<1,t>>>(df,dgh,dfn,HIDDEN,1e-6f);};
        float a=time_ms(kin,30,o.iters),b=time_ms(kqkv,30,o.iters),c=time_ms(kattnq,30,o.iters),d=time_ms(ko,30,o.iters),e=time_ms(kgu,30,o.iters),f=time_ms(kd,30,o.iters),fn=time_ms(kfn,30,o.iters),support=a+b+c+d+e+f;
        std::cout<<"\n=== V8 fused per-layer support ===\n"<<std::fixed<<std::setprecision(4)
                 <<"input RMSNorm + A8 quant          : "<<a<<" ms\nQKV postscale + RoPE + KV append  : "<<b<<" ms\nattn subnorm + O-input A8 quant   : "<<c<<" ms\nO postscale+resid+norm+gate A8    : "<<d<<" ms\ngate/up+ReLU2+FFN norm+down A8    : "<<e<<" ms\ndown postscale + residual         : "<<f<<" ms\nfused support / layer             : "<<support<<" ms\nfused support / 30L               : "<<support*30<<" ms\nV7 launch-separated aux / 30L     : 3.5100 ms\nsupport fusion speedup             : "<<(3.5100f/(support*30))<<"x\nfinal RMSNorm (once/token)         : "<<fn<<" ms\n";

        // Restore pristine attention inputs after QKV epilogue timing mutated Q and KV position 0.
        CUDA_CHECK(cudaMemcpy(dq,hq.data(),Q_DIM*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dk,hk.data(),hk.size()*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dv,hv.data(),hv.size()*2,cudaMemcpyHostToDevice));

        attention_qhead<<<Q_HEADS,256,128*sizeof(float)>>>(dq,dk,dv,da,128);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16>got(Q_DIM);CUDA_CHECK(cudaMemcpy(got.data(),da,Q_DIM*2,cudaMemcpyDeviceToHost));std::vector<float>ref;cpu_attn0(hq,hk,hv,ref,128);float err=0;for(int i=0;i<128;++i)err=std::max(err,std::fabs(ref[i]-__bfloat162float(got[i])));bool ok=err<0.03f;
        std::cout<<"\n=== V8 exact GQA decode attention ===\ncorrectness head0@128 : "<<(ok?"PASS":"FAIL")<<"  max_abs_err="<<std::setprecision(6)<<err<<"\n";if(!ok)return 5;

        std::array<int,5>ctxs{{128,512,1024,2048,4096}};
        for(int L:ctxs){if(L>o.max_context)continue;size_t sq=(size_t)L*4,sg=(size_t)4*L*4;CUDA_CHECK(cudaFuncSetAttribute(attention_gqa4,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sg));
            auto qhcall=[&]{attention_qhead<<<Q_HEADS,256,sq>>>(dq,dk,dv,da,L);};auto g4call=[&]{attention_gqa4<<<KV_HEADS,256,sg>>>(dq,dk,dv,da2,L);};float tq=time_ms(qhcall,20,o.iters),tg=time_ms(g4call,20,o.iters),best=std::min(tq,tg);
            qhcall();g4call();CUDA_CHECK(cudaDeviceSynchronize());std::vector<__nv_bfloat16>x(Q_DIM),y(Q_DIM);CUDA_CHECK(cudaMemcpy(x.data(),da,Q_DIM*2,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(y.data(),da2,Q_DIM*2,cudaMemcpyDeviceToHost));float diff=0;for(int i=0;i<Q_DIM;++i)diff=std::max(diff,std::fabs(__bfloat162float(x[i])-__bfloat162float(y[i])));
            double minbytes=(double)L*KV_DIM*4.0;float acct=V6_LINEAR_MS+support*30+best*30+fn+V7_LM_HEAD_MS;
            std::cout<<std::fixed<<std::setprecision(4)<<"ctx "<<std::setw(4)<<L<<"  qhead="<<tq<<" ms  gqa4-reuse="<<tg<<" ms  best="<<best<<" ms/layer  best-30L="<<best*30<<" ms  min-KV-equiv="<<std::setprecision(1)<<(minbytes/(best*1e6))<<" GB/s  cross-kernel-diff="<<std::setprecision(5)<<diff<<"\n"
                     <<std::setprecision(4)<<"          GPU decode accounting: "<<acct<<" ms => "<<std::setprecision(1)<<(1000.0f/acct)<<" tok/s ceiling\n";
        }
        std::cout<<"\nGuardrails:\n  * attention performs QK scaling, softmax and weighted V; KV reads are included.\n  * min-KV-equiv uses one physical K+V pass as a lower-bound byte count; qhead may reread a GQA KV group.\n  * accounting combines V6 linears, fused support, measured attention, final norm and V7 LM head.\n  * still a GPU-kernel accounting ceiling, not tokenizer/sampling/host end-to-end generation.\n  * epilogues use representative scales; runtime cost is measured, not checkpoint-level numerical equivalence.\n";

        cudaFree(dh);cudaFree(dgh);cudaFree(dgi);cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(da);cudaFree(da2);cudaFree(dmid);cudaFree(dtmp);cudaFree(df);cudaFree(dfn);cudaFree(dqh);cudaFree(dqi);cudaFree(dsh);cudaFree(dsi);cudaFree(dqkv);cudaFree(doo);cudaFree(dgu);cudaFree(ddown);cudaFree(dcs);cudaFree(dsn);
        std::cout<<"\nV8 completed.\n";return 0;
    }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
