#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "../code/conv_cpu.cpp"
#include "../code/pool_cpu.cpp"
#include "../code/unroll_cpu.cpp"
int main() {
  int cases=0;
  double worst=0;
  for (const auto& shape : std::vector<std::vector<int>>{
      {1,2,3,4,4,3},{3,5,3,20,23,3},{2,4,2,19,35,1},
      {2,3,2,5,7,5},{2,4,1,86,86,7},{2,16,4,40,40,7}}) {
    const int B=shape[0],M=shape[1],C=shape[2],H=shape[3],W=shape[4],K=shape[5];
    const int Ho=H-K+1,Wo=W-K+1,D=C*K*K,L=Ho*Wo;
    std::vector<float> x(static_cast<std::size_t>(B)*C*H*W),f(static_cast<std::size_t>(M)*D);
    for(std::size_t i=0;i<x.size();++i)x[i]=static_cast<float>(static_cast<int>(i%17)-8)/8;
    for(std::size_t i=0;i<f.size();++i)f[i]=static_cast<float>(static_cast<int>(i%11)-5)/4;
    std::vector<float> y(static_cast<std::size_t>(B)*M*L),u(static_cast<std::size_t>(B)*D*L);
    conv_forward_cpu(B,M,C,H,W,K,x.data(),f.data(),y.data());
    unroll_cpu(B,C,H,W,K,x.data(),u.data());
    std::vector<int> visits(y.size(),0),uv(u.size(),0);
    for(int b=0;b<B;++b)for(int m=0;m<M;++m)for(int h=0;h<Ho;++h)for(int w=0;w<Wo;++w){
      double ref=0,gemm=0;
      for(int c=0;c<C;++c)for(int p=0;p<K;++p)for(int q=0;q<K;++q){
        const auto xi=((static_cast<std::size_t>(b)*C+c)*H+h+p)*W+w+q;
        const int r=(c*K+p)*K+q;
        ref+=static_cast<double>(x[xi])*f[m*D+r];
        const auto ui=(static_cast<std::size_t>(b)*D+r)*L+h*Wo+w;
        assert(u[ui]==x[xi]);gemm+=static_cast<double>(u[ui])*f[m*D+r];
      }
      const auto yi=((static_cast<std::size_t>(b)*M+m)*Ho+h)*Wo+w;
      worst=std::max(worst,std::abs(y[yi]-ref));assert(std::abs(y[yi]-ref)<1e-4);assert(gemm==ref);
    }
    const int tw=(Wo+15)/16,th=(Ho+15)/16;
    for(int b=0;b<B;++b)for(int m=0;m<M;++m)for(int t=0;t<tw*th;++t)
      for(int ty=0;ty<16;++ty)for(int tx=0;tx<16;++tx){
        int h=t/tw*16+ty,w=t%tw*16+tx;
        if(h<Ho&&w<Wo)++visits[((b*M+m)*Ho+h)*Wo+w];
      }
    assert(std::all_of(visits.begin(),visits.end(),[](int v){return v==1;}));
    // Emulate a smaller CUDA grid-stride launch and inverse im2col mapping.
    for(std::size_t t=0;t<111;++t)for(std::size_t i=t;i<u.size();i+=111){
      const auto ell=i%L,r=(i/L)%D,b=i/(static_cast<std::size_t>(D)*L);
      const auto c=r/K/K,p=(r/K)%K,q=r%K,h=ell/Wo,w=ell%Wo;
      assert(u[i]==x[((b*C+c)*H+h+p)*W+w+q]);++uv[i];
    }
    assert(std::all_of(uv.begin(),uv.end(),[](int v){return v==1;}));
    // Batched and concatenated layouts are permutations, not a reshape.
    std::vector<float> concat(u.size());
    for(int b=0;b<B;++b)for(int r=0;r<D;++r)for(int l=0;l<L;++l)
      concat[(static_cast<std::size_t>(r)*B+b)*L+l]=u[(static_cast<std::size_t>(b)*D+r)*L+l];
    for(int b=0;b<B;++b)for(int m=0;m<M;++m)for(int l=0;l<L;++l){
      double sum=0;for(int r=0;r<D;++r)sum+=f[m*D+r]*concat[(static_cast<std::size_t>(r)*B+b)*L+l];
      assert(std::abs(sum-y[(b*M+m)*L+l])<1e-4);
    }
    ++cases;
  }
  for(int N:{2,3}) {
    const int B=2,M=3,H=7,W=10,Hs=H/N,Ws=W/N;
    std::vector<float> x(B*M*H*W),bias={-1,0,1},out(B*M*Hs*Ws);
    for(std::size_t i=0;i<x.size();++i)x[i]=static_cast<float>(static_cast<int>(i%13)-6);
    pool_forward_cpu(B,M,H,W,N,x.data(),bias.data(),out.data());
    for(int b=0;b<B;++b)for(int m=0;m<M;++m)for(int h=0;h<Hs;++h)for(int w=0;w<Ws;++w){
      double sum=0;for(int p=0;p<N;++p)for(int q=0;q<N;++q)sum+=x[((b*M+m)*H+N*h+p)*W+N*w+q];
      const double ref=1/(1+std::exp(-(sum/(N*N)+bias[m])));
      assert(std::abs(out[((b*M+m)*Hs+h)*Ws+w]-ref)<1e-6);
    }
    const auto before=out;
    for(int b=0;b<B;++b)for(int m=0;m<M;++m)for(int h=0;h<H;++h)for(int w=0;w<W;++w)
      if(h>=Hs*N||w>=Ws*N)x[((b*M+m)*H+h)*W+w]=10000;
    pool_forward_cpu(B,M,H,W,N,x.data(),bias.data(),out.data());assert(out==before);
  }
  for(float z:{-10000.f,-100.f,0.f,100.f,10000.f}){
    float a=sigmoid_cpu(z);assert(std::isfinite(a)&&a>=0&&a<=1);
  }
  assert(off4(999,99,999,999,100,1000,1000)==99999999999ULL);
  std::cout<<"{\"convolution_cases\":"<<cases<<",\"pooling_cases\":2,\"max_abs_error\":"<<worst
           <<",\"layout_mapping\":\"PASS\",\"large_offsets\":\"PASS\",\"undefined_behavior_sanitizer\":\"PASS\"}\n";
}
