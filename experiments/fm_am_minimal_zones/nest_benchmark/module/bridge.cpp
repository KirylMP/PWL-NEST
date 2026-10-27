#include "bridge.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <utility>
extern "C" int pwla_simulate(const double*,int,const double*,const double*,double,double,double,double,double,const double*,const double*,int,double*,int,int*);
namespace {
using V=std::array<double,4>; using M=std::array<std::array<double,4>,4>;
M identity(){M a{};for(int i=0;i<4;++i)a[i][i]=1.;return a;}
M multiply(const M&a,const M&b){M c{};for(int i=0;i<4;++i)for(int k=0;k<4;++k)for(int j=0;j<4;++j)c[i][j]+=a[i][k]*b[k][j];return c;}
V multiply(const M&a,const V&x){V y{};for(int i=0;i<4;++i)for(int j=0;j<4;++j)y[i]+=a[i][j]*x[j];return y;}
M exponential(M a,double t){
 double norm=0.;for(int i=0;i<4;++i){double sum=0.;for(int j=0;j<4;++j)sum+=std::abs(a[i][j]*t);norm=std::max(norm,sum);}
 int scale=norm>.5?std::max(0,(int)std::ceil(std::log2(norm/.5))):0;double factor=t/std::ldexp(1.,scale);
 for(auto&row:a)for(double&x:row)x*=factor;
 M sum=identity(),term=identity();
 for(int k=1;k<=32;++k){term=multiply(term,a);double largest=0.;for(int i=0;i<4;++i)for(int j=0;j<4;++j){term[i][j]/=k;sum[i][j]+=term[i][j];largest=std::max(largest,std::abs(term[i][j]));}if(largest<2e-17)break;}
 for(int i=0;i<scale;++i)sum=multiply(sum,sum);
 return sum;
}
struct Cache {const PwaConfig* cfg;double baseline,tau,dt;std::vector<M> flow;};
const Cache& get_cache(const PwaConfig&m,double baseline,double tau,double dt){
 static thread_local std::vector<Cache> caches;
 for(const auto& c:caches)if(c.cfg==&m&&c.baseline==baseline&&c.tau==tau&&c.dt==dt)return c;
 Cache c{&m,baseline,tau,dt,{}};
 for(int j=0;j<m.regions;++j){const double* v=m.coeff+j*8;M a{};
  a[0]={v[1],v[2],v[3],v[0]};a[1]={v[5],v[6],v[7],v[4]};a[2][2]=-1./tau;a[2][3]=baseline/tau;
  c.flow.push_back(exponential(a,dt));}
 caches.push_back(std::move(c));return caches.back();
}
double phi(double c,const PwaConfig&m){
 if(m.affine)return c/m.ref-1.;
 const double t=c/m.ref,mx=std::max({m.p,m.e,0.});
 const double wp=std::exp(m.p-mx),we=std::exp(m.e-mx),wl=std::exp(-mx),sum=wp+we+wl;
 const double ep=std::abs(m.k)>1e-7?std::expm1(std::clamp(m.k*(t-1.),-700.,700.))/m.k:t-1.;
 return (wp*std::expm1(m.n*std::log(t))/m.n+we*ep+wl*std::log(t))/sum;
}
double dphi(double c,const PwaConfig&m){
 if(m.affine)return 1./m.ref;
 const double t=c/m.ref,mx=std::max({m.p,m.e,0.});
 const double wp=std::exp(m.p-mx),we=std::exp(m.e-mx),wl=std::exp(-mx),sum=wp+we+wl;
 return (wp*std::pow(t,m.n-1.)+we*std::exp(std::clamp(m.k*(t-1.),-700.,700.))+wl/t)/(sum*m.ref);
}
}
int pwa_advance(const PwaConfig&m,double&C,double&h,double&I,double I0,double tau,double dt){
 const double CMAX=2./1.185;
 if(!std::isfinite(C)||C<=0.||C>=CMAX||!std::isfinite(h)||!std::isfinite(I)||!std::isfinite(I0)||I0<0.||I0>1.3||tau<=0.||dt<=0.)return -100;
 const double z0=phi(C,m);const int region=std::upper_bound(m.thresholds,m.thresholds+m.regions-1,z0)-m.thresholds;
 const Cache& cache=get_cache(m,I0,tau,dt);V y=multiply(cache.flow[region],V{z0,h,I,1.});
 const bool crossed=(region<m.regions-1&&z0<m.thresholds[region]&&y[0]>=m.thresholds[region]) ||
   (region>0&&z0>m.thresholds[region-1]&&y[0]<=m.thresholds[region-1]);
 if(crossed){const double initial[]={z0,h,I};double output[6],dummy=0.;int counts[3];
  const int n=pwla_simulate(m.thresholds,m.regions,m.coeff,initial,I0,tau,dt,dt,dt,&dummy,&dummy,0,output,2,counts);
  if(n!=2)return n<0?n:-101;
  y={output[3],output[4],output[5],1.};}
 const double z=y[0];h=y[1];I=y[2];
 if(!std::isfinite(z)||!std::isfinite(h)||!std::isfinite(I))return -102;
 if(m.affine){C=m.ref*(z+1.);return (C>0.&&C<CMAX)?0:-103;}
 double low=1e-14,high=CMAX,x=C;
 for(int j=0;j<16;++j){const double r=phi(x,m)-z;if(std::abs(r)<2e-13){C=x;return 0;}
  if(r>0.)high=x;else low=x;const double d=dphi(x,m);double next=x-r/d;
  if(!std::isfinite(next)||next<=low||next>=high)next=(low+high)/2.;
  x=next;}
 if(z<=phi(low,m)||z>=phi(high,m))return -103;
 for(int j=0;j<52;++j){const double mid=(low+high)/2.;if(phi(mid,m)<z)low=mid;else high=mid;}
 C=(low+high)/2.;return 0;
}
