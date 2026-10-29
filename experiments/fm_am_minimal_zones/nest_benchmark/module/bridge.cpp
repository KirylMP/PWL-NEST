#include "bridge.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>
#ifdef PWA_PROFILE
#include <chrono>
#include <cstdint>
#include <mutex>
namespace {
struct ProfileCounts {std::uint64_t values[12]{};};
std::mutex profile_mutex;
std::vector<ProfileCounts*> profile_registry;
ProfileCounts& profile_local(){
 static thread_local ProfileCounts* data=[](){auto* p=new ProfileCounts;std::lock_guard<std::mutex> lock(profile_mutex);profile_registry.push_back(p);return p;}();
 return *data;
}
struct ProfileTimer {
 int slot; std::chrono::steady_clock::time_point start;
 explicit ProfileTimer(int n):slot(n),start(std::chrono::steady_clock::now()){}
 ~ProfileTimer(){profile_local().values[slot]+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();}
};
}
// Called only while NEST workers are idle, outside nest.Run.
extern "C" void pwa_profile_reset(){std::lock_guard<std::mutex> lock(profile_mutex);for(auto* p:profile_registry)for(auto& v:p->values)v=0;}
extern "C" void pwa_profile_snapshot(std::uint64_t* out){std::lock_guard<std::mutex> lock(profile_mutex);for(int i=0;i<12;++i){out[i]=0;for(auto* p:profile_registry)out[i]+=p->values[i];}}
#define PWA_COUNT(i) (++profile_local().values[i])
#define PWA_TIMER(i) ProfileTimer timer_##i(i)
#else
#define PWA_COUNT(i) ((void)0)
#define PWA_TIMER(i) ((void)0)
#endif
namespace {
constexpr double CMAX=2./1.185;
using V=std::array<double,4>; using M=std::array<std::array<double,4>,4>;
M identity(){M a{};for(int i=0;i<4;++i)a[i][i]=1.;return a;}
M multiply(const M&a,const M&b){M c{};for(int i=0;i<4;++i)for(int k=0;k<4;++k)for(int j=0;j<4;++j)c[i][j]+=a[i][k]*b[k][j];return c;}
V multiply(const M&a,const V&x){V y{};for(int i=0;i<4;++i)for(int j=0;j<4;++j)y[i]+=a[i][j]*x[j];return y;}
M exponential(M a,double t){
 PWA_COUNT(9);PWA_TIMER(10);
 double norm=0.;for(int i=0;i<4;++i){double sum=0.;for(int j=0;j<4;++j)sum+=std::abs(a[i][j]*t);norm=std::max(norm,sum);}
 int scale=norm>.5?std::max(0,(int)std::ceil(std::log2(norm/.5))):0;double factor=t/std::ldexp(1.,scale);
 for(auto&row:a)for(double&x:row)x*=factor;
 M sum=identity(),term=identity();
 for(int k=1;k<=32;++k){term=multiply(term,a);double largest=0.;for(int i=0;i<4;++i)for(int j=0;j<4;++j){term[i][j]/=k;sum[i][j]+=term[i][j];largest=std::max(largest,std::abs(term[i][j]));}if(largest<2e-17)break;}
 for(int i=0;i<scale;++i)sum=multiply(sum,sum);
 return sum;
}
struct Transform {
 const PwaConfig* cfg;double wp,we,wl,zmin,zmax;
 explicit Transform(const PwaConfig& m):cfg(&m){const double mx=std::max({m.p,m.e,0.});wp=std::exp(m.p-mx);we=std::exp(m.e-mx);wl=std::exp(-mx);const double sum=wp+we+wl;wp/=sum;we/=sum;wl/=sum;zmin=value(1e-14);zmax=value(CMAX);}
 double value(double c)const{
  const auto&m=*cfg;if(m.direct_calcium)return c;if(m.affine)return c/m.ref-1.;const double t=c/m.ref;
  const double e=std::abs(m.k)>1e-7?std::expm1(std::clamp(m.k*(t-1.),-700.,700.))/m.k:t-1.;
  return wp*std::expm1(m.n*std::log(t))/m.n+we*e+wl*std::log(t);
 }
 double derivative(double c)const{
  const auto&m=*cfg;if(m.direct_calcium)return 1.;if(m.affine)return 1./m.ref;const double t=c/m.ref;
  return (wp*std::pow(t,m.n-1.)+we*std::exp(std::clamp(m.k*(t-1.),-700.,700.))+wl/t)/m.ref;
 }
};
const Transform& transform(const PwaConfig&m){
 static thread_local std::vector<Transform> entries;
 for(const auto& x:entries)if(x.cfg==&m)return x;
 entries.emplace_back(m);return entries.back();
}
struct Cache {const PwaConfig* cfg;double baseline,tau,dt;std::vector<M> matrix,flow;};
const Cache& get_cache(const PwaConfig&m,double baseline,double tau,double dt){
 static thread_local std::vector<Cache> caches;
 for(const auto& c:caches)if(c.cfg==&m&&c.baseline==baseline&&c.tau==tau&&c.dt==dt)return c;
 Cache c{&m,baseline,tau,dt,{},{}};
 for(int j=0;j<m.regions;++j){const double* v=m.coeff+j*8;M a{};
  a[0]={v[1],v[2],v[3],v[0]};a[1]={v[5],v[6],v[7],v[4]};a[2][2]=-1./tau;a[2][3]=baseline/tau;
  c.matrix.push_back(a);c.flow.push_back(exponential(a,dt));}
 caches.push_back(std::move(c));return caches.back();
}
struct SicCut {const PwaConfig* cfg;double threshold,z;};
double sic_cut(const PwaConfig&m,double threshold){
 static thread_local std::vector<SicCut> cuts;
 for(const auto& c:cuts)if(c.cfg==&m&&c.threshold==threshold)return c.z;
 const double c=threshold+.001;
 const double z=c<=1e-14?-INFINITY:(c>=CMAX?INFINITY:transform(m).value(c));
 cuts.push_back(SicCut{&m,threshold,z});return z;
}
}
int pwa_prepare(const PwaConfig&m,double I0,double tau,double dt,double threshold){
 if(!std::isfinite(I0)||I0<0.||I0>1.3||!std::isfinite(tau)||tau<=0.||!std::isfinite(dt)||dt<=0.||!std::isfinite(threshold))return -100;
 (void)get_cache(m,I0,tau,dt);(void)transform(m);(void)sic_cut(m,threshold);return 0;
}
int pwa_to_z(const PwaConfig&m,double C,double&z){
 if(!std::isfinite(C)||C<=1e-14||C>=CMAX)return -100;
 z=transform(m).value(C);return std::isfinite(z)?0:-100;
}
int pwa_to_c(const PwaConfig&m,double z,double&hint){
 PWA_COUNT(4);PWA_TIMER(5);
 if(!std::isfinite(z))return -102;
 const auto& tr=transform(m);if(m.direct_calcium){hint=z;return (hint>1e-14&&hint<CMAX)?0:-103;}if(m.affine){hint=m.ref*(z+1.);return (hint>1e-14&&hint<CMAX)?0:-103;}
 double low=1e-14,high=CMAX,x=(hint>low&&hint<high)?hint:.5;
 for(int j=0;j<16;++j){const double r=tr.value(x)-z;if(std::abs(r)<2e-13){hint=x;return 0;}
  if(r>0.)high=x;else low=x;double next=x-r/tr.derivative(x);
  if(!std::isfinite(next)||next<=low||next>=high)next=(low+high)/2.;
  x=next;}
 if(z<=tr.value(low)||z>=tr.value(high))return -103;
 for(int j=0;j<52;++j){const double mid=(low+high)/2.;if(tr.value(mid)<z)low=mid;else high=mid;}
 hint=(low+high)/2.;return 0;
}
int pwa_advance_z(const PwaConfig&m,double&z,double&h,double&I,double I0,double tau,double dt){
 PWA_COUNT(0);PWA_TIMER(1);
 if(!std::isfinite(z)||!std::isfinite(h)||!std::isfinite(I)||!std::isfinite(I0)||I0<0.||I0>1.3||tau<=0.||dt<=0.)return -100;
 const Cache& cache=get_cache(m,I0,tau,dt);V y{z,h,I,1.};
 int r=std::upper_bound(m.thresholds,m.thresholds+m.regions-1,z)-m.thresholds;
 // On an exact boundary choose the region into which the vector field points.
 if(r>0&&z==m.thresholds[r-1]){const double* p=m.coeff+8*r;if(p[0]+p[1]*z+p[2]*h+p[3]*I<0.)--r;}
 double left=dt;int exits=0;
 while(left>1e-15){
  const bool cached=std::abs(left-dt)<1e-15;if(cached)PWA_COUNT(11);
  V end=multiply(cached?cache.flow[r]:exponential(cache.matrix[r],left),y);
  int direction=0;double boundary=0.;
  if(r<m.regions-1&&end[0]>m.thresholds[r]){direction=1;boundary=m.thresholds[r];}
  else if(r>0&&end[0]<m.thresholds[r-1]){direction=-1;boundary=m.thresholds[r-1];}
  if(!direction){y=end;left=0.;break;}
  PWA_COUNT(2);PWA_TIMER(3);
  double low=0.,high=left;
  for(int j=0;j<42&&high-low>1e-13;++j){const double mid=(low+high)/2.;
   if((multiply(exponential(cache.matrix[r],mid),y)[0]-boundary)*direction>=0.)high=mid;else low=mid;}
  const double step=(low+high)/2.;y=multiply(exponential(cache.matrix[r],step),y);y[0]=boundary;
  left-=step;r+=direction;if(++exits>100)return -104;
 }
 z=y[0];h=y[1];I=y[2];if(!std::isfinite(z)||!std::isfinite(h)||!std::isfinite(I))return -102;
 const auto& tr=transform(m);if(z<=tr.zmin||z>=tr.zmax)return -103;
 return 0;
}
int pwa_sic(const PwaConfig&m,double z,double threshold,double scale,double&hint,double&value){
 PWA_COUNT(6);PWA_TIMER(7);
 if(z<=sic_cut(m,threshold)){value=0.;return 0;}
 PWA_COUNT(8);
 const int status=pwa_to_c(m,z,hint);if(status)return status;
 const double a=(hint-threshold)*1000.;value=a>1.?std::log(a)*scale:0.;return 0;
}
