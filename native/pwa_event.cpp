#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
using V4 = std::array<double,4>;
using M4 = std::array<std::array<double,4>,4>;

M4 identity() {
  M4 a{};
  for (int i=0;i<4;++i) a[i][i]=1.0;
  return a;
}
M4 multiply(const M4& a,const M4& b) {
  M4 c{};
  for(int i=0;i<4;++i) for(int k=0;k<4;++k) for(int j=0;j<4;++j) c[i][j]+=a[i][k]*b[k][j];
  return c;
}
V4 multiply(const M4& a,const V4& x) {
  V4 y{};
  for(int i=0;i<4;++i) for(int j=0;j<4;++j) y[i]+=a[i][j]*x[j];
  return y;
}
M4 exponential(M4 a,double t) {
  double norm=0.0;
  for(int i=0;i<4;++i) { double s=0.0; for(int j=0;j<4;++j) s+=std::abs(a[i][j]*t); norm=std::max(norm,s); }
  int scale=norm>0.5 ? std::max(0,(int)std::ceil(std::log2(norm/0.5))) : 0;
  double divisor=std::ldexp(1.0,scale);
  for(auto& row:a) for(double& x:row) x*=t/divisor;
  M4 sum=identity(),term=identity();
  for(int k=1;k<=32;++k) {
    term=multiply(term,a);
    double largest=0.0;
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) { term[i][j]/=k; sum[i][j]+=term[i][j]; largest=std::max(largest,std::abs(term[i][j])); }
    if(largest<2e-17) break;
  }
  for(int i=0;i<scale;++i) sum=multiply(sum,sum);
  return sum;
}

V4 advance(const double* coeff,const V4& y,double baseline,double tau,double dt,int* flows) {
  M4 a{};
  // coeff is [row0: intercept,u,v,I; row1: intercept,u,v,I].
  a[0]={coeff[1],coeff[2],coeff[3],coeff[0]};
  a[1]={coeff[5],coeff[6],coeff[7],coeff[4]};
  a[2][2]=-1.0/tau; a[2][3]=baseline/tau;
  V4 z={y[0],y[1],y[2],1.0};
  ++*flows;
  return multiply(exponential(a,dt),z);
}
}

// coeff: region-major [nregion, output row, (intercept,u,v,I)].
// Returns output row count; negative values identify invalid input or event-loop failure.
extern "C" int pwla_simulate(const double* thresholds,int nregions,const double* coeff,
    const double* initial,double baseline,double tau,double duration,double output_dt,double scan_dt,
    const double* jump_times,const double* jump_deltas,int njumps,
    double* out,int capacity,int* counters) {
  if(!thresholds||!coeff||!initial||!out||!counters||nregions<1||capacity<2||njumps<0||
     !std::isfinite(duration)||!std::isfinite(output_dt)||!std::isfinite(scan_dt)||!std::isfinite(tau)||
     !(duration>0&&output_dt>0&&scan_dt>0&&tau>0)||!std::isfinite(baseline)||baseline<0||baseline>1.3) return -1;
  for(int j=0;j<nregions-1;++j) if(!std::isfinite(thresholds[j])||(j&&thresholds[j]<=thresholds[j-1])) return -2;
  for(int j=0;j<8*nregions;++j) if(!std::isfinite(coeff[j])) return -3;
  if(!std::isfinite(initial[0])||!std::isfinite(initial[1])||!std::isfinite(initial[2])) return -4;
  for(int j=0;j<njumps;++j) if(!jump_times||!jump_deltas||!std::isfinite(jump_times[j])||!std::isfinite(jump_deltas[j])||
      jump_times[j]<=0||jump_times[j]>=duration||(j&&jump_times[j]<=jump_times[j-1])) return -5;
  int count=0,flows=0,crossings=0,jump_count=0;
  V4 y={initial[0],initial[1],initial[2],0.};
  int region=(int)(std::upper_bound(thresholds,thresholds+nregions-1,y[0])-thresholds);
  double t=0.0; out[0]=y[0];out[1]=y[1];out[2]=y[2];count=1;
  int jump=0; double output_time=std::min(output_dt,duration);
  const int max_events=10000000;
  while(t<duration-2e-13) {
    if(count>=capacity) return -6;
    double target=output_time;
    if(jump<njumps && jump_times[jump]<target) target=jump_times[jump];
    if(target<=t+2e-13) {
      if(jump<njumps && std::abs(jump_times[jump]-t)<=2e-12) { y[2]+=jump_deltas[jump++]; ++jump_count; continue; }
      if(std::abs(output_time-t)<=2e-12) { out[3*count]=y[0];out[3*count+1]=y[1];out[3*count+2]=y[2];++count;output_time=std::min(output_time+output_dt,duration);continue; }
      return -7;
    }
    double h=std::min(scan_dt,target-t);
    const double* local=coeff+8*region;
    V4 end=advance(local,y,baseline,tau,h,&flows);
    int direction=0; double boundary=0.0;
    if(region<nregions-1 && y[0]<thresholds[region] && end[0]>=thresholds[region]) { direction=1;boundary=thresholds[region]; }
    else if(region>0 && y[0]>thresholds[region-1] && end[0]<=thresholds[region-1]) { direction=-1;boundary=thresholds[region-1]; }
    if(direction) {
      double lo=0.0,hi=h;
      for(int k=0;k<56;++k) {
        double mid=.5*(lo+hi);V4 probe=advance(local,y,baseline,tau,mid,&flows);
        if((probe[0]-boundary)*direction>=0) hi=mid; else lo=mid;
      }
      double step=.5*(lo+hi); y=advance(local,y,baseline,tau,step,&flows);y[0]=boundary;
      t+=step;region+=direction;++crossings;
      if(crossings>max_events) return -8;
    } else { y=end;t+=h; }
    if(std::abs(t-target)<=2e-12) {
      t=target;
      if(jump<njumps && std::abs(jump_times[jump]-t)<=2e-12) { y[2]+=jump_deltas[jump++];++jump_count; }
      if(std::abs(output_time-t)<=2e-12) {
        out[3*count]=y[0];out[3*count+1]=y[1];out[3*count+2]=y[2];++count;
        output_time=std::min(output_time+output_dt,duration);
      }
    }
  }
  counters[0]=flows;counters[1]=crossings;counters[2]=jump_count;
  return count;
}
