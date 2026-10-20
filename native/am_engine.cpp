#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
struct State {double c,v;};
static double *trace_dest=nullptr;static int trace_capacity=0,trace_count=0;
static bool stop_on_first_maximum=false;
static bool use_target_section=false;
static double target_section_C=0.;static int target_section_direction=1;
static State flow(double t,State z,double q,double k,double mu){
 double u=z.c-q,a=mu/2,d=a*a-k;
 if(d < -1e-12){double w=sqrt(-d),e=exp(a*t),co=cos(w*t),si=sin(w*t)/w;
  return {q+e*(u*co+(z.v-a*u)*si),e*(z.v*co+(a*z.v-k*u)*si)};}
 if(d>1e-12){double w=sqrt(d),l1=a+w,l2=a-w,b1=(z.v-l2*u)/(l1-l2),b2=u-b1;
  double e1=exp(l1*t),e2=exp(l2*t);return {q+b1*e1+b2*e2,l1*b1*e1+l2*b2*e2};}
 double e=exp(a*t);return {q+e*(u+(z.v-a*u)*t),e*(z.v+(a*z.v-k*u)*t)};
}
static double turn(State z,double q,double k,double mu){
 double a=mu/2,d=a*a-k,u=z.c-q,eps=1e-9;
 if(d < -1e-12){double w=sqrt(-d),b=(a*z.v-k*u)/w;
  double ang=atan2(-z.v,b);while(ang/w<=eps)ang+=M_PI;return ang/w;}
 if(d>1e-12){double w=sqrt(d),l1=a+w,l2=a-w,b1=(z.v-l2*u)/(l1-l2),b2=u-b1;
  double A=l1*b1,B=l2*b2;if(A!=0 && -B/A>0){double t=log(-B/A)/(2*w);if(t>eps)return t;}return 1e100;}
 double b=a*z.v-k*u;if(b!=0){double t=-z.v/b;if(t>eps)return t;}return 1e100;
}
extern "C" int simulate(double I,const double*theta,const double*mu,const double*k,const double*qi,const double*qs,int n,double c0,double v0,double T,double*res){
 State z{c0,v0};double t=0,tail=T-400,mid=T-200,minall=c0;
 double mn=1e100,mx=-1e100,mn1=1e100,mx1=-1e100,mn2=1e100,mx2=-1e100;
 std::vector<double> peaks;
 int j=std::upper_bound(theta,theta+n-1,z.c)-theta;bool stopped=false;
 if(j>0 && fabs(z.c-theta[j-1])<1e-13 &&
    (z.v<0 || (z.v==0 && qi[j]-k[j]*z.c<0)))j--;
 auto record=[&](){if(trace_dest&&trace_count<trace_capacity){trace_dest[3*trace_count]=t;trace_dest[3*trace_count+1]=z.c;trace_dest[3*trace_count+2]=z.v;trace_count++;}minall=std::min(minall,z.c);if(t>=tail-1e-8){mn=std::min(mn,z.c);mx=std::max(mx,z.c);if(t<mid){mn1=std::min(mn1,z.c);mx1=std::max(mx1,z.c);}else{mn2=std::min(mn2,z.c);mx2=std::max(mx2,z.c);}}};
 for(int step=0;step<300000 && t<T-1e-9;step++){
  double q=qi[j]/k[j];
  if(k[j]>0 && mu[j]<0 && (j==0||q>=theta[j-1]) && (j==n-1||q<=theta[j]) && fabs(z.c-q)<1e-11 && fabs(z.v)<1e-11){z={q,0};stopped=true;break;}
  double tt=turn(z,q,k[j],mu[j]);double rate=mu[j]/2+sqrt(std::max(0.,mu[j]*mu[j]/4-k[j]));double guard=rate>0?30/rate:1e100;double dt=std::min({guard,tt,T-t});
  if(t<tail && t+dt>tail)dt=tail-t;
  if(t<mid && t+dt>mid)dt=mid-t;
  State end=flow(dt,z,q,k[j],mu[j]);
  int dir=0;double boundary=0;
  if(j<n-1 && end.c>=theta[j] && z.c<=theta[j]){dir=1;boundary=theta[j];}
  if(j>0 && end.c<=theta[j-1] && z.c>=theta[j-1]){dir=-1;boundary=theta[j-1];}
  bool hit_target=false;
  if(use_target_section && fabs(z.c-target_section_C)>1e-12 &&
     (end.c-z.c)*target_section_direction>0 &&
     (target_section_C-z.c)*target_section_direction>0 &&
     (end.c-target_section_C)*target_section_direction>=0 &&
     (!dir || fabs(target_section_C-z.c)<=fabs(boundary-z.c))){
    dir=target_section_direction;boundary=target_section_C;hit_target=true;
  }
  if(dir){
   double lo=0,hi=dt,m=dt/2;for(int b=0;b<80;b++){State y=flow(m,z,q,k[j],mu[j]);double f=y.c-boundary;if((f)*dir>=0)hi=m;else lo=m;double dx=y.v!=0?f/y.v:INFINITY;if(fabs(dx)<2e-12){lo=hi=m;break;}if(hi-lo<2e-12)break;double nt=m-dx;m=(std::isfinite(nt)&&nt>lo&&nt<hi)?nt:(lo+hi)/2;}
   dt=(lo+hi)/2;end=flow(dt,z,q,k[j],mu[j]);end.c=boundary;
   if(dt<1e-12){res[0]=res[1]=res[2]=1e4;return 2;}
  }
  double oldv=z.v;t+=dt;z=end;record();
  if(!std::isfinite(z.c)||!std::isfinite(z.v)||fabs(z.c)>20||fabs(z.v)>100){res[0]=res[1]=res[2]=1e4;return 3;}
  if(hit_target){stopped=true;break;}
  if(dir)j+=dir;
  else if(fabs(dt-tt)<1e-8){if(oldv>0 && t>=tail)peaks.push_back(t);z.v=0.;if(stop_on_first_maximum && oldv>0){stopped=true;break;}}
 }
 if(t<T-1e-7 && !stopped)return 5;
 if(stopped){mn=mx=z.c;mn1=mx1=mn2=mx2=z.c;}
 if(mn>mx){mn=mx=z.c;}
 double amp=mx-mn,ratio=(mx1-mn1>1e-12)?(mx2-mn2)/(mx1-mn1):1.;
 double period=std::numeric_limits<double>::quiet_NaN();
 if(amp>1e-4 && peaks.size()>=4 && ratio>.98 && ratio<1.02){std::vector<double> periods;for(size_t i=1;i<peaks.size();i++)periods.push_back(peaks[i]-peaks[i-1]);std::sort(periods.begin(),periods.end());period=periods[periods.size()/2];}
 res[0]=mn;res[1]=mx;res[2]=period;res[3]=ratio;res[4]=minall;res[5]=stopped?1:0;return 0;
}
// Exact affine propagator in physical continuous coordinates (C,w).
extern "C" void advance_region(double t,double C,double w,double mu,double k,double q,double g,double*out){
 State z=flow(t,{C,w+mu*C+g},q/k,k,mu);
 out[0]=z.c;out[1]=z.v-mu*z.c-g;
}

extern "C" int simulate_trace(double I,const double*theta,const double*mu,const double*k,const double*qi,const double*qs,int n,double c0,double v0,double T,double*res,double*dest,int capacity){trace_dest=dest;trace_capacity=capacity;trace_count=0;int code=simulate(I,theta,mu,k,qi,qs,n,c0,v0,T,res);trace_dest=nullptr;return code?-code:trace_count;}

// A Poincare return needs exactly one circuit, independent of later transients.
// The final event is recorded before terminating; res is not an orbit statistic.
extern "C" int first_return_trace(double I,const double*theta,const double*mu,const double*k,const double*qi,const double*qs,int n,double c0,double v0,double T,double*res,double*dest,int capacity){stop_on_first_maximum=true;int code=simulate_trace(I,theta,mu,k,qi,qs,n,c0,v0,T,res,dest,capacity);stop_on_first_maximum=false;return code;}

extern "C" int section_trace(double I,const double*theta,const double*mu,const double*k,const double*qi,const double*qs,int n,double c0,double v0,double T,double*res,double*dest,int capacity,double section,int direction){use_target_section=true;target_section_C=section;target_section_direction=direction;int code=simulate_trace(I,theta,mu,k,qi,qs,n,c0,v0,T,res,dest,capacity);use_target_section=false;return code;}
