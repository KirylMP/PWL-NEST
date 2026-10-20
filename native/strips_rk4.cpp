// Fitting accelerator only: RK4 for two PWA state coordinates and exact stage IP3.
// This is NOT the analytic event solver. Verified against DOP853 after fitting.
#include <cmath>
#include <algorithm>
struct S {double c,h;};
static int run_rk4(const double*theta,int n,const double*p,const double*initial,
 double base,double tau,double duration,int outputs,double maxdt,double*dest,double lower,double upper,double hlower,double hupper,
 const double*slopes=nullptr,double iref=0.,const double*knots=nullptr,int knot_count=0){
 auto rhs=[&](S s,double I){
  int j=0;
  if(slopes){while(j<n-1 && s.c>=theta[j]+slopes[j]*(I-iref))++j;}
  else j=std::upper_bound(theta,theta+n-1,s.c)-theta;
  int band=knot_count?std::upper_bound(knots,knots+knot_count,I)-knots:0;
  const double*a=p+8*(band*n+j);
  return S{a[0]+a[1]*s.c+a[2]*s.h+a[3]*I,a[4]+a[5]*s.c+a[6]*s.h+a[7]*I};
 };
 S s{initial[0],initial[1]};dest[0]=s.c;dest[1]=s.h;
 double sample=duration/(outputs-1),time=0.;
 int steps=std::max(1,(int)std::ceil(sample/maxdt));double dt=sample/steps;
 auto I=[&](double t){return base+(initial[2]-base)*std::exp(-t/tau);};
 for(int i=1;i<outputs;i++){
  for(int k=0;k<steps;k++){
   S a=rhs(s,I(time));
   S b=rhs({s.c+dt*a.c/2,s.h+dt*a.h/2},I(time+dt/2));
   S c=rhs({s.c+dt*b.c/2,s.h+dt*b.h/2},I(time+dt/2));
   S d=rhs({s.c+dt*c.c,s.h+dt*c.h},I(time+dt));
   s={s.c+dt*(a.c+2*b.c+2*c.c+d.c)/6,s.h+dt*(a.h+2*b.h+2*c.h+d.h)/6};time+=dt;
   if(!std::isfinite(s.c)||!std::isfinite(s.h)||fabs(s.c)>20||fabs(s.h)>20)return 1;
   if(s.c<=lower||s.c>=upper)return 2;
   if(s.h<=hlower||s.h>=hupper)return 3;
  }
  dest[2*i]=s.c;dest[2*i+1]=s.h;
 }
 return 0;
}

extern "C" int strips_rk4(const double*theta,int n,const double*p,const double*initial,
 double base,double tau,double duration,int outputs,double maxdt,double*dest){
 return run_rk4(theta,n,p,initial,base,tau,duration,outputs,maxdt,dest,-INFINITY,INFINITY,-INFINITY,INFINITY);
}
extern "C" int strips_rk4_domain(const double*theta,int n,const double*p,const double*initial,
 double base,double tau,double duration,int outputs,double maxdt,double*dest,double lower,double upper){
 return run_rk4(theta,n,p,initial,base,tau,duration,outputs,maxdt,dest,lower,upper,-INFINITY,INFINITY);
}
extern "C" int strips_rk4_box(const double*theta,int n,const double*p,const double*initial,
 double base,double tau,double duration,int outputs,double maxdt,double*dest,double lower,double upper,double hlower,double hupper){
 return run_rk4(theta,n,p,initial,base,tau,duration,outputs,maxdt,dest,lower,upper,hlower,hupper);
}
extern "C" int strips_rk4_geometry(const double*theta,const double*slopes,int n,double iref,
 const double*knots,int knot_count,const double*p,const double*initial,
 double base,double tau,double duration,int outputs,double maxdt,double*dest,double lower,double upper,double hlower,double hupper){
 return run_rk4(theta,n,p,initial,base,tau,duration,outputs,maxdt,dest,lower,upper,hlower,hupper,slopes,iref,knots,knot_count);
}
