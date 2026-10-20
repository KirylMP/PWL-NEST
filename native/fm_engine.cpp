#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
struct State {double c,v;};
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
 auto record=[&](){minall=std::min(minall,z.c);if(t>=tail-1e-8){mn=std::min(mn,z.c);mx=std::max(mx,z.c);if(t<mid){mn1=std::min(mn1,z.c);mx1=std::max(mx1,z.c);}else{mn2=std::min(mn2,z.c);mx2=std::max(mx2,z.c);}}};
 for(int step=0;step<300000 && t<T-1e-9;step++){
  double q=qi[j]+qs[j]*I;
  if(k[j]>0 && mu[j]<0 && fabs(z.c-q)<1e-11 && fabs(z.v)<1e-11){z={q,0};stopped=true;break;}
  double tt=turn(z,q,k[j],mu[j]);double dt=std::min({1.,tt,T-t});
  if(t<tail && t+dt>tail)dt=tail-t;
  if(t<mid && t+dt>mid)dt=mid-t;
  State end=flow(dt,z,q,k[j],mu[j]);
  int dir=0;double boundary=0;
  if(j<n-1 && end.c>=theta[j] && z.c<=theta[j]){dir=1;boundary=theta[j];}
  if(j>0 && end.c<=theta[j-1] && z.c>=theta[j-1]){dir=-1;boundary=theta[j-1];}
  if(dir){
   double lo=0,hi=dt;for(int b=0;b<42;b++){double m=(lo+hi)/2;double cm=flow(m,z,q,k[j],mu[j]).c;if((cm-boundary)*dir>=0)hi=m;else lo=m;}
   dt=(lo+hi)/2;end=flow(dt,z,q,k[j],mu[j]);end.c=boundary;
   if(dt<1e-12){res[0]=res[1]=res[2]=1e4;return 2;}
  }
  double oldv=z.v;t+=dt;z=end;record();
  if(!std::isfinite(z.c)||!std::isfinite(z.v)||fabs(z.c)>20||fabs(z.v)>100){res[0]=res[1]=res[2]=1e4;return 3;}
  if(dir)j+=dir;
  else if(fabs(dt-tt)<1e-8){if(oldv>0 && t>=tail)peaks.push_back(t);z.v=0.;}
 }
 if(stopped){mn=mx=z.c;mn1=mx1=mn2=mx2=z.c;}
 if(mn>mx){mn=mx=z.c;}
 double amp=mx-mn,ratio=(mx1-mn1>1e-12)?(mx2-mn2)/(mx1-mn1):1.;
 double period=std::numeric_limits<double>::quiet_NaN();
 if(amp>1e-4 && peaks.size()>=4 && ratio>.98 && ratio<1.02){std::vector<double> periods;for(size_t i=1;i<peaks.size();i++)periods.push_back(peaks[i]-peaks[i-1]);std::sort(periods.begin(),periods.end());period=periods[periods.size()/2];}
 res[0]=mn;res[1]=mx;res[2]=period;res[3]=ratio;res[4]=minall;res[5]=stopped?1:0;return 0;
}
