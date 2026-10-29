// Finite-time RK4 observable waveform, sampled from the last two peaks.
#include <algorithm>
#include <cmath>
#include <vector>
extern "C" int simulate_shape(const double *co,const double *cuts,double I,double C0,double q0,double dt,int steps,int nphase,double *out) {
  double c=C0,q=q0,prev2=c,prev1=c; int peak0=-1,peak1=-1,peaks=0;
  std::vector<double> trace;trace.reserve(steps/2+2);
  auto f=[&](double x,double y,double &dx,double &dy){double u=std::max(0.,x-cuts[0]),v=std::max(0.,x-cuts[1]);dx=co[0]+co[1]*x+co[2]*y+co[3]*I+co[4]*u+co[5]*v;dy=co[11]+co[12]*x+co[13]*y+co[14]*I+co[15]*u+co[16]*v;};
  for(int n=1;n<=steps;n++){
    double a,b,d,e,g,h,j,k;f(c,q,a,b);f(c+.5*dt*a,q+.5*dt*b,d,e);f(c+.5*dt*d,q+.5*dt*e,g,h);f(c+dt*g,q+dt*h,j,k);
    double next=c+dt*(a+2*d+2*g+j)/6.,qn=q+dt*(b+2*e+2*h+k)/6.;
    if(!std::isfinite(next)||!std::isfinite(qn)||next<=0||next>=2./1.185||qn<=0||qn>=1)return 0;
    if(n>steps/2){trace.push_back(next);int idx=int(trace.size())-1;if(idx>=2 && prev1>prev2 && prev1>=next){peak0=peak1;peak1=idx-1;peaks++;}}
    prev2=prev1;prev1=next;c=next;q=qn;
  }
  if(peaks<3||peak0<0||peak1<=peak0)return 0;
  double mn=1e99,mx=-1e99;for(int j=peak0;j<=peak1;j++){mn=std::min(mn,trace[j]);mx=std::max(mx,trace[j]);}
  if(mx-mn<.001)return 0;
  out[0]=mn;out[1]=mx;out[2]=(peak1-peak0)*dt;out[3]=peaks;
  for(int j=0;j<nphase;j++){double p=peak0+(peak1-peak0)*(double(j)/nphase);int a=int(p);double frac=p-a;out[4+j]=trace[a]*(1-frac)+trace[a+1]*frac;}
  return 1;
}
