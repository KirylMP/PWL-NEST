// Fixed-step RK4 for physical calcium and two continuous PWA boundaries.
#include <algorithm>
#include <cmath>
extern "C" int simulate(const double *co,const double *cuts,double I,double C0,double q0,
                        double dt,int steps,double *out) {
  double c=C0,q=q0,old2=C0,old1=C0,lastpeak=-1.,sumperiod=0.;
  double cmin=1e100,cmax=-1e100,qmin=q,qmax=q;int peaks=0,periods=0;
  auto f=[&](double x,double y,double &dx,double &dy){
    double u=std::max(0.,x-cuts[0]),v=std::max(0.,x-cuts[1]);
    dx=co[0]+co[1]*x+co[2]*y+co[3]*I+co[4]*u+co[5]*v;
    dy=co[11]+co[12]*x+co[13]*y+co[14]*I+co[15]*u+co[16]*v;
  };
  for(int n=1;n<=steps;n++) {
    double a,b,d,e,g,h,j,k; f(c,q,a,b); f(c+.5*dt*a,q+.5*dt*b,d,e);
    f(c+.5*dt*d,q+.5*dt*e,g,h); f(c+dt*g,q+dt*h,j,k);
    double next=c+dt*(a+2*d+2*g+j)/6.,qn=q+dt*(b+2*e+2*h+k)/6.;
    if(!std::isfinite(next)||!std::isfinite(qn)||next<0||next>2./1.185||qn<0||qn>1)return 0;
    qmin=std::min(qmin,qn);qmax=std::max(qmax,qn);
    if(n>steps/2) {
      cmin=std::min(cmin,next);cmax=std::max(cmax,next);
      if(old1>old2 && old1>=next) {
        double den=old2-2*old1+next,shift=std::abs(den)>1e-16?.5*(old2-next)/den:0.;
        shift=std::max(-1.,std::min(1.,shift));double t=(n-1+shift)*dt;
        if(lastpeak>=0){sumperiod+=t-lastpeak;periods++;}lastpeak=t;peaks++;
      }
    }
    old2=old1;old1=next;c=next;q=qn;
  }
  out[0]=cmin;out[1]=cmax;out[2]=periods?sumperiod/periods:0.;out[3]=peaks;out[4]=qmin;out[5]=qmax;
  return 1;
}
