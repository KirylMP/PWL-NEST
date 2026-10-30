#include <algorithm>
#include <cmath>
#include <vector>
static double phi(double c,const double*){return c;}
static double inv(double z,const double*){return z;}
extern "C" int simulate_shape(const double *co,const double *cuts,const double *sh,double I,double C0,double q0,double dt,int steps,int nphase,double *out){
 double c=phi(C0,sh),q=q0,prev2=c,prev1=c,zmin=phi(1e-12,sh),zmax=phi(2/1.185,sh);double peak0=-1,peak1=-1;int peaks=0;std::vector<double> trace;trace.reserve(steps/2+2);
 auto f=[&](double x,double y,double &dx,double &dy){double u=std::max(0.,x-cuts[0]),v=std::max(0.,x-cuts[1]);dx=co[0]+co[1]*x+co[2]*y+co[3]*I+co[4]*u+co[5]*v;dy=co[11]+co[12]*x+co[13]*y+co[14]*I+co[15]*u+co[16]*v;};
 for(int n=1;n<=steps;n++){double a,b,d,e,g,h,j,k;f(c,q,a,b);f(c+.5*dt*a,q+.5*dt*b,d,e);f(c+.5*dt*d,q+.5*dt*e,g,h);f(c+dt*g,q+dt*h,j,k);double next=c+dt*(a+2*d+2*g+j)/6.,qn=q+dt*(b+2*e+2*h+k)/6.;if(!std::isfinite(next)||!std::isfinite(qn)||next<=zmin||next>=zmax)return 0;
 if(n>steps/2){trace.push_back(next);int idx=int(trace.size())-1;if(idx>=2&&prev1>prev2&&prev1>=next){double den=prev2-2*prev1+next;double offset=std::abs(den)>1e-15?.5*(prev2-next)/den:0.;offset=std::max(-.5,std::min(.5,offset));peak0=peak1;peak1=idx-1+offset;peaks++;}}prev2=prev1;prev1=next;c=next;q=qn;}
 if(peaks<3||peak0<0||peak1<=peak0)return 0;double mn=1e99,mx=-1e99;for(int j=int(std::floor(peak0));j<=int(std::ceil(peak1));j++){mn=std::min(mn,trace[j]);mx=std::max(mx,trace[j]);}double cmin=inv(mn,sh),cmax=inv(mx,sh);if(cmax-cmin<.001)return 0;
 out[0]=cmin;out[1]=cmax;out[2]=(peak1-peak0)*dt;out[3]=peaks;for(int j=0;j<nphase;j++){double p=peak0+(peak1-peak0)*(double(j)/nphase);int a=int(p);double frac=p-a;out[4+j]=inv(trace[a]*(1-frac)+trace[a+1]*frac,sh);}return 1;
}
