#include <algorithm>
#include <cmath>
#include <vector>
extern "C" int cycle(const double* co,const double* cuts,double gamma,const double* init,double dt,int steps,double* out){
 double x[3]={init[0],init[1],init[2]};double prev2=x[0],prev1=x[0],peak0=-1,peak1=-1;int peaks=0;std::vector<double> C,I,Q; C.reserve(steps/2+2); I.reserve(steps/2+2); Q.reserve(steps/2+2);
 auto f=[&](const double*y,double*d){double v[8]={1,y[0],y[1],y[2],std::max(0.,y[0]-cuts[0]),std::max(0.,y[0]-cuts[1]),std::max(0.,y[0]-cuts[2]),gamma};for(int j=0;j<3;j++){d[j]=0;for(int k=0;k<8;k++)d[j]+=co[j*8+k]*v[k];}};
 for(int n=1;n<=steps;n++){double a[3],b[3],c[3],d[3],t[3],y[3];f(x,a);for(int j=0;j<3;j++)t[j]=x[j]+.5*dt*a[j];f(t,b);for(int j=0;j<3;j++)t[j]=x[j]+.5*dt*b[j];f(t,c);for(int j=0;j<3;j++)t[j]=x[j]+dt*c[j];f(t,d);for(int j=0;j<3;j++)y[j]=x[j]+dt*(a[j]+2*b[j]+2*c[j]+d[j])/6.;
  if(!std::isfinite(y[0])||!std::isfinite(y[1])||!std::isfinite(y[2])||y[0]<=0||y[0]>=2./1.185||y[2]<0||y[2]>8||std::abs(y[1])>1e4)return 0;
  if(n>steps/2){C.push_back(y[0]);I.push_back(y[2]);Q.push_back(y[1]);int j=int(C.size())-1;if(j>=2&&prev1>prev2&&prev1>=y[0]){double den=prev2-2*prev1+y[0];double delta=std::abs(den)>1e-15?.5*(prev2-y[0])/den:0.;delta=std::max(-.5,std::min(.5,delta));peak0=peak1;peak1=j-1+delta;peaks++;}}
  prev2=prev1;prev1=y[0];for(int j=0;j<3;j++)x[j]=y[j];
 }
 if(peaks<3||peak0<0||peak1<=peak0)return 0;int a=std::max(0,int(std::floor(peak0))),b=std::min(int(C.size())-1,int(std::ceil(peak1)));double cmin=1e9,cmax=-1e9,imin=1e9,imax=-1e9; int p0=std::max(0,int(std::round(peak0))),p1=std::min(int(C.size())-1,int(std::round(peak1))); if(std::abs(C[p0]-C[p1])>0.001*std::max(1.,std::abs(C[p1])) || std::abs(I[p0]-I[p1])>0.001*std::max(1.,std::abs(I[p1])) || std::abs(Q[p0]-Q[p1])>0.001*std::max(1.,std::abs(Q[p1]))) return 0;
 for(int j=a;j<=b;j++){cmin=std::min(cmin,C[j]);cmax=std::max(cmax,C[j]);imin=std::min(imin,I[j]);imax=std::max(imax,I[j]);}if(cmax-cmin<.001)return 0;
 out[0]=cmin;out[1]=cmax;out[2]=imin;out[3]=imax;out[4]=(peak1-peak0)*dt;return 1;
}
