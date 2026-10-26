#include <cmath>
#include <algorithm>
#include <cstring>
// Adaptive Dormand-Prince 5(4), 2D state plus integrated divergence.
// mode=0 LR FM; mode=1 PWA in z,h. Outputs forward-time stability even in reverse.
static void fun(int mode,const double* co,const double* th,double I,int dir,const double* y,double* f){
 double c=y[0],h=y[1],tr;
 if(mode==0){double q=1.049*(I+.13)/(I+.9434),m=I/(I+.13),n=c/(c+.08234),d=2-1.185*c;
  f[0]=(6*m*m*m*n*n*n*h*h*h+.11)*d-.9*c*c/(c*c+.01);
  f[1]=.2*(q*(1-h)-c*h);
  tr=6*m*m*m*h*h*h*(3*n*n*.08234/((c+.08234)*(c+.08234))*d-1.185*n*n*n)-.13035-.018*c/((c*c+.01)*(c*c+.01))-.2*(q+c);
 }else{f[0]=co[0]+co[1]*c+co[2]*h+co[3]*I;f[1]=co[11]+co[12]*c+co[13]*h+co[14]*I;tr=co[1]+co[13];
  for(int k=0;k<7;k++)if(c>th[k]){f[0]+=co[k+4]*(c-th[k]);f[1]+=co[k+15]*(c-th[k]);tr+=co[k+4];}
 }
 f[0]*=dir;f[1]*=dir;f[2]=tr*dir;
}
static const double a[7][7]={{0},{1./5},{3./40,9./40},{44./45,-56./15,32./9},{19372./6561,-25360./2187,64448./6561,-212./729},{9017./3168,-355./33,46732./5247,49./176,-5103./18656},{35./384,0,500./1113,125./192,-2187./6784,11./84}};
static const double b[7]={35./384,0,500./1113,125./192,-2187./6784,11./84,0};
static const double err[7]={35./384-5179./57600,0,500./1113-7571./16695,125./192-393./640,-2187./6784+92097./339200,11./84-187./2100,-1./40};
static void step(int mode,const double* co,const double* th,double I,int dir,const double* y,double dt,double* z,double* e){
 double k[7][3],v[3];
 for(int j=0;j<7;j++){for(int d=0;d<3;d++){v[d]=y[d];for(int l=0;l<j;l++)v[d]+=dt*a[j][l]*k[l][d];}fun(mode,co,th,I,dir,v,k[j]);}
 for(int d=0;d<3;d++){z[d]=y[d];e[d]=0;for(int j=0;j<7;j++){z[d]+=dt*b[j]*k[j][d];e[d]+=dt*err[j]*k[j][d];}}
}
extern "C" int first_return(int mode,const double* co,const double* th,double I,int dir,double section,double s,double lo,double hi,double rtol,double tmax,double* out){
 double y[3]={section,s,0},z[3],e[3],v0[3];fun(mode,co,th,I,dir,y,v0);
 if(v0[0]*dir<=1e-13)return -3;
 double t=0,dt=.002,mn=section,mx=section,hmn=s,hmx=s;bool departed=false;
 for(int iter=0;iter<250000 && t<tmax;iter++){
  dt=std::min(dt,tmax-t);step(mode,co,th,I,dir,y,dt,z,e);double en=0;
  int dims=mode==2?2:3;
  for(int d=0;d<dims;d++){double scale=1e-11+rtol*std::max(std::abs(y[d]),std::abs(z[d]));en+=std::pow(e[d]/scale,2);}en=std::sqrt(en/dims);
  if(!std::isfinite(en)||!std::isfinite(z[0])||!std::isfinite(z[1]))return -4;
  if(en>1 && dt>1e-10){dt*=std::max(.1,.9*std::pow(en,-.2));continue;}
  if(departed && t>1e-7 && (y[0]-section)*dir<0 && (z[0]-section)*dir>=0){
   double l=0,r=dt,root[3],ee[3];
   for(int j=0;j<34;j++){double mid=(l+r)/2;step(mode,co,th,I,dir,y,mid,root,ee);if((root[0]-section)*dir>=0)r=mid;else l=mid;}
   double rt=(l+r)/2;step(mode,co,th,I,dir,y,rt,root,ee);double vend[3];fun(mode,co,th,I,dir,root,vend);
   out[0]=root[1]-s;out[1]=t+rt;out[2]=dir*(root[2]+std::log(std::abs(v0[0]/vend[0])));out[3]=std::min(mn,root[0]);out[4]=std::max(mx,root[0]);out[5]=std::min(hmn,root[1]);out[6]=std::max(hmx,root[1]);return 1;
  }
  if((z[0]-section)*dir>1e-9)departed=true;
  mn=std::min(mn,z[0]);mx=std::max(mx,z[0]);hmn=std::min(hmn,z[1]);hmx=std::max(hmx,z[1]);t+=dt;std::memcpy(y,z,sizeof(y));
  if(y[0]<lo-1e-7||y[0]>hi+1e-7||y[1]<-1e-7||y[1]>1+1e-7)return -1;
  double vel[3];fun(mode,co,th,I,dir,y,vel);if(t>5 && std::hypot(vel[0],vel[1])<1e-11)return 0;
  dt=std::min(.35,dt*std::min(4.,std::max(.2,.9*std::pow(std::max(en,1e-12),-.2))));
 }
 return -2;
}
