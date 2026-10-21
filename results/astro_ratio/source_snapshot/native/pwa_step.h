// Cached, per-NEST-tick adapter around the existing affine flow implementation.
// pwa_event.cpp is included once by the module translation unit, not linked twice.
#pragma once
#include "pwa_event.cpp"
#include <stdexcept>
#include <vector>

namespace pwla {
struct Stepper {
  std::vector<double> thresholds, coefficients, input_thresholds;
  std::vector<M4> cached;
  double baseline=.45, tau=20., dt=.001;
  long flows=0, crossings=0, input_crossings=0;

  void prepare() {
    cached.clear();
    for (size_t r=0;r<(thresholds.size()+1)*(input_thresholds.size()+1);++r) {
      const double* p=coefficients.data()+8*r;
      M4 a{};
      a[0]={p[1],p[2],p[3],p[0]};
      a[1]={p[5],p[6],p[7],p[4]};
      a[2][2]=-1./tau; a[2][3]=baseline/tau;
      cached.push_back(exponential(a,dt));
    }
  }
  V4 flow(int r,const V4& y,double h) {
    ++flows;
    if (std::abs(h-dt)<1e-15) return multiply(cached[r],y);
    int unused=0;
    return advance(coefficients.data()+8*r,y,baseline,tau,h,&unused);
  }
  void step(V4& y) {
    int r=std::upper_bound(thresholds.begin(),thresholds.end(),y[0])-thresholds.begin();
    int band=std::upper_bound(input_thresholds.begin(),input_thresholds.end(),y[2])-input_thresholds.begin();
    if(band>0 && y[2]==input_thresholds[band-1] && baseline<y[2]) --band;
    // At a face select the departing region by the continuous vector field.
    if (r>0 && y[0]==thresholds[r-1]) {
      const double* p=coefficients.data()+8*(band*(thresholds.size()+1)+r);
      if (p[0]+p[1]*y[0]+p[2]*y[1]+p[3]*y[2]<0) --r;
    }
    double left=dt;
    int exits=0;
    while (left>1e-15) {
      double h=left,knot=0.;int input_direction=0;
      if(band>0 && baseline<input_thresholds[band-1]) {
        knot=input_thresholds[band-1];input_direction=-1;
      } else if(band<(int)input_thresholds.size() && baseline>input_thresholds[band]) {
        knot=input_thresholds[band];input_direction=1;
      }
      if(input_direction) {
        const double when=tau*std::log((y[2]-baseline)/(knot-baseline));
        if(when>1e-15 && when<=h) h=when; else input_direction=0;
      }
      const int index=band*(thresholds.size()+1)+r;
      V4 end=flow(index,y,h);
      int direction=0; double boundary=0.;
      if (r<(int)thresholds.size() && end[0]>thresholds[r]) {direction=1;boundary=thresholds[r];}
      else if (r>0 && end[0]<thresholds[r-1]) {direction=-1;boundary=thresholds[r-1];}
      if (!direction) {
        y=end;left-=h;
        if(input_direction) {y[2]=knot;band+=input_direction;++input_crossings;}
        continue;
      }
      double lo=0.,hi=h;
      for (int k=0;k<42 && hi-lo>1e-13;++k) {
        double mid=(lo+hi)*.5;
        if ((flow(index,y,mid)[0]-boundary)*direction>=0) hi=mid; else lo=mid;
      }
      h=(lo+hi)*.5;
      y=flow(index,y,h); y[0]=boundary; left-=h; r+=direction;
      ++crossings;
      if (++exits>100) throw std::runtime_error("PWA crossing loop did not converge");
    }
    for (int d=0;d<3;++d) if (!std::isfinite(y[d])) throw std::runtime_error("Nonfinite PWA state");
  }
};
}
