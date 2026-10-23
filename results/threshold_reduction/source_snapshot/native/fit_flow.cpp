// Full sampled trajectories for fitting, using the same cached affine stepper as NEST.
#include "pwa_step.h"
extern "C" int fit_flow(const double* thresholds,int regions,const double* knots,int bands,
    const double* coefficients,const double* initial,double baseline,double tau,
    double duration,int outputs,double maxdt,double lower,double upper,double hlower,double hupper,double* out) {
  if(regions<1||bands<0||outputs<2||!(duration>0&&tau>0&&maxdt>0)) return 1;
  try {
    pwla::Stepper s;
    s.thresholds.assign(thresholds,thresholds+regions-1);
    s.input_thresholds.assign(knots,knots+bands);
    s.coefficients.assign(coefficients,coefficients+8*regions*(bands+1));
    const double sample=duration/(outputs-1);
    const int steps=std::max(1,(int)std::ceil(sample/maxdt));
    s.dt=sample/steps;s.baseline=baseline;s.tau=tau;s.prepare();
    V4 y{initial[0],initial[1],initial[2],1.};
    out[0]=y[0];out[1]=y[1];
    for(int row=1;row<outputs;++row) {
      for(int j=0;j<steps;++j) {
        s.step(y);
        if(y[0]<=lower||y[0]>=upper||y[1]<=hlower||y[1]>=hupper) return 2;
      }
      out[2*row]=y[0];out[2*row+1]=y[1];
    }
    return 0;
  } catch(...) {return 3;}
}
