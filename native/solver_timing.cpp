// Isolated kernels: same cached PWA stepper and GSL RKF45 used by NEST LR.
// No spike delivery, SIC, neuron, multimeter or Python per-step calls.
#include "pwa_step.h"
#include <chrono>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv.h>

namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) {return std::chrono::duration<double>(Clock::now()-start).count();}
struct LRParams {double tau,baseline,ker;long rhs=0;};
int lr_rhs(double,const double* y,double* f,void* context) {
  auto& p=*static_cast<LRParams*>(context);++p.rhs;
  const double c=std::clamp(y[0],0.,2.),h=y[1],ip3=y[2];
  const double m=ip3/(ip3+.13),n=c/(c+.08234),q=1.049*(ip3+.13)/(ip3+.9434);
  f[0]=(6*m*m*m*n*n*n*h*h*h+.11)*(2-1.185*c)-.9*c*c/(c*c+p.ker*p.ker);
  f[1]=.2*(q-(q+c)*h);f[2]=-(ip3-p.baseline)/p.tau;
  return GSL_SUCCESS;
}
}
extern "C" int time_solvers(int kind,int population,int steps,double dt,
    const double* initial,const double* thresholds,int regions,const double* coefficients,
    const double* input_thresholds,int input_count,
    double baseline,double tau,double ker,const int* event_steps,const double* deltas,int events,
    double* result) {
  // result: setup seconds, solve seconds, sum terminal states[3], flows/RHS, crossings.
  if(population<1 || steps<1 || !(dt>0&&tau>0) || regions<1) return -1;
  std::fill(result,result+7,0.);
  for(int node=0;node<population;++node) {
    auto start=Clock::now();
    pwla::Stepper pwa;
    LRParams params{tau,baseline,ker};
    gsl_odeiv_step* s=nullptr;gsl_odeiv_control* c=nullptr;gsl_odeiv_evolve* e=nullptr;
    gsl_odeiv_system sys{lr_rhs,nullptr,3,&params};
    if(kind==0) {
      pwa.thresholds.assign(thresholds,thresholds+regions-1);
      pwa.input_thresholds.assign(input_thresholds,input_thresholds+input_count);
      pwa.coefficients.assign(coefficients,coefficients+8*regions*(input_count+1));
      pwa.baseline=baseline;pwa.tau=tau;pwa.dt=dt;pwa.prepare();
    } else {
      s=gsl_odeiv_step_alloc(gsl_odeiv_step_rkf45,3);
      c=gsl_odeiv_control_y_new(1e-3,0.);e=gsl_odeiv_evolve_alloc(3);
    }
    result[0]+=elapsed(start);
    V4 y{initial[0],initial[1],initial[2],1.};
    double integration_step=dt;
    int event=0;
    start=Clock::now();
    for(int step=1;step<=steps;++step) {
      if(kind==0) pwa.step(y);
      else {
        double t=0.;
        while(t<dt) {
          int code=gsl_odeiv_evolve_apply(e,c,s,&sys,&t,dt,&integration_step,y.data());
          if(code!=GSL_SUCCESS) return -2;
        }
        y[0]=std::clamp(y[0],0.,2.);
      }
      if(event<events && event_steps[event]==step) y[2]+=deltas[event++];
    }
    result[1]+=elapsed(start);
    for(int d=0;d<3;++d) result[2+d]+=y[d];
    result[5]+=kind==0?pwa.flows:params.rhs;result[6]+=pwa.crossings;
    if(s) {gsl_odeiv_evolve_free(e);gsl_odeiv_control_free(c);gsl_odeiv_step_free(s);}
  }
  return 0;
}
