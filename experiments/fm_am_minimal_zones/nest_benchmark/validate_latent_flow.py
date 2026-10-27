#!/usr/bin/env python3
"""Compare standalone latent C++ flow with DOP853, including IP3 jumps.

This validates the building block only; it is not a NEST speed benchmark.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from latent_flow import Candidate,NativeFlow,ROOT


def reference(model,initial,baseline,tau,duration,jumps,t):
    y=np.array([model.phi(initial[0]),initial[1],initial[2]],float)
    boundaries=[(0.,0.)]+list(jumps)+[(duration,0.)]
    out=np.empty((len(t),3)); assigned=np.zeros(len(t),bool)
    for j in range(len(boundaries)-1):
        start=boundaries[j][0];stop=boundaries[j+1][0]
        sol=solve_ivp(lambda time,state:model.rhs(time,state,baseline,tau),(start,stop),y,
                      method='DOP853',rtol=1e-11,atol=1e-13,max_step=.03,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        select=(t>=start-1e-11)&((t<stop-1e-11) if j<len(boundaries)-2 else (t<=stop+1e-11))
        out[select]=sol.sol(t[select]).T;assigned[select]=True
        y=sol.y[:,-1].copy();y[2]+=boundaries[j+1][1]
    if not assigned.all():raise RuntimeError('Reference did not cover all samples')
    out[:,0]=model.inverse(out[:,0]);return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--am',type=Path,default=ROOT/'experiments/fm_am_minimal_zones/am_two_thresholds/fits/mixed_power_fixed/candidate.json')
    parser.add_argument('--fm',type=Path,default=ROOT/'experiments/fm_am_minimal_zones/fm_four_zone_fit/candidate.json')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results'/'latent_flow_validation.json')
    args=parser.parse_args();flow=NativeFlow();report=[]
    for case,path,base in [('AM',args.am,.4),('FM',args.fm,.55)]:
        model=Candidate(path);initial=[.1,.75,base];duration=120.;tau=30.;jumps=[(10.,.2),(40.,.15),(70.,.15)]
        start=time.perf_counter();t,y,counts=flow.simulate(model,initial,base,tau,duration,jumps,output_dt=.05,scan_dt=.01);native_seconds=time.perf_counter()-start
        start=time.perf_counter();ref=reference(model,initial,base,tau,duration,jumps,t);reference_seconds=time.perf_counter()-start
        err=np.abs(y-ref);row=dict(case=case,candidate=str(path),parameters=dict(duration_s=duration,tau_s=tau,baseline=base,jumps=jumps,output_dt_s=.05,scan_dt_s=.01),max_abs_error_C_h_I=err.max(axis=0).tolist(),rmse_C_h_I=np.sqrt(np.mean(err**2,axis=0)).tolist(),counters=counts,timing_seconds=dict(native_including_inverse=native_seconds,python_scipy_reference=reference_seconds))
        report.append(row);print(json.dumps(row),flush=True)
        if counts['input_jumps']!=len(jumps) or np.max(err[:,0])>2e-4 or np.max(err[:,1])>2e-4 or np.max(err[:,2])>1e-8:raise RuntimeError(f'{case}: validation tolerance exceeded')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(dict(scope='standalone PWA kernel validation; not a NEST timing comparison',results=report),indent=2)+'\n')
    print(f'Saved to {args.output}',flush=True)

if __name__=='__main__':main()
