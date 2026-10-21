"""Validate the C++ analytic PWA event solver against PWA DOP853 and LR DOP853."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from pwl_nest.calcium_coordinates import CalciumTransformPWA, simulate_coordinates
from pwl_nest.event_solver import EventSolver
from pwl_nest.models import ROOT
from pwl_nest.protocols import simulate as simulate_lr


def protocols():
    rng=np.random.default_rng(20261030)
    times=20.+np.r_[0.,np.cumsum(rng.uniform(16.,30.,7))]
    return [
        dict(name='pulse_tau_20',duration=220.,tau=20.,baseline=.45,jumps=[[20.,.7]]),
        dict(name='pulse_tau_100',duration=300.,tau=100.,baseline=.45,jumps=[[20.,.7]]),
        dict(name='pulse_tau_180',duration=420.,tau=180.,baseline=.45,jumps=[[20.,.7]]),
        dict(name='periodic',duration=240.,tau=20.,baseline=.45,jumps=[[t,.3] for t in [20.,40.,60.,80.,100.]]),
        dict(name='irregular',duration=340.,tau=22.,baseline=.45,
             jumps=list(map(list,zip(times.tolist(),rng.uniform(.2,.3,len(times)).tolist())))),
    ]


def peak_data(t,x):
    ix=find_peaks(x[:,0],prominence=.02)[0]
    return t[ix],x[ix,0]


def compare_pair(t_reference,reference,t_candidate,candidate):
    # C and h remain continuous at IP3 jumps; interpolate onto one common grid.
    keep_ref=np.r_[True,np.diff(t_reference)>1e-12]
    keep_cand=np.r_[True,np.diff(t_candidate)>1e-12]
    tt=t_reference[keep_ref];a=reference[keep_ref]
    tc=t_candidate[keep_cand];xc=candidate[keep_cand]
    b=np.column_stack([np.interp(tt,tc,xc[:,d]) for d in range(3)])
    delta=b-a
    return dict(rmse_C=float(np.sqrt(np.mean(delta[:,0]**2))),
        rmse_h=float(np.sqrt(np.mean(delta[:,1]**2))),
        max_abs_C=float(np.max(np.abs(delta[:,0]))),max_abs_h=float(np.max(np.abs(delta[:,1]))))


def run(args):
    model=CalciumTransformPWA.load(args.model)
    if model.case!='FM':raise ValueError('The selected benchmark model must be FM')
    initial=np.asarray(model.metadata['exact_rest_constraint'],dtype=float)
    solver=EventSolver()
    validation_dt=min(.005,args.output_dt/10.)
    report=dict(model=str(Path(args.model)),protocols=[],solver='C++ 4x4 scaling/squaring exponential; exact exponential I; bisection-localized vertical C exits',
        output_dt=args.output_dt,scan_dt=args.scan_dt,validation_dt=validation_dt,
        validation='same initial LR equilibrium and same IP3 jumps; high-accuracy dense DOP853 for canonical LR and PWA')
    for p in protocols():
        kw={k:v for k,v in p.items() if k!='name'}
        start=time.perf_counter();lr_t,lr_x,_=simulate_lr('FM',initial,**kw,dt=validation_dt,rtol=2e-11,atol=2e-13,max_step=.02)
        lr_seconds=time.perf_counter()-start
        start=time.perf_counter();ref_t,ref_x,_=simulate_coordinates(model,initial,**kw,dt=validation_dt,rtol=2e-11,atol=2e-13,max_step=.02)
        pwa_dop_seconds=time.perf_counter()-start
        start=time.perf_counter();ev_t,ev_x,counters=solver.run(model,initial,**kw,output_dt=args.output_dt,scan_dt=args.scan_dt)
        event_seconds=time.perf_counter()-start
        # Both references are on the same grid, except I's left/right value at jump instants.
        event_against_pwa=compare_pair(ev_t,ev_x,ref_t,ref_x)
        pwa_against_lr=compare_pair(ref_t,ref_x,lr_t,lr_x)
        event_against_lr=compare_pair(ev_t,ev_x,lr_t,lr_x)
        lpt,lph=peak_data(lr_t,lr_x);ept,eph=peak_data(ev_t,ev_x)
        protocol=dict(name=p['name'],duration=p['duration'],tau=p['tau'],jump_count=len(p['jumps']),
            pwa_event_vs_pwa_DOP853=event_against_pwa,pwa_DOP853_vs_canonical_LR=pwa_against_lr,
            pwa_event_vs_canonical_LR=event_against_lr,
            peaks=dict(canonical_count=len(lpt),event_count=len(ept),
                canonical_times=lpt.tolist(),event_times=ept.tolist(),
                canonical_heights=lph.tolist(),event_heights=eph.tolist()),
            event_counters=counters,timing_seconds=dict(canonical_LR_DOP853=lr_seconds,
                PWA_DOP853=pwa_dop_seconds,PWA_event_CPP=event_seconds))
        if p['name']=='irregular':
            fine_scan=max(args.scan_dt/4.,.001)
            start=time.perf_counter();fine_t,fine_x,fine_counters=solver.run(model,initial,**kw,
                output_dt=args.output_dt,scan_dt=fine_scan)
            protocol['scan_refinement']=dict(fine_scan_dt=fine_scan,
                coarse_vs_fine=compare_pair(ev_t,ev_x,fine_t,fine_x),
                fine_counters=fine_counters,seconds=time.perf_counter()-start)
        report['protocols'].append(protocol)
        print(json.dumps(dict(name=p['name'],event_vs_pwa=event_against_pwa,
            pwa_vs_LR=pwa_against_lr,counters=counters,timing=protocol['timing_seconds'])),flush=True)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=str(ROOT/'models/phase_shape_candidates/FM_width_soft_candidate.json'))
    parser.add_argument('--output',default=str(ROOT/'results/nest_event_solver/benchmark.json'))
    parser.add_argument('--output-dt',type=float,default=.05)
    parser.add_argument('--scan-dt',type=float,default=.02)
    args=parser.parse_args();run(args)
