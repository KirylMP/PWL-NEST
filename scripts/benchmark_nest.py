"""Matched spiking networks, delivered SIC and solver-only timings in NEST 3.10."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nest
import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks

from pwl_nest.models import ROOT
from pwl_nest.nest_adapter import build_network, load_candidate, sic
from pwl_nest.strict_pwa import lr_field, equilibrium_h


def initial_state(case, baseline):
    c=brentq(lambda c: lr_field([c,equilibrium_h(c,baseline),baseline],case)[0],1e-6,.2)
    return [c,float(equilibrium_h(c,baseline)),baseline]


def protocols(case):
    baseline=.2 if case=='AM' else .45
    rng=np.random.default_rng(20261021)
    ts=np.round((20.+np.r_[0.,np.cumsum(rng.uniform(16,30,7))])/.05)*.05
    return [dict(name='pulse_tau20',duration=220.,tau=20.,baseline=baseline,jumps=[[20.,.7]]),
            dict(name='pulse_tau100',duration=300.,tau=100.,baseline=baseline,jumps=[[20.,.7]]),
            dict(name='pulse_tau180',duration=420.,tau=180.,baseline=baseline,jumps=[[20.,.7]]),
            dict(name='periodic',duration=240.,tau=20.,baseline=baseline,
                 jumps=[[t,.3] for t in [20.,40.,60.,80.,100.]]),
            dict(name='irregular',duration=340.,tau=22.,baseline=baseline,
                 jumps=list(map(list,zip(ts.tolist(),rng.uniform(.2,.3,len(ts)).tolist()))))]


def network_run(case,model,initial,protocol,kind,resolution,population=1,
                record=True,with_targets=True,threads=1,delay_ms=None):
    start=time.perf_counter()
    a,targets,meters=build_network(case,model,initial,protocol,kind=kind,population=population,
        resolution_ms=resolution,record_interval_ms=resolution,record=record,
        with_targets=with_targets,threads=threads,delay_ms=delay_ms)
    setup=time.perf_counter()-start
    start=time.perf_counter();nest.Prepare();prepare=time.perf_counter()-start
    try:
        start=time.perf_counter();nest.Run(protocol['duration']*1000.);run=time.perf_counter()-start
    finally:
        nest.Cleanup()
    start=time.perf_counter()
    data={}
    if record:
        ev=meters[0].events
        data=dict(t=ev['times']/1000.,C=ev['Ca_astro'],h=ev['h_IP3R'],IP3=ev['IP3'])
        if targets is not None:
            ev=meters[1].events
            data.update(I_SIC=ev['I_SIC'],V_m=ev['V_m'])
    readout=time.perf_counter()-start
    terminal={f:float(np.asarray(a.get(f)).flat[0]) for f in ['Ca_astro','h_IP3R','IP3']}
    counters={f:int(np.sum(a.get(f))) for f in ['affine_flows','region_crossings','input_spikes']} if kind=='pwa' else {}
    return data,dict(build_seconds=setup,prepare_seconds=prepare,run_seconds=run,
                     readout_seconds=readout,terminal=terminal,counters=counters)


def compare(a,b,dt):
    result={}
    for key in ['C','h','IP3','I_SIC']:
        diff=b[key]-a[key]
        result[key]=dict(rmse=float(np.sqrt(np.mean(diff**2))),max_abs=float(np.max(np.abs(diff))))
    for name,x in [('lr',a),('pwa',b)]:
        peak=find_peaks(x['C'],prominence=.02)[0]
        sic_peak=find_peaks(x['I_SIC'],prominence=.02)[0]
        result[name]=dict(calcium_peaks=len(peak),calcium_peak_times=x['t'][peak].tolist(),
            sic_peak_count=len(sic_peak),sic_peak_times=x['t'][sic_peak].tolist(),
            sic_area=float(np.trapezoid(x['I_SIC'],x['t'])),
            sic_max=float(np.max(x['I_SIC'])),
            sic_active_seconds=float(np.sum(x['I_SIC']>0)*dt),
            C_range=[float(x['C'].min()),float(x['C'].max())],
            h_range=[float(x['h'].min()),float(x['h'].max())],
            sic_delivery_max_error=float(np.max(np.abs(x['I_SIC'][1:]-sic(x['C'][:-1])))))
    ap=result['lr']['calcium_peak_times'];bp=result['pwa']['calcium_peak_times']
    result['matched_count_peak_time_MAE_seconds']=float(np.mean(np.abs(np.array(ap)-bp))) if len(ap)==len(bp) and ap else None
    return result


def isolated(case,model,initial,p,resolution,population):
    lib=ct.CDLL(str(ROOT/'.build/solver_timing.so'))
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    iptr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    fn=lib.time_solvers
    fn.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.c_double,ptr,ptr,ct.c_int,ptr,ptr,ct.c_int,
                 ct.c_double,ct.c_double,ct.c_double,iptr,ptr,ct.c_int,ptr]
    fn.restype=ct.c_int
    dt=resolution/1000.
    steps=np.asarray([round(t/dt) for t,_ in p['jumps']],dtype=np.int32)
    deltas=np.asarray([d for _,d in p['jumps']],dtype=np.float64)
    values={}
    for kind in ['lr','pwa']:
        y=np.asarray(initial if kind=='lr' else model.to_latent(initial),dtype=np.float64)
        out=np.zeros(7)
        code=fn(int(kind=='lr'),population,round(p['duration']/dt),dt,y,
                np.ascontiguousarray(model.latent.thresholds),model.latent.region_count,
                np.ascontiguousarray(model.geometry_coefficients().ravel()),
                np.ascontiguousarray(model.input_thresholds),len(model.input_thresholds),
                p['baseline'],p['tau'],.1 if case=='AM' else .05,steps,deltas,len(steps),out)
        if code: raise RuntimeError(f'Isolated solver failed: {code}')
        terminal=out[2:5]/population
        if kind=='pwa':terminal=model.to_physical(terminal)
        values[kind]=dict(setup_seconds=float(out[0]),solve_seconds=float(out[1]),
                          terminal=terminal.tolist(),rhs_or_flows=int(out[5]),crossings=int(out[6]))
    return values


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        nest_version=nest.__version__,platform=platform.platform(),cpu=platform.processor(),
        threads=args.threads,resolution_ms=args.resolution_ms,repeats=args.repeats,populations=args.populations,
        solver='Cached regional 4x4 matrix exponential, bisection at vertical exits; GSL RKF45 baseline (NEST default abs tol 1e-3)',
        timing_scope='Run excludes build, Prepare and readout. Isolated solve excludes allocation and matrix-cache preparation; includes the deterministic IP3 jump check.',
        cases={})
    for case in ['AM','FM']:
        model,identity=load_candidate(case)
        specs=protocols(case);initial=initial_state(case,specs[0]['baseline'])
        record=dict(model=identity,initial=initial,validation=[],timing=[])
        traces={}
        for p in specs:
            traces[p['name']]={};times={}
            for kind in ['lr','pwa']:
                data,timing=network_run(case,model,initial,p,kind,args.resolution_ms,threads=args.threads)
                traces[p['name']][kind]=data;times[kind]=timing
                np.savez_compressed(out/f'{case}_{p["name"]}_{kind}.npz',**data)
            metrics=compare(traces[p['name']]['lr'],traces[p['name']]['pwa'],args.resolution_ms/1000.)
            record['validation'].append(dict(protocol=p,metrics=metrics,timing=times))
            print(json.dumps(dict(case=case,protocol=p['name'],SIC_RMSE=metrics['I_SIC']['rmse'],
                                 peaks=[metrics[k]['calcium_peaks'] for k in ['lr','pwa']])),flush=True)
        # Fine-grid check separates tick/bracket errors from PWA approximation.
        p=specs[0];fine={}
        for kind in ['lr','pwa']:
            data,_=network_run(case,model,initial,p,kind,args.resolution_ms/2,threads=args.threads,
                               delay_ms=args.resolution_ms)
            coarse=traces[p['name']][kind]
            fine[kind]={k:float(np.max(np.abs(coarse[k]-np.interp(coarse['t'],data['t'],data[k]))))
                        for k in ['C','h','I_SIC']}
        record['half_tick_max_difference']=fine
        # Alternate order between repetitions; no multimeters in primary timings.
        for population in args.populations:
            for mode in ['astrocytes_only','network_with_SIC']:
                rows={k:[] for k in ['lr','pwa']}
                p=dict(specs[0],duration=args.timing_duration)
                for repeat in range(args.repeats+1):
                    order=['lr','pwa'] if repeat%2==0 else ['pwa','lr']
                    for kind in order:
                        _,timing=network_run(case,model,initial,p,kind,args.resolution_ms,population,
                            record=False,with_targets=mode=='network_with_SIC',threads=args.threads)
                        if repeat:rows[kind].append(timing)
                med={k:statistics.median(r['run_seconds'] for r in rows[k]) for k in rows}
                record['timing'].append(dict(population=population,mode=mode,raw=rows,
                    median_run_seconds=med,lr_over_pwa=med['lr']/med['pwa']))
                print(json.dumps(dict(case=case,population=population,mode=mode,seconds=med,
                                      lr_over_pwa=med['lr']/med['pwa'])),flush=True)
            kernels=[isolated(case,model,initial,p,args.resolution_ms,population) for _ in range(args.repeats)]
            med={k:statistics.median(r[k]['solve_seconds'] for r in kernels) for k in ['lr','pwa']}
            record['timing'].append(dict(population=population,mode='isolated_solver',raw=kernels,
                median_run_seconds=med,lr_over_pwa=med['lr']/med['pwa']))
            # Terminal states prove both isolated kernels really executed the same protocol.
            for kind in ['lr','pwa']:
                target=rows[kind][-1]['terminal']
                expected=np.array([target[f] for f in ['Ca_astro','h_IP3R','IP3']])
                np.testing.assert_allclose(kernels[-1][kind]['terminal'],expected,rtol=0,atol=2e-8)
        fig,axes=plt.subplots(2,3,figsize=(13,6),sharex='col')
        for col,name in enumerate(['pulse_tau20','pulse_tau100','irregular']):
            for kind,color in [('lr','black'),('pwa','tab:orange')]:
                d=traces[name][kind];stride=max(1,round(.05/(args.resolution_ms/1000)))
                axes[0,col].plot(d['t'][::stride],d['C'][::stride],color=color,label=kind,lw=1)
                axes[1,col].plot(d['t'][::stride],d['I_SIC'][::stride],color=color,label=kind,lw=1)
            axes[0,col].set_title(name);axes[1,col].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Ca (µM)');axes[1,0].set_ylabel('Delivered SIC (pA)')
        axes[0,0].legend();fig.suptitle(f'{case}: same spikes → astrocyte → sic_connection → AEIF')
        fig.tight_layout();fig.savefig(out/f'{case}_transfer.png',dpi=160);fig.savefig(out/f'{case}_transfer.pdf');plt.close(fig)
        report['cases'][case]=record
        (out/'benchmark.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(out/'benchmark.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=str(ROOT/'results/nest_benchmark'))
    p.add_argument('--resolution-ms',type=float,default=10.)
    p.add_argument('--populations',type=int,nargs='+',default=[1,64,256])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--timing-duration',type=float,default=120.)
    p.add_argument('--threads',type=int,default=1)
    args=p.parse_args()
    if args.repeats<1 or args.resolution_ms<=0 or args.timing_duration<=20 or min(args.populations)<1:
        p.error('Positive repeats/populations/resolution and timing-duration >20 s required')
    run(args)
