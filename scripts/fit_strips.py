"""Test the original spectral calcium-threshold heuristic in strict 3D PWA."""
import argparse
import json
import os
import time
os.environ.setdefault('MPLCONFIGDIR','/workspace/PWL-NEST/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/PWL-NEST/.cache')
import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.signal import find_peaks
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import lr_field,C_MAX
from pwl_nest.calcium_strips import spectral_seed,decode,NativeStatistics,NativeDynamic
from pwl_nest.protocols import simulate
from fit_strict import stats,protocols

OUT=ROOT/'results/calcium_strips'
MODELS=ROOT/'models/calcium_strips'


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def run(case,budget):
    OUT.mkdir(exist_ok=True,parents=True);MODELS.mkdir(exist_ok=True,parents=True)
    train=protocols(case)
    c0=.6 if case=='AM' else 1.3
    train.insert(0,dict(name='fixed_lower_rest',initial=[c0,.8,.3 if case=='AM' else .48],
                      duration=1600.,tau=20.,baseline=.3 if case=='AM' else .48))
    reference={};points=[]
    for p in train:
        t,y,meta=simulate(case,**{k:v for k,v in p.items() if k!='name'})
        reference[p['name']]=(t,y,meta)
        points.append(y[::max(1,len(y)//300)])
    points=np.vstack(points)
    z0,ms,ks,metadata=spectral_seed(case,points)
    theta=np.array([e['C'] for e in metadata['events']]);n=len(theta)+1
    make=lambda z:decode(case,theta,z,ms,ks,metadata.copy())
    seed=make(z0);seed.save(MODELS/f'{case}_spectral_seed.json')
    static=NativeStatistics();dynamic=NativeDynamic()
    lower=z0.copy();upper=z0.copy()
    lower[:2*n]=np.maximum(-12,z0[:2*n]-4)
    upper[:2*n]=np.minimum(5,z0[:2*n]+4)
    lower[2*n:]=[-4,-10,-5,-7,-5,-5]
    upper[2*n:]=[4,2,4,2,5,5]
    z0=np.clip(z0,lower+1e-8,upper-1e-8)
    truth=lr_field(points,case)
    expected_disc=np.sign(metadata['source_average_trace'])**2*np.array(metadata['source_average_trace'])**2-4*np.array(metadata['source_average_determinant'])
    signs=np.sign(expected_disc)
    def field_residual(z):
        model=make(z)
        rr=((model.field(points)-truth)/np.array([.2,.05])).ravel()/np.sqrt(len(points))
        return np.r_[rr,.02*(z-z0)]
    print(f'{case}: {len(theta)} thresholds, {n} regions, {len(z0)} parameters; preparing field seed',flush=True)
    field=least_squares(field_residual,z0,bounds=(lower,upper),max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    make(field.x).save(MODELS/f'{case}_field_seed.json')
    fixed=[p for p in train if p['name'].startswith('fixed')]
    decays=[p for p in train if p['name'].startswith('decay')]
    targets={p['name']:stats(*reference[p['name']][:2]) for p in fixed}
    sampled={}
    for p in decays:
        t,y,_=reference[p['name']]
        times=np.linspace(0,p['duration'],361)
        sampled[p['name']]=np.column_stack([np.interp(times,t,y[:,d]) for d in [0,1]])
    evaluations=0;best=float('inf');best_z=z0.copy();history=[]
    started=time.monotonic()
    def residual(z):
        nonlocal evaluations,best,best_z
        evaluations+=1
        model=make(z);rr=[]
        for p in fixed:
            target=targets[p['name']]
            obs=static.run(model,p['baseline'],p['initial'],p['duration'])
            if obs is None:
                rr.extend([100.,100.,100.,100.]);continue
            mn,mx,period,ratio,minall,stopped=obs
            rr.extend([(mn-target['Cmin'])/.02,(mx-target['Cmax'])/.04])
            if target['period'] is not None:
                rr.append((period-target['period'])/(1. if case=='AM' else 3.) if np.isfinite(period) else 15.)
            else:rr.append((mx-mn)/.02)
            rr.append(min(0.,minall)/.01)
        for p in decays:
            y=dynamic.run(model,p['initial'],p['duration'],p['tau'],p['baseline'])
            if y is None:rr.extend(np.full(361*2+4,100.));continue
            rr.extend(((y-sampled[p['name']])/np.array([.03,.05])/np.sqrt(361)).ravel())
            rr.extend([min(0.,y[:,0].min())/.01,max(0.,y[:,0].max()-C_MAX)/.05,
                       min(0.,y[:,1].min())/.02,max(0.,y[:,1].max()-1)/.02])
        # Keep source node/focus subtypes as a soft, explicitly measured target.
        mu,k,_,_=model.spectral_coefficients()
        disc=mu*mu-4*k
        rr.extend(2*np.maximum(0,-signs*disc)/(1+abs(expected_disc)))
        rr.extend(.01*(z-z0))
        rr=np.array(rr)
        val=float(rr@rr)
        if val<best:
            best,best_z=val,z.copy()
            if len(history)==0 or evaluations-history[-1]['evaluation']>=20:
                history.append(dict(evaluation=evaluations,loss=val,seconds=time.monotonic()-started))
                print(f'{case}: eval={evaluations}, loss={val:.6g}',flush=True)
        return rr
    initial_scores=[]
    for label,z in [('spectral',z0),('field',field.x)]:
        r=residual(z);initial_scores.append(dict(label=label,loss=float(r@r)))
    # Both starts get a bounded local least-squares attempt; Powell tests nonsmooth basins.
    runs=[]
    for label,z in [('spectral',z0),('field',field.x)]:
        fit=least_squares(residual,z,bounds=(lower,upper),diff_step=2e-4,x_scale='jac',
                           max_nfev=budget,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        residual(fit.x)
        runs.append(dict(start=label,method='least_squares',status=int(fit.status),nfev=int(fit.nfev),message=fit.message))
    result=minimize(lambda z:float(np.dot(r:=residual(z),r)),best_z,method='Powell',
                    bounds=list(zip(lower,upper)),options=dict(maxfev=budget*20,xtol=1e-4,ftol=1e-5))
    residual(result.x)
    runs.append(dict(method='Powell',status=int(result.status),nfev=int(result.nfev),message=str(result.message)))
    selected=make(best_z)
    selected.metadata.update(selected_by='training loss only; holdouts excluded',loss=best,
                              parameters=best_z.tolist(),mu_sign=ms.tolist(),k_sign=ks.tolist(),
                              accepted=False,optimizer_runs=runs)
    selected.save(MODELS/f'{case}.json')
    write(OUT/f'{case}_fit.json',dict(initial_scores=initial_scores,best_loss=best,
        residual_evaluations=evaluations,history=history,runs=runs,training_protocols=train,
        targets=targets,bounds=[lower.tolist(),upper.tolist()],seed_parameters=z0.tolist()))
    # All finalist traces checked with an independent high-accuracy solver.
    report=[]
    for split,protos in [('train',train),('holdout',protocols(case,heldout=True))]:
        for p in protos:
            print(f'{case}: validate {split} {p["name"]}',flush=True)
            kwargs={k:v for k,v in p.items() if k!='name'}
            ref=reference[p['name']] if split=='train' else simulate(case,**kwargs)
            try:
                t,y,meta=simulate(case,**kwargs,model=selected,enforce_domain=False,rtol=2e-9,atol=2e-11,max_step=.15)
                if not np.isfinite(y).all() or abs(y).max()>100:
                    raise RuntimeError('Unbounded candidate trajectory')
                record=dict(name=p['name'],split=split,completed=True,
                     rmse=np.sqrt(np.mean((y-ref[1])**2,axis=0)).tolist(),max_abs_C=float(np.max(abs(y[:,0]-ref[1][:,0]))),
                     C_range=[float(y[:,0].min()),float(y[:,0].max())],h_range=[float(y[:,1].min()),float(y[:,1].max())],
                     physical=bool(y[:,0].min()>=-1e-7 and y[:,0].max()<=C_MAX+1e-7 and y[:,1].min()>=-1e-7 and y[:,1].max()<=1+1e-7),
                     IP3_error=meta['ip3_max_abs_error'],jumps=meta['jumps'])
                if p['name'].startswith('fixed'):
                    record['reference'],record['candidate']=stats(ref[0],ref[1]),stats(t,y)
                    record['equilibria']=selected.equilibria(p['baseline'])
                    native=static.run(selected,p['baseline'],p['initial'],p['duration'])
                    record['native']=None if native is None else [float(x) if np.isfinite(x) else None for x in native]
                else:
                    a=find_peaks(ref[1][:,0],prominence=.02)[0];b=find_peaks(y[:,0],prominence=.02)[0]
                    record['peaks']=dict(reference=len(a),candidate=len(b),
                        max_time_error=float(np.max(abs(t[a]-t[b]))) if len(a)==len(b) and len(a)>0 else None)
                    np.savez_compressed(OUT/f'{case}_{p["name"]}_{split}.npz',t=t,lr=ref[1],candidate=y)
                report.append(record)
            except (ValueError,RuntimeError,OverflowError) as exc:
                report.append(dict(name=p['name'],split=split,completed=False,error=str(exc)))
    write(OUT/f'{case}.json',dict(case=case,model_metadata=selected.metadata,protocols=train+protocols(case,heldout=True),results=report,
          unrun=['unstable cycle continuation','SIC','NEST','full analytic first-exit solver','global parameter search'],
          note='AM .25 threshold excluded: only original source spectral thresholds used. No model accepted by this pilot.'))
    print(f'{case}: complete, training loss={best:.6g}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['AM','FM','both'],default='both')
    parser.add_argument('--budget',type=int,default=100)
    args=parser.parse_args()
    for case in (['AM','FM'] if args.case=='both' else [args.case]):run(case,args.budget)
