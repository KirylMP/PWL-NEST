"""Validate all calcium-coordinate variants on common physical protocols."""
import json
import numpy as np
from scipy.special import expit
from scipy.signal import find_peaks
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX
from pwl_nest.calcium_strips import NativeStatistics
from pwl_nest.calcium_coordinates import CalciumTransformPWA,NativeCoordinateDynamic,simulate_coordinates
from pwl_nest.logit_pwa import LogitPWA,simulate_logit
from pwl_nest.protocols import simulate
from fit_strict import protocols,stats
from validate_logit import fresh_protocols as initial_h_protocols

OUT=ROOT/'results/calcium_coordinates';KINDS=['identity','sqrt','cbrt','log']


def new_initial_C(case):
    start,base=(.65,.3) if case=='AM' else (1.15,.45)
    return [dict(name=f'new_initial_C_{c}',initial=[c,.8,start],duration=180.,tau=35.,baseline=base) for c in [.03,.15]]


def validate(case,models=None,training=None,refs=None):
    if models is None:models=[CalciumTransformPWA.load(ROOT/f'models/calcium_coordinates/{kind}/{case}.json') for kind in KINDS]
    if training is None:
        training=protocols(case)
        training.insert(0,dict(name='fixed_lower_rest',initial=[.6 if case=='AM' else 1.3,.8,.3 if case=='AM' else .48],
                              duration=1600.,tau=20.,baseline=.3 if case=='AM' else .48))
    refs={} if refs is None else refs
    historical=protocols(case,heldout=True)+initial_h_protocols(case)
    sets=[('train',training),('historical_diagnostic',historical),('new_diagnostic',new_initial_C(case))]
    baseline_json=json.loads((ROOT/f'results/logit_pwa/{case}.json').read_text())
    baseline={r['name']:r for r in baseline_json['results']}
    baseline_model=LogitPWA.load(ROOT/f'models/logit_pwa/{case}.json')
    # Compute common references once, outside variant evaluation/selection.
    reference_cache={};baseline_cache={}
    for split,protos in sets:
        for p in protos:
            kwargs={k:v for k,v in p.items() if k!='name'};kwargs['dt']=.2 if p['name'].startswith('fixed') else .1
            reference_cache[p['name']]=refs[p['name']] if p['name'] in refs else simulate(case,**kwargs)
            if p['name'] in baseline:baseline_cache[p['name']]=baseline[p['name']]
            else:
                t,y,meta=simulate_logit(baseline_model,**kwargs)
                truth=reference_cache[p['name']][1]
                baseline_cache[p['name']]=dict(completed=True,rmse=np.sqrt(np.mean((y-truth)**2,axis=0)).tolist(),physical=meta['physical'])
    native=NativeStatistics();fast=NativeCoordinateDynamic()
    for model in models:
        kind=model.coordinate.kind;destination=OUT/kind;destination.mkdir(parents=True,exist_ok=True);records=[]
        for split,protos in sets:
            for p in protos:
                print(f'{case} {kind}: {split} {p["name"]}',flush=True)
                kwargs={k:v for k,v in p.items() if k!='name'};kwargs['dt']=.2 if p['name'].startswith('fixed') else .1
                reference=reference_cache[p['name']]
                try:
                    t,y,meta=simulate_coordinates(model,**kwargs)
                    if t.shape!=reference[0].shape or not np.allclose(t,reference[0],atol=1e-12,rtol=0):raise ValueError('Output time mismatch')
                    r=dict(name=p['name'],split=split,completed=True,rmse=np.sqrt(np.mean((y-reference[1])**2,axis=0)).tolist(),
                           max_abs_C=float(abs(y[:,0]-reference[1][:,0]).max()),C_range=[float(y[:,0].min()),float(y[:,0].max())],
                           h_range=[float(y[:,1].min()),float(y[:,1].max())],physical=meta['physical'],IP3_error=meta['ip3_max_abs_error'],
                           logistic_saturation_samples=meta['logistic_saturation_samples'],
                           logit_only_baseline=baseline_cache[p['name']])
                    if p['name'].startswith('fixed'):
                        r['reference'],r['candidate']=stats(reference[0],reference[1]),stats(t,y)
                        obs=native.run(model.latent,p['baseline'],model.to_latent(p['initial']),p['duration'])
                        if obs is not None and obs[4]>model.coordinate.lower:
                            cmn,cmx=map(float,model.coordinate.inverse(obs[:2]))
                            r['native_physical_extrema']=[cmn,cmx]
                            r['native_vs_DOP_extrema_error']=max(abs(cmn-r['candidate']['Cmin']),abs(cmx-r['candidate']['Cmax']))
                            r['native_period']=float(obs[2]) if np.isfinite(obs[2]) else None
                    else:
                        a=find_peaks(reference[1][:,0],prominence=.02)[0];b=find_peaks(y[:,0],prominence=.02)[0]
                        r['peaks']=dict(reference=len(a),candidate=len(b),max_time_error=float(abs(t[a]-t[b]).max()) if len(a)==len(b) and len(a) else None,
                                        prominence_uM=.02,time_grid_step_s=.1)
                        r['jumps']=meta['jumps']
                        np.savez_compressed(destination/f'{case}_{p["name"]}_{split}.npz',t=t,lr=reference[1],candidate=y,latent=meta['latent'])
                        if split=='train':
                            rk=fast.run(model,p['initial'],p['duration'],p['tau'],p['baseline'])
                            rk2=fast.run(model,p['initial'],p['duration'],p['tau'],p['baseline'],maxdt=.005)
                            if rk is None or rk2 is None:raise RuntimeError('RK convergence check failed domain guard')
                            grid=np.linspace(0,p['duration'],361)
                            numeric=np.column_stack([np.interp(grid,t,y[:,d]) for d in [0,1]])
                            mapped=np.column_stack([model.coordinate.inverse(rk[:,0]),expit(rk[:,1]+model.zref)])
                            refined=model.coordinate.inverse(rk2[:,0])
                            r['RK_check']=dict(max_abs_C_vs_DOP=float(abs(mapped[:,0]-numeric[:,0]).max()),
                                               max_abs_C_step_refinement=float(abs(mapped[:,0]-refined).max()),
                                               max_abs_h_vs_DOP=float(abs(mapped[:,1]-numeric[:,1]).max()))
                    records.append(r)
                except (ValueError,RuntimeError,FloatingPointError,OverflowError) as exc:
                    records.append(dict(name=p['name'],split=split,completed=False,error=str(exc)))
        (destination/f'{case}.json').write_text(json.dumps(dict(case=case,kind=kind,metadata=model.metadata,results=records,
            training_protocols=training,historical_diagnostics=historical,new_diagnostics=new_initial_C(case),
            scope='PWA in transformed coordinates, not physical PWA; no full acceptance',
            unrun=['full analytic first-exit solver','unstable cycle continuation','SIC','NEST','global optimum']),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    for case in ['AM','FM']:validate(case)
