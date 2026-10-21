"""Independent latent-state DOP853 validation with physical readouts."""
import json
import numpy as np
from scipy.signal import find_peaks
from scipy.special import expit
from pwl_nest.models import ROOT
from pwl_nest.calcium_strips import CalciumStrips,NativeStatistics,NativeDynamic
from pwl_nest.logit_pwa import LogitPWA,to_latent,simulate_logit
from pwl_nest.protocols import simulate
from pwl_nest.strict_pwa import C_MAX
from fit_strict import protocols,stats

OUT=ROOT/'results/logit_pwa'


def fresh_protocols(case):
    start,base=(.65,.3) if case=='AM' else (1.15,.45)
    return [dict(name=f'novel_initial_h_{h}',initial=[.08,h,start],duration=180.,tau=35.,baseline=base) for h in [.65,.9]]


def metrics(protocol,reference,trial):
    t,y,meta=trial;truth=reference[1]
    if t.shape!=reference[0].shape or not np.allclose(t,reference[0],atol=1e-12,rtol=0):
        raise ValueError('Reference and candidate output times must agree')
    record=dict(completed=True,rmse=np.sqrt(np.mean((y-truth)**2,axis=0)).tolist(),
                max_abs_C=float(abs(y[:,0]-truth[:,0]).max()),C_range=[float(y[:,0].min()),float(y[:,0].max())],
                h_range=[float(y[:,1].min()),float(y[:,1].max())],
                physical=bool(y[:,0].min()>=-1e-7 and y[:,0].max()<=C_MAX+1e-7 and y[:,1].min()>=0 and y[:,1].max()<=1),
                IP3_error=meta['ip3_max_abs_error'],logistic_saturation_samples=meta.get('logistic_saturation_samples',0))
    if protocol['name'].startswith('fixed'):
        record['reference'],record['candidate']=stats(reference[0],truth),stats(t,y)
    else:
        a=find_peaks(truth[:,0],prominence=.02)[0];b=find_peaks(y[:,0],prominence=.02)[0]
        record['peaks']=dict(reference=len(a),candidate=len(b),max_time_error=float(abs(t[a]-t[b]).max()) if len(a)==len(b) and len(a) else None)
    return record


def validate(case,model=None,control=None,training=None,refs=None):
    OUT.mkdir(exist_ok=True,parents=True)
    if model is None:model=LogitPWA.load(ROOT/f'models/logit_pwa/{case}.json')
    if control is None:control=LogitPWA.load(ROOT/f'models/logit_pwa/{case}_calcium_matched_control.json')
    if training is None:
        training=protocols(case)
        training.insert(0,dict(name='fixed_lower_rest',initial=[.6 if case=='AM' else 1.3,.8,.3 if case=='AM' else .48],
                              duration=1600.,tau=20.,baseline=.3 if case=='AM' else .48))
    refs={} if refs is None else refs
    baseline_file=json.loads((ROOT/f'results/calcium_strips/{case}.json').read_text())
    previous={r['name']:r for r in baseline_file['results']}
    physical_model=CalciumStrips.load(ROOT/f'models/calcium_strips/{case}.json')
    static=NativeStatistics();dynamic=NativeDynamic();results=[]
    for split,protos in [('train',training),('historical_diagnostic',protocols(case,heldout=True)),('new_diagnostic',fresh_protocols(case))]:
        for p in protos:
            print(f'{case}: validate {split} {p["name"]}',flush=True)
            kwargs={k:v for k,v in p.items() if k!='name'}
            kwargs['dt']=.2 if p['name'].startswith('fixed') else .1
            reference=refs.get(p['name']) if split=='train' else None
            if reference is None:reference=simulate(case,**kwargs)
            try:
                trial=simulate_logit(model,**kwargs)
                t,y,meta=trial
                record=dict(name=p['name'],split=split,**metrics(p,reference,trial))
                if p['name'].startswith('fixed'):
                    latent_initial=to_latent(p['initial'])
                    native=static.run(model.latent,p['baseline'],latent_initial,p['duration'])
                    record['native']=None if native is None else [float(v) if np.isfinite(v) else None for v in native]
                    record['native_vs_DOP_extrema_error']=None if native is None else max(abs(native[0]-record['candidate']['Cmin']),abs(native[1]-record['candidate']['Cmax']))
                else:
                    ct,cx,cm=simulate_logit(control,**kwargs)
                    record['calcium_matched_control']=metrics(p,reference,(ct,cx,cm))
                    np.savez_compressed(OUT/f'{case}_{p["name"]}_{split}.npz',t=t,lr=reference[1],candidate=y,
                                        latent=meta['latent'],control=cx,control_latent=cm['latent'])
                    record['jumps']=meta['jumps']
                    if split=='train':
                        fast=dynamic.run(model.latent,to_latent(p['initial']),p['duration'],p['tau'],p['baseline'])
                        finer=dynamic.run(model.latent,to_latent(p['initial']),p['duration'],p['tau'],p['baseline'],maxdt=.005)
                        times=np.linspace(0,p['duration'],361)
                        numerical=np.column_stack([np.interp(times,t,y[:,d]) for d in [0,1]])
                        fastphysical=np.column_stack([fast[:,0],expit(fast[:,1])])
                        record['native_RK_check']=dict(max_abs_C_vs_DOP=float(abs(fastphysical[:,0]-numerical[:,0]).max()),
                            max_abs_h_vs_DOP=float(abs(fastphysical[:,1]-numerical[:,1]).max()),
                            max_abs_C_step_refinement=float(abs(fast[:,0]-finer[:,0]).max()))
                if p['name'] in previous:
                    record['physical_PWA_baseline']=previous[p['name']]
                else:
                    base=simulate(case,**kwargs,model=physical_model,enforce_domain=False,rtol=2e-9,atol=2e-11,max_step=.15)
                    record['physical_PWA_baseline']=metrics(p,reference,base)
                results.append(record)
            except (ValueError,RuntimeError,OverflowError) as exc:
                results.append(dict(name=p['name'],split=split,completed=False,error=str(exc)))
    (OUT/f'{case}.json').write_text(json.dumps(dict(case=case,metadata=model.metadata,results=results,
        training_protocols=training,historical_diagnostics=protocols(case,heldout=True),new_diagnostics=fresh_protocols(case),
        scope='transformed-coordinate branch; physical field not PWA; no full acceptance',
        unrun=['full analytic first-exit solver','unstable cycle continuation','SIC','NEST','global parameter search']),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    for case in ['AM','FM']:validate(case)
