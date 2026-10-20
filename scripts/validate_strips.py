"""Independent physical-state validation of persisted compact candidates."""
import json
import numpy as np
from scipy.signal import find_peaks
from pwl_nest.models import ROOT
from pwl_nest.calcium_strips import CalciumStrips,NativeStatistics,NativeDynamic
from pwl_nest.strict_pwa import C_MAX
from pwl_nest.protocols import simulate
from fit_strict import protocols,stats

OUT=ROOT/'results/calcium_strips'


def validate(case,selected,training=None,references=None):
    if training is None:
        training=protocols(case)
        training.insert(0,dict(name='fixed_lower_rest',initial=[.6 if case=='AM' else 1.3,.8,.3 if case=='AM' else .48],
                              duration=1600.,tau=20.,baseline=.3 if case=='AM' else .48))
    references={} if references is None else references
    report=[];static=NativeStatistics();dynamic=NativeDynamic()
    for split,protos in [('train',training),('holdout',protocols(case,heldout=True))]:
        for p in protos:
            print(f'{case}: validate {split} {p["name"]}',flush=True)
            kwargs={k:v for k,v in p.items() if k!='name'}
            if not p['name'].startswith('fixed'):
                # 0.1 s output aligns with the 0.5 s fitting samples. Avoid
                # measuring interpolation error across sharp peaks as RK error.
                kwargs['dt']=.1
            ref=references.get(p['name']) if split=='train' else None
            if not p['name'].startswith('fixed'):
                ref=None
            if ref is None:ref=simulate(case,**kwargs)
            try:
                t,y,meta=simulate(case,**kwargs,model=selected,enforce_domain=False,rtol=2e-9,atol=2e-11,max_step=.15)
                if not np.isfinite(y).all() or abs(y).max()>100:raise RuntimeError('Unbounded trajectory')
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
                    if native is not None:
                        record['native_vs_DOP_extrema_error']=max(abs(native[0]-record['candidate']['Cmin']),abs(native[1]-record['candidate']['Cmax']))
                else:
                    a=find_peaks(ref[1][:,0],prominence=.02)[0];b=find_peaks(y[:,0],prominence=.02)[0]
                    record['peaks']=dict(reference=len(a),candidate=len(b),
                        max_time_error=float(np.max(abs(t[a]-t[b]))) if len(a)==len(b) and len(a)>0 else None)
                    np.savez_compressed(OUT/f'{case}_{p["name"]}_{split}.npz',t=t,lr=ref[1],candidate=y)
                    if split=='train':
                        rk=dynamic.run(selected,p['initial'],p['duration'],p['tau'],p['baseline'])
                        rk2=dynamic.run(selected,p['initial'],p['duration'],p['tau'],p['baseline'],maxdt=.005)
                        checktimes=np.linspace(0,p['duration'],361)
                        dop=np.column_stack([np.interp(checktimes,t,y[:,d]) for d in [0,1]])
                        record['native_RK_checks']=dict(max_abs_C_vs_DOP=float(abs(rk[:,0]-dop[:,0]).max()),
                                                       max_abs_C_step_refinement=float(abs(rk[:,0]-rk2[:,0]).max()))
                report.append(record)
            except (ValueError,RuntimeError,OverflowError) as exc:
                report.append(dict(name=p['name'],split=split,completed=False,error=str(exc)))
    (OUT/f'{case}.json').write_text(json.dumps(dict(case=case,model_metadata=selected.metadata,
        protocols=training+protocols(case,heldout=True),results=report,
        note='v2 same thresholds, hard spectral subtypes, staged training. Historical holdouts were inspected in v1; no claim of blind external validation.',
        unrun=['unstable cycle continuation','SIC','NEST','full analytic first-exit solver','global parameter search']),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    for case in ['AM','FM']:validate(case,CalciumStrips.load(ROOT/f'models/calcium_strips/{case}.json'))
