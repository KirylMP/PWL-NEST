"""Independent DOP853 audit; selection uses calibration, never fresh tests."""
import argparse, json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX, regional_flow
from pwl_nest.calcium_coordinates import CalciumTransformPWA,NativeCoordinateDynamic,simulate_coordinates
from pwl_nest.protocols import simulate
from fit_positive_coordinates import OUT,MODELS,VARIANTS,CalibrationResidual,is_fixed
from fit_strict import protocols,stats
from validate_logit import fresh_protocols as initial_h_protocols
from validate_calcium_coordinates import new_initial_C
from fit_calcium_coordinates import dump


def structure(model):
    co,gate=model.coordinate,model.gate;n=model.latent.region_count
    local=model.geometry_coefficients();edges=np.r_[0,model.input_thresholds,1.3]
    continuity=0.;flow_error=0.;readout_error=0.;types=[];reduction=0.
    np.testing.assert_allclose(co.forward(model.metadata['physical_thresholds_C']),model.latent.thresholds,rtol=1e-12,atol=1e-24)
    for s in range(len(edges)-1):
        ip3=(edges[s]+edges[s+1])/2
        limits=np.r_[co.forward(co.reference*.9),model.latent.thresholds,co.forward(C_MAX*.9)]
        for j,theta in enumerate(model.latent.thresholds):
            for h in [.2,.8,.95]:
                y=np.array([1.,theta,gate.forward(h),ip3])
                continuity=max(continuity,float(abs(local[s,j]@y-local[s,j+1]@y).max()))
        labels=[]
        for j in range(n):
            y=np.array([(limits[j]+limits[j+1])/2,gate.forward(.75),ip3]);a,b,_=model.region(y,20.,.2)
            margin=(limits[j+1]-limits[j])/2
            velocity=a@y+b
            dt=min(1e-4,.01*margin/(1+abs(velocity[0])),1e-5/(1+np.linalg.norm(velocity,np.inf)),
                1e-3/(1+np.linalg.norm(a,np.inf)))
            exact=regional_flow(a,b,y,dt)
            numeric=solve_ivp(model.rhs(20.,.2),(0,dt),y,rtol=2e-12,atol=1e-25)
            flow_error=max(flow_error,float(abs(exact-numeric.y[:,-1]).max()))
            readout_error=max(readout_error,float(abs(model.to_physical(exact)-model.to_physical(numeric.y[:,-1])).max()))
            reduction=max(reduction,float(abs(model.field(y)-model.at_input(ip3).latent.field(y)).max()))
            tr,det=np.trace(a[:2,:2]),np.linalg.det(a[:2,:2])
            labels.append('saddle' if det<0 else ('stable_' if tr<0 else 'unstable_')+('focus' if tr*tr<4*det else 'node'))
        types.append(labels)
    for s,knot in enumerate(model.input_thresholds):
        for c in model.metadata['physical_thresholds_C']:
            y=np.r_[1,model.to_latent([c,.75,knot])]
            for j in range(n):continuity=max(continuity,float(abs((local[s,j]-local[s+1,j])@y).max()))
    passed=(continuity<1e-8 and flow_error<1e-8 and readout_error<1e-7 and reduction<1e-8
        and all(labels==model.metadata['target_types'] for labels in types))
    return dict(passed=passed,continuity_error=continuity,regional_flow_error=flow_error,
        regional_readout_error=readout_error,constant_input_reduction_error=reduction,
        types=types,regions=n*(len(model.input_thresholds)+1),active_parameters=model.metadata['active_fit_parameters'],
        minimum_latent_C_threshold_gap=float(np.diff(model.latent.thresholds).min()),
        C_positive_branch_lower=co.lower if np.isfinite(co.lower) else None,
        no_clipping=True)


def audit(case,variants=None):
    specification=json.loads((OUT/f'{case}_protocols.json').read_text());training=specification['calibration']
    by_name={p['name']:p for p in training}
    historical=protocols(case,heldout=True)+initial_h_protocols(case)+new_initial_C(case)
    for p in historical+specification['fresh_validation']:by_name.setdefault(p['name'],p)
    fresh={p['name'] for p in specification['fresh_validation']};calibrated={p['name'] for p in training}
    refs={};refdir=OUT/'references';refdir.mkdir(exist_ok=True)
    for name,p in by_name.items():
        print(f'{case}: source {name}',flush=True)
        path=refdir/f'{case}_{name}.npz'
        if path.exists():
            with np.load(path) as data:refs[name]=(data['t'],data['x'],{})
        else:
            # Keep the original calibration references for the seven old protocols.
            strict=name not in {p['name'] for p in training[:7]}
            kwargs=dict(rtol=2e-9,atol=2e-11,max_step=.15) if strict else {}
            refs[name]=simulate(case,**{k:v for k,v in p.items() if k!='name'},dt=.25 if not is_fixed(p) else .2,**kwargs)
            np.savez_compressed(path,t=refs[name][0],x=refs[name][1])
    # Reconstruct original calibration references at their original sample grids.
    calrefs={}
    for p in training:
        old=p['name'] in {p['name'] for p in training[:7]}
        kwargs={} if old else dict(rtol=2e-7,atol=2e-9,max_step=.5)
        calrefs[p['name']]=simulate(case,**{k:v for k,v in p.items() if k!='name'},dt=.1 if not is_fixed(p) else .2,**kwargs)
    seed=CalciumTransformPWA.load(MODELS/'baseline'/f'{case}.json')
    objective=CalibrationResidual(case,training,calrefs,seed);scores={};structures={};fast=NativeCoordinateDynamic()
    for variant in ['baseline']+VARIANTS:
        if not (MODELS/variant/f'{case}.json').exists():continue
        model=CalciumTransformPWA.load(MODELS/variant/f'{case}.json');destination=OUT/variant;destination.mkdir(exist_ok=True)
        try:structures[variant]=structure(model)
        except (ValueError,RuntimeError,FloatingPointError,OverflowError) as exc:
            structures[variant]=dict(passed=False,error=str(exc))
        records=[];replay={};failed_calibration=[]
        if variants is not None and variant not in variants:
            report_path=destination/f'{case}.json'
            if report_path.exists():scores[variant]=json.loads(report_path.read_text())['verified_calibration_loss']
            continue
        # Small absolute tolerances are essential for the C^6 coordinate near rest.
        atol=np.array([min(2e-12,float(model.coordinate.derivative(.005))*2e-12),
            min(2e-12,float(model.gate.derivative(.03))*2e-12),2e-12])
        for name,p in by_name.items():
            split='calibration' if name in calibrated else 'fresh_validation' if name in fresh else 'historical_diagnostic'
            print(f'{case} {variant}: {split} {name}',flush=True)
            try:
                kwargs={k:v for k,v in p.items() if k!='name'}
                trial=simulate_coordinates(model,**kwargs,dt=.25 if not is_fixed(p) else .2,
                    rtol=2e-10,atol=atol,max_step=.5 if is_fixed(p) else .15)
                t,x,meta=trial;ref=refs[name]
                truth=np.column_stack([np.interp(t,ref[0],ref[1][:,d]) for d in range(3)])
                r=dict(name=name,split=split,completed=True,physical=meta['physical'],
                    C_range=[float(x[:,0].min()),float(x[:,0].max())],h_range=[float(x[:,1].min()),float(x[:,1].max())],
                    rmse=np.sqrt(np.mean((x-truth)**2,axis=0)).tolist(),ip3_error=meta['ip3_max_abs_error'])
                if is_fixed(p):r.update(reference=stats(t,truth),candidate=stats(t,x))
                else:
                    a=find_peaks(truth[:,0],prominence=.02)[0];b=find_peaks(x[:,0],prominence=.02)[0]
                    r['peaks']=dict(reference=len(a),candidate=len(b),
                        max_time_error=float(abs(t[a]-t[b]).max()) if len(a)==len(b) and len(a) else None)
                if name in calibrated:
                    replay[name]=trial
                    if not is_fixed(p):
                        grid=np.linspace(0,p['duration'],721);expected=np.interp(grid,t,x[:,0]);errors={}
                        for step in [.005,.001]:
                            y=fast.run(model,p['initial'],p['duration'],p['tau'],p['baseline'],outputs=721,maxdt=step)
                            try:errors[str(step)]=None if y is None else float(abs(model.coordinate.inverse(y[:,0])-expected).max())
                            except (ValueError,FloatingPointError):errors[str(step)]=None
                        r['RK_C_errors_vs_DOP']=errors
                if name in calibrated or not is_fixed(p):
                    np.savez_compressed(destination/f'{case}_{name}.npz',t=t,lr=truth,candidate=x,latent=meta['latent'])
                records.append(r)
            except (ValueError,RuntimeError,FloatingPointError,OverflowError) as exc:
                records.append(dict(name=name,split=split,completed=False,error=str(exc)))
                if name in calibrated:failed_calibration.append(name)
        if failed_calibration:
            score=None;parts=None
        else:
            r,parts=objective(model,replay=replay,parts=True);score=float(r@r)
        coarse=objective(model)
        scores[variant]=score
        model.metadata.update(verified_calibration_loss=score,verified_calibration_method='all calibration paths DOP853; fresh tests excluded')
        model.save(MODELS/variant/f'{case}.json')
        dump(destination/f'{case}.json',dict(case=case,variant=variant,metadata=model.metadata,
            results=records,verified_calibration_loss=score,optimizer_recomputed_loss=float(coarse@coarse),
            calibration_parts=parts,failed_calibration=failed_calibration,DOP_atol=atol.tolist(),
            all_candidates_unaccepted=True,full_event_solver_implemented=False))
    eligible={v:s for v,s in scores.items() if s is not None and structures[v]['passed']}
    if not eligible:raise RuntimeError('No numerically verified candidate')
    selected=min(eligible,key=eligible.get)
    CalciumTransformPWA.load(MODELS/selected/f'{case}.json').save(MODELS/f'selected_{case}.json')
    dump(OUT/f'{case}_selection.json',dict(selected=selected,verified_calibration_scores=scores,
        selection='lowest independently verified new calibration loss among structural passes; fresh validation excluded',
        variants=['baseline']+VARIANTS,accepted=False,structures=structures))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['AM','FM','both'],default='both')
    parser.add_argument('--variants',nargs='+');args=parser.parse_args()
    for case in (['AM','FM'] if args.case=='both' else [args.case]):audit(case,args.variants)
