#!/usr/bin/env python3
"""Fit one nested FM construction. See README for stages and limitations."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares,minimize
from model import Model,reference as f,BASE,EVENT_I
from fold import BoundaryFold,FoldError
from equilibrium_loss import model_cost
from loss_weights import closed_components
import closed_fit
import cycle_count_loss

ROOT=Path(__file__).resolve().parent


def geometry(model,x):
    m=model.unpack(x);C,I,w,y,ce,ie,he,sp=f.DATA
    b=model.basis(C,I,m);geo=np.stack([b@m['N'],b@m['H']],axis=-1)-y
    be=model.basis(ce,ie,m);er=np.stack([be@m['N']-he,be@m['H']-he],axis=-1)
    regions=np.searchsorted(model.cuts,ce)
    sr=np.column_stack([m['T'][regions]-sp[:,0],(m['D'][regions]-sp[:,1])/.15])
    heights=model.event_heights(x)
    domain=1000*np.r_[np.maximum(-heights,0),np.maximum(heights-1,0)]
    events=(model.basis(BASE,0,m)@(m['N']-m['H']))/(m['H'][2]-m['N'][2])
    ev=(events-[.52907,.482304,.860021])/.04
    return np.r_[(geo*w[:,:,None]/.025).ravel(),(er/.04/np.sqrt(len(ce))).ravel(),.35*sr.ravel()/np.sqrt(len(ce)),2*ev,domain]


def install_model(model):
    # Reuse the exact existing cycle loss and finite-grid count machinery in
    # this isolated process; no existing fitting files are mutated.
    f.unpack=model.unpack;f.basis=model.basis;f.equilibrium=model.equilibrium


def json_numpy(value):
    """Convert NumPy values nested in optimization and audit records."""
    if isinstance(value, np.generic):return value.item()
    if isinstance(value, np.ndarray):return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def save(path,model,x,extra):
    m=model.unpack(x)
    if not model.physically_valid_events(x):raise ValueError('Nonphysical fixed-event height')
    obj=dict(variant=model.variant,parameters=x.tolist(),physical_thresholds_C=model.cuts.tolist(),
        fixed_event_threshold_indices=model.fixed,fixed_equilibrium_threshold_I=EVENT_I.tolist(),
        fixed_event_heights=model.event_heights(x).tolist(),regional_types=[['stable_node','saddle','unstable_focus','stable_focus'][k] for k in model.kind],
        alpha=m['N'].tolist(),beta=m['H'].tolist(),a=m['a'],b=m['b'],
        transform=dict(zip(['p','e','n','k'],m['shape'])),trace=m['T'].tolist(),determinant=m['D'].tolist(),
        derived_parameter_indices=extra.pop('derived_parameter_indices',model.derived),**extra)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=json_numpy));tmp.replace(path)


def start_times(model,x,I,target):
    m=model.unpack(x);bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
    found,diag=f.cycles(I,model.equilibrium(I,m),m['co'],m['padded'],bounds,n=96,tol=2e-10)
    large=[q for q in found if q['stable'] and q['min']<m['th'][model.fixed[0]] and q['max']>m['th'][model.fixed[1]]]
    if not large:raise FoldError(f'No large cycle crossing all cuts at I={I}: {diag}')
    orbit=max(large,key=lambda q:q['max']-q['min'])
    crossed=np.flatnonzero((m['th']>orbit['min'])&(m['th']<orbit['max']))
    fold=BoundaryFold(model,target,int(crossed[0]),int(crossed[-1]))
    def rhs(t,y):return m['co']@np.r_[1,y[0],y[1],I,np.maximum(y[0]-m['padded'],0)]
    events=[]
    for cut in m['th']:
        def ev(t,y,cut=cut):return y[0]-cut
        ev.direction=0;ev.terminal=False;events.append(ev)
    sol=solve_ivp(rhs,(0,2.2*orbit['T']),[orbit['section'],orbit['s']],method='DOP853',rtol=3e-10,atol=1e-12,max_step=.04,events=events)
    hits=sorted((float(t),j,y) for j,(ts,ys) in enumerate(zip(sol.t_events,sol.y_events)) for t,y in zip(ts,ys))
    starts=[j for j,(t,k,y) in enumerate(hits) if k==fold.boundaries[0] and rhs(t,y)[0]>0]
    for a,b in zip(starts[:-1],starts[1:]):
        seq=hits[a:b+1]
        if tuple(row[1] for row in seq)==fold.boundaries:return fold,np.diff([row[0] for row in seq])
    raise FoldError('Detected cycle does not follow all-threshold itinerary')


def independent_fold(model,x,fold,meta):
    t=np.asarray(meta['crossing_times']);_,m,z,h,_=fold.matching(x,t,True)
    def rhs(t,y):
        flow=m['co']@np.r_[1,y[0],y[1],fold.I,np.maximum(y[0]-m['padded'],0)]
        trace=m['co'][0,1]+m['co'][1,2]+sum(m['co'][0,4+j] for j in range(len(m['th'])) if y[0]>m['th'][j])
        return np.r_[flow,trace]
    eps=1e-7;y0=np.array([z[0],h[0],0.]);initial=solve_ivp(rhs,(0,eps),y0,method='DOP853',rtol=3e-12,atol=1e-14).y[:,-1]
    def crossing(t,y):return y[0]-z[0]
    crossing.direction=1;crossing.terminal=True
    sol=solve_ivp(rhs,(eps,1.2*sum(t)),initial,method='DOP853',rtol=3e-12,atol=1e-14,max_step=.01,events=crossing)
    if not len(sol.t_events[0]):raise FoldError('Independent integrator did not return')
    end=sol.y_events[0][0];period=float(sol.t_events[0][0]);closure=float(end[1]-h[0])
    if abs(closure)>1e-7 or abs(period-sum(t))>1e-5 or abs(end[2])>1e-4:raise FoldError('Independent fold verification failed')
    return dict(period=period,closure_h=closure,log_multiplier=float(end[2]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('variant',choices=['base','region1','region2','region3','all']);p.add_argument('stage',choices=['prepare','geometry','poincare','closed'])
    args=p.parse_args();model=Model(args.variant);out=ROOT/'results'/args.variant;out.mkdir(parents=True,exist_ok=True)
    path=out/'candidate.json';start=time.monotonic()
    if args.stage=='prepare':
        if args.variant!='base':raise ValueError('Prepare is only available for base')
        if path.exists():raise ValueError(f'{path} already exists; preserve it or move it before preparing another baseline')
        source=Path(os.environ.get('FM_PARENT_CANDIDATE',str(f.P/'candidate.json'))).resolve()
        parent=json.loads(source.read_text()); x=model.lift_original(parent['parameters'])
        if not np.allclose(parent['physical_thresholds_C'],BASE,rtol=0,atol=1e-12):raise ValueError('Parent has different physical thresholds')
        save(path,model,x,dict(stage='poincare',source_candidate=str(source),source_fold_constrained=parent.get('fold_constrained'),note='Exact embedding of four-region parent, before comparable closed-stage optimization'))
        print(f'Prepared baseline from {source}; source fold_constrained={parent.get("fold_constrained")}',flush=True)
        return
    if args.stage=='geometry':
        x=model.initial();base_path=ROOT/'results'/'base'/'candidate.json'
        if args.variant!='base' and base_path.exists():
            x=model.lift_base(json.loads(base_path.read_text())['parameters'])
            print(f'GEOMETRY initialized from {base_path}',flush=True)
        rng=np.random.default_rng(20261027);best=np.inf
        runs=int(os.environ.get('FM_REFINEMENT_STARTS','10'))
        for j in range(runs):
            raw=x.copy() if j==0 else x+rng.normal(0,.2,model.size)
            raw=np.clip(raw,model.lo+1e-8,model.hi_bounds-1e-8);raw=model.constrain_events(raw)
            def expand(v):
                y=raw.copy();y[model.active]=v;return model.constrain_events(y)
            sol=least_squares(lambda v:geometry(model,expand(v)),raw[model.active],bounds=(model.lo[model.active],model.hi_bounds[model.active]),max_nfev=int(os.environ.get('FM_REFINEMENT_GEOMETRY_MAX_NFEV','1300')),x_scale='jac')
            y=expand(sol.x);loss=float(sol.fun@sol.fun)
            print(f'GEOMETRY {args.variant} start={j+1}/{runs} loss={loss:.7g} nfev={sol.nfev}',flush=True)
            if loss<best and model.physically_valid_events(y):
                best=loss;save(path,model,y,dict(stage='geometry',loss=loss,optimizer=dict(nfev=sol.nfev,message=sol.message)))
        if not np.isfinite(best):raise ValueError('No physical geometry candidate')
        return
    if not path.exists():raise ValueError('Run geometry first')
    seed=json.loads(path.read_text());base=model.constrain_events(seed['parameters']);install_model(model)
    if args.stage=='poincare':
        if seed.get('stage')!='geometry':raise ValueError('Poincare stage requires a geometry candidate')
        best=np.inf;calls=0
        def expand(v):
            q=base.copy();q[model.active]=v;return model.constrain_events(q)
        def residual(v):
            nonlocal best,calls
            calls+=1;x=expand(v);g=geometry(model,x);r=f.p_residual(x)
            loss=np.r_[g,np.sqrt(6*f.POINCARE_SCALE)*r];cost=float(loss@loss)
            if cost<best and model.physically_valid_events(x):
                best=cost;save(out/'poincare_best.json',model,x,dict(stage='poincare',loss=cost,geometry_raw=float(g@g),poincare_raw=float(r@r),calls=calls))
            if calls==1 or calls%100==0:print(f'POINCARE {args.variant} eval={calls} best={best:.7g} elapsed={time.monotonic()-start:.1f}s',flush=True)
            return loss
        sol=least_squares(residual,base[model.active],bounds=(model.lo[model.active],model.hi_bounds[model.active]),max_nfev=int(os.environ.get('FM_REFINEMENT_POINCARE_MAX_NFEV','500')),x_scale='jac')
        if not np.isfinite(best):raise ValueError('No physical Poincare candidate; geometry candidate retained')
        (out/'before_poincare.json').write_text(path.read_text());path.write_text((out/'poincare_best.json').read_text())
        (out/'poincare_optimizer.json').write_text(json.dumps(dict(nfev=sol.nfev,success=bool(sol.success),message=sol.message,best=best),indent=2))
        return
    if seed.get('stage') not in ('poincare','closed'):raise ValueError('Run poincare before closed')
    target=float(os.environ.get('FM_FIXED_FOLD_I','1.1231503495944923'))
    probes=list(dict.fromkeys([float(os.environ.get('FM_FOLD_PROBE_I','1.1')),1.05,1.,.9,.8,.65,.55]))
    seed_options=[('current candidate',base)]
    base_path=ROOT/'results'/'base'/'candidate.json'
    if args.variant!='base' and base_path.exists():
        parent=json.loads(base_path.read_text())
        if parent.get('stage')=='closed':
            seed_options.append(('lifted base closed candidate',model.lift_base(parent['parameters'])))
        else:
            print(f'FOLD fallback unavailable: {base_path} is not a closed candidate',flush=True)
    anchor=None;meta=None;dep=None;seed_name=None
    candidates=[j for j in model.active if j not in (model.n0,model.ni)]
    budget=int(os.environ.get('FM_FOLD_SOLVER_MAX_NFEV','200'))
    for name,trial_base in seed_options:
        print(f'FOLD initialization seed: {name}',flush=True)
        times=[]
        for I in probes:
            try:times.append((I,*start_times(model,trial_base,I,target)))
            except (ValueError,FloatingPointError) as e:print(f'FOLD no orbit at I={I}: {e}',flush=True)
        for index in candidates:
            for I,fold,t in times:
                try:
                    print(f'FOLD trying parameter={index} I={I} seed={name}',flush=True)
                    projected,info=fold.solve(trial_base,t,index,budget)
                    info['independent_check']=independent_fold(model,projected,fold,info)
                    base=projected;anchor=info['crossing_times'];meta=info;dep=index;selected_fold=fold;seed_name=name
                    break
                except (ValueError,FloatingPointError) as e:print(f'FOLD rejected: {e}',flush=True)
            if dep is not None:break
        if dep is not None:break
        print(f'FOLD seed failed: {name}',flush=True)
    if dep is None:raise FoldError('No verified fold initialization from any available seed. Existing candidate retained.')
    fold=selected_fold
    print(f'FOLD initialized parameter={int(dep)} target_I={target:.12g} crossings={fold.boundaries} seed={seed_name}',flush=True)
    active=np.array([j for j in model.active if j!=dep]);scale=np.full(model.size,.2);scale[model.n0:model.hi+1]=.03
    bounds=list(zip(np.maximum(-1.,(model.lo[active]-base[active])/scale[active]),np.minimum(1.,(model.hi_bounds[active]-base[active])/scale[active])))
    count_weight=float(os.environ.get('FM_CYCLE_COUNT_WEIGHT',str(6*f.POINCARE_SCALE)));count_reference=cycle_count_loss.reference_grid() if count_weight else []
    seeds=int(os.environ.get('FM_CYCLE_COUNT_SEEDS','32'));source=closed_fit.EQ_SOURCE
    best=np.inf;calls=0
    def objective(y):
        nonlocal best,calls
        calls+=1;raw=base.copy();raw[active]+=scale[active]*y
        try:
            if np.all(y==0):x,info=base,meta
            else:x,info=fold.solve(raw,anchor,dep,budget)
            g=geometry(model,x);r=closed_fit.residual(x);eq=model_cost(model.unpack(x),source,model.equilibrium)
            parts=closed_components(g@g,eq,r@r,6.,f.POINCARE_SCALE)
            count,report=cycle_count_loss.evaluate(x,count_reference,seeds) if count_weight else (0.,[])
            parts['poincare_count_raw']=count;parts['poincare_count_weight']=count_weight;parts['total']+=count_weight*count
            cost=float(parts['total'])
            if cost<best:
                info=dict(info);info['independent_check']=independent_fold(model,x,fold,info)
                save(path,model,x,dict(stage='closed',loss=cost,loss_components=parts,fold_constraint=info,fold_initialization_seed=seed_name,training_settings=dict(poincare_weight_scale=f.POINCARE_SCALE,training_cycle_inputs=sorted({q['I'] for q in f.TARGETS}),count_weight=count_weight,count_seeds=seeds,count_inputs=[r['I'] for r in count_reference],fixed_fold_I=target),optimized_parameter_count=len(active),derived_parameter_indices=sorted(model.derived+[dep]),cycle_count_training=report,calls=calls))
                best=cost
            if calls==1 or calls%20==0:print(f'CLOSED {args.variant} eval={calls} best={best:.7g} elapsed={time.monotonic()-start:.1f}s parts={parts}',flush=True)
            return cost
        except (ValueError,FloatingPointError,OverflowError) as e:
            if calls<10 or calls%100==0:print(f'CLOSED rejected eval={calls}: {e}',flush=True)
            return 1e12+float(y@y)
    sol=minimize(objective,np.zeros(len(active)),method='Powell',bounds=bounds,options=dict(maxfev=int(os.environ.get('FM_CLOSED_MAXFEV','4500')),xtol=2e-3,ftol=3e-5))
    (out/'optimizer.json').write_text(json.dumps(dict(nfev=sol.nfev,best_visited=best,success=bool(sol.success),message=sol.message),indent=2))
    if not np.isfinite(best):raise ValueError('No verified closed candidate; geometry candidate retained')

if __name__=='__main__':main()
