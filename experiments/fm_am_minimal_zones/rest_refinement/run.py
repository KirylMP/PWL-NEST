#!/usr/bin/env python3
"""Fit nested AM/FM models with exact rest and fixed threshold events."""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize
from model import Model,ROOT


def geometry(model,x):
    f=model.f;m=model.unpack(x)
    if model.case=='am':
        C,I,w,y,ie,ee,sp,_=f.DATA;b=model.basis(C,I,m)
        geo=np.stack([b@m['N'],b@m['H']],axis=-1)-y
        be=model.basis(ee[:,0],ie,m);er=np.stack([be@m['N']-ee[:,1],be@m['H']-ee[:,1]],axis=-1)
        regs=np.searchsorted(model.cuts,ee[:,0]);sr=np.column_stack([m['T'][regs]-sp[:,0],(m['D'][regs]-sp[:,1])/.15])
        return np.r_[(geo*w[:,:,None]/.04).ravel(),(er/.025/np.sqrt(len(ie))).ravel(),.18*sr.ravel()/np.sqrt(len(ie))]
    C,I,w,y,ce,ie,he,sp=f.DATA;b=model.basis(C,I,m)
    geo=np.stack([b@m['N'],b@m['H']],axis=-1)-y
    be=model.basis(ce,ie,m);er=np.stack([be@m['N']-he,be@m['H']-he],axis=-1)
    regs=np.searchsorted(m['tc'],ce);sr=np.column_stack([m['T'][regs]-sp[:,0],(m['D'][regs]-sp[:,1])/.15])
    mid=m['tc'][4] if model.construction in ('full_fixed','upper70','upper100','upper_both') else (m['tc'][2] if model.construction in ('split_fixed','six_fixed') else model.base[1]);bmid=model.basis(mid,0.,m);ev=(bmid@(m['N']-m['H']))/(m['H'][2]-m['N'][2])
    return np.r_[(geo*w[:,:,None]/.025).ravel(),(er/.04/np.sqrt(len(ce))).ravel(),.35*sr.ravel()/np.sqrt(len(ce)),2*(ev-.482304)/.04]


def install(model):
    f=model.f;f.unpack=model.unpack;f.basis=model.basis
    if model.case=='am':f.equilibria=model.equilibrium
    else:f.equilibrium=model.equilibrium


def json_numpy(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Cannot encode {type(value).__name__}')


def save(path,model,x,extra):
    x=model.constrain_events(x);m=model.unpack(x)
    if not model.physically_valid_events(x):raise ValueError('Fixed event heights outside physical domain')
    eq=model.equilibrium(0.,m);near=min(eq,key=lambda q:abs(q['C']-model.rest[0]))
    events=[]
    for j,I in zip(model.event_indices,model.event_I):
        b=model.basis(m['tc'][j],I,m);events.append(float(b@(m['N']-m['H'])))
    obj=dict(case=model.case,coordinate_family=model.coordinate,construction=model.construction,threshold_count_m=len(model.cuts),fm_lower_type=model.fm_lower_type if model.case=='fm' else None,
        physical_thresholds_C=m['tc'].tolist(),original_thresholds_C=model.base.tolist(),thresholds_optimized=bool(model.extra_cuts),
        optimized_threshold_indices=list(range(7,len(model.cuts))) if model.extra_cuts else [],
        fixed_event_threshold_indices=model.event_indices,fixed_equilibrium_threshold_I=model.event_I.tolist(),
        fixed_event_residuals=events,fixed_event_heights=model.event_heights(x).tolist(),
        fixed_rest=dict(I=0.,C=float(model.rest[0]),h=float(model.rest[1]),model_C=near['C'],model_h=near['z'][1],stable=near['stable']),
        parameters=x.tolist(),alpha=m['N'].tolist(),beta=m['H'].tolist(),a=m['a'],b=m['b'],
        transform=None if m['shape'] is None else dict(zip(['p','e','n','k'],m['shape']))|{'C_ref':.5},
        trace=m['T'].tolist(),determinant=m['D'].tolist(),regional_parent_types=model.kind.tolist(),
        regional_types=([('stable_focus' if model.fm_lower_type=='focus' else 'stable_node') if k==0 else {1:'saddle',2:'unstable_focus',3:'stable_focus',4:'unstable_node'}[int(k)] for k in model.kind] if model.case=='fm' else None),
        derived_parameter_indices=model.derived,accepted=False,**extra)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=json_numpy));tmp.replace(path)


def am_cycle_residual(model,x):
    f=model.f;m=model.unpack(x);bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape']);res=[];details=[]
    for q in f.TARGETS:
        I=q['I'];sec=float(f.phi(q['C'],m['shape']));n=float(model.basis(q['C'],I,m)@m['N']);low=max(n+1e-7,1e-7);high=1-1e-7
        cache={}
        def calc(h):
            h=float(h)
            if h not in cache:cache[h]=f.ret(I,h,sec,m['co'],m['padded'],bounds,tol=2e-9,fast=False)
            return cache[h]
        h=float(np.clip(q['h'],low,high));found=False;v=None
        if low<high:
            for _ in range(14):
                status,v=calc(h)
                if status!=1:break
                if abs(v[0])<3e-8 and v[2]<-1e-6 and v[4]-v[3]>1e-5:found=True;break
                derivative=np.expm1(np.clip(v[2],-700,50))
                if abs(derivative)<1e-8:break
                hn=float(np.clip(h+np.clip(-v[0]/derivative,-.04,.04),low,high))
                if abs(hn-h)<1e-11:break
                h=hn
            if not found:
                grid=low+(high-low)*np.r_[np.geomspace(1e-5,.1,12),np.linspace(.1,1,16)]
                vals=[calc(s)[1][0] if calc(s)[0]==1 else np.nan for s in grid]
                for a,b,va,vb in zip(grid[:-1],grid[1:],vals[:-1],vals[1:]):
                    if not (np.isfinite(va) and np.isfinite(vb) and va*vb<0):continue
                    try:h=float(brentq(lambda s:calc(s)[1][0] if calc(s)[0]==1 else np.nan,a,b,xtol=2e-10));status,v=calc(h)
                    except ValueError:continue
                    if status==1 and abs(v[0])<3e-8 and v[2]<-1e-6 and v[4]-v[3]>1e-5:found=True;break
        if found:
            lo,hi=np.asarray(f.inverse(v[3:5],m),float);rr=[(v[1]-q['T'])/2,(lo-q['Cmin'])/.025,(hi-q['Cmax'])/.035,((hi-lo)-(q['Cmax']-q['Cmin']))/.02]
        else:rr=[8.,8.,8.,8.]
        res.extend(rr);details.append(dict(I=I,found=found,h=h,residual=rr))
    return np.asarray(res),details


def fm_fold_seed(model,x,target):
    f=model.f
    spec=importlib.util.spec_from_file_location('refinement_fold',ROOT/'fm_four_zone_refinement/fold.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for I in [1.1,1.05,1.,.9,.8,.65,.55]:
        print(f'FOLD scanning for a large cycle at I={I}',flush=True)
        m=model.unpack(x);bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
        found,diag=f.cycles(I,model.equilibrium(I,m),m['co'],m['padded'],bounds,n=96,tol=2e-10)
        large=[q for q in found if q['stable'] and q['min']<m['th'][model.fixed[0]] and q['max']>m['th'][model.fixed[-1]]]
        if not large:print(f'FOLD no large orbit I={I}: {diag}',flush=True);continue
        orbit=max(large,key=lambda q:q['max']-q['min']);crossed=np.flatnonzero((m['th']>orbit['min'])&(m['th']<orbit['max']))
        fold=mod.BoundaryFold(model,target,int(crossed[0]),int(crossed[-1]))
        def rhs(t,y):return m['co']@np.r_[1,y[0],y[1],I,np.maximum(y[0]-m['padded'],0)]
        events=[]
        for cut in m['th']:
            def ev(t,y,cut=cut):return y[0]-cut
            ev.direction=0;ev.terminal=False;events.append(ev)
        sol=solve_ivp(rhs,(0,2.2*orbit['T']),[orbit['section'],orbit['s']],method='DOP853',rtol=3e-10,atol=1e-12,max_step=.04,events=events)
        hits=sorted((float(t),j,y) for j,(ts,ys) in enumerate(zip(sol.t_events,sol.y_events)) for t,y in zip(ts,ys))
        starts=[j for j,(t,k,y) in enumerate(hits) if k==fold.boundaries[0] and rhs(t,y)[0]>0]
        times=None
        for a,b in zip(starts[:-1],starts[1:]):
            seq=hits[a:b+1]
            if tuple(q[1] for q in seq)==fold.boundaries:times=np.diff([q[0] for q in seq]);break
        if times is None:print(f'FOLD itinerary unresolved I={I}',flush=True);continue
        for dep in model.active:
            if dep==model.ni or (model.extra_cuts and dep>=model.cut_start):continue
            try:
                y,info=fold.solve(x,times,int(dep),int(os.environ.get('FM_FOLD_SOLVER_MAX_NFEV','200')))
                print(f'FOLD found target={target} probe={I} parameter={dep}',flush=True)
                return fold,y,int(dep),info
            except (ValueError,FloatingPointError) as e:print(f'FOLD rejected I={I} parameter={dep}: {e}',flush=True)
    raise ValueError('Fixed FM cycle fold could not be initialized; saved candidate retained')


def verify_fold(model,x,fold,info):
    t=np.asarray(info['crossing_times']);_,m,z,h,_=fold.matching(x,t,True)
    def rhs(t,y):
        flow=m['co']@np.r_[1,y[0],y[1],fold.I,np.maximum(y[0]-m['padded'],0)]
        trace=m['co'][0,1]+m['co'][1,2]+sum(m['co'][0,4+j] for j in range(len(m['th'])) if y[0]>m['th'][j])
        return np.r_[flow,trace]
    eps=1e-7;y0=np.array([z[0],h[0],0.]);initial=solve_ivp(rhs,(0,eps),y0,method='DOP853',rtol=3e-12,atol=1e-14).y[:,-1]
    def crossing(t,y):return y[0]-z[0]
    crossing.direction=1;crossing.terminal=True
    sol=solve_ivp(rhs,(eps,1.2*sum(t)),initial,method='DOP853',rtol=3e-12,atol=1e-14,max_step=.01,events=crossing)
    if not len(sol.t_events[0]):raise ValueError('Independent fold integrator did not return')
    end=sol.y_events[0][0];period=float(sol.t_events[0][0]);closure=float(end[1]-h[0])
    if abs(closure)>1e-7 or abs(period-sum(t))>1e-5 or abs(end[2])>1e-4:raise ValueError(f'Independent fold check failed closure={closure}, period={period}, log_multiplier={end[2]}')
    return dict(period=period,closure_h=closure,log_multiplier=float(end[2]))


def plot_branches(model,rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for row in rows:
        for eq in row['equilibria']:ax[0].scatter(row['I'],eq['C'],s=5,color='tab:blue' if eq['stable'] else 'tab:red')
        for q in row['cycles']:
            kw=dict(s=8,edgecolors='black',facecolors='black' if q['stable'] else 'none')
            ax[0].scatter([row['I']]*2,[q['Cmin'],q['Cmax']],**kw);ax[1].scatter(row['I'],q['T'],**kw)
    ax[2].step([r['I'] for r in rows],[len(r['cycles']) for r in rows],where='mid',color='black',label='PWA: found cycles')
    if model.case=='fm':
        reference=json.loads((model.f.DATA_DIR/'LR_cycles_reference.json').read_text())['rows']
        for row in reference:
            for q in row['cycles']:
                kw=dict(s=9,edgecolors='tab:purple',facecolors='tab:purple' if q['stable'] else 'none')
                ax[0].scatter([row['I']]*2,[q['min'],q['max']],**kw);ax[1].scatter(row['I'],q['T'],**kw)
        ax[2].step([r['I'] for r in reference],[len(r['cycles']) for r in reference],where='mid',color='tab:purple',label='LR: archived finite scan')
    else:
        source=np.genfromtxt(model.f.ROOT/'LR_stable_cycles.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
        ax[0].scatter(source['I'],source['Cmin'],s=7,c='tab:purple');ax[0].scatter(source['I'],source['Cmax'],s=7,c='tab:purple')
        ax[1].scatter(source['I'],source['T'],s=7,c='tab:purple')
    for t in model.unpack(model.constrain_events(json.loads((path.parent/'candidate.json').read_text())['parameters']))['tc']:ax[0].axhline(t,color='tab:cyan',linestyle='--',alpha=.5,linewidth=.8)
    for a in ax:a.set_xlabel('I (µM)');a.grid(alpha=.2)
    ax[0].set_ylabel('C (µM)');ax[1].set_ylabel('Period (s)');ax[2].set_ylabel('Detected cycles');ax[2].legend()
    fig.suptitle(f'{model.case.upper()}: fixed rest and refined thresholds');fig.savefig(path,dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('case',choices=['am','fm']);ap.add_argument('stage',choices=['prepare','geometry','poincare','closed','validate'])
    ap.add_argument('--coordinate',choices=['mixed','affine'],default='mixed');ap.add_argument('--construction',choices=['refined','base','split','split_fixed','six_fixed','full_fixed','upper70','upper100','upper_both'],default='refined');ap.add_argument('--fm-lower-type',choices=['node','focus'],default='node');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--parent',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);model=Model(a.case,a.coordinate,a.construction,a.fm_lower_type);f=model.f;install(model)
    path=a.out/'candidate.json';scale=float(os.environ.get('POINCARE_WEIGHT_SCALE','100'));start=time.monotonic()
    if not np.isfinite(scale) or scale<=0:raise ValueError('POINCARE_WEIGHT_SCALE must be finite and positive')
    if a.stage=='prepare':
        if path.exists():raise ValueError(f'Candidate exists: {path}')
        parent=(a.parent or model.default_parent).resolve();x=model.lift_parent(json.loads(parent.read_text()))
        save(path,model,x,dict(stage='prepare',parent=str(parent),note=f'Rest and fixed events exact; construction={a.construction}.'))
        print(f'PREPARE {a.case} saved {path}; rest={model.rest.tolist()}',flush=True);return
    if not path.exists():raise ValueError('Run prepare first')
    seed=json.loads(path.read_text())
    if seed.get('threshold_count_m')!=len(model.cuts):raise ValueError('Candidate threshold count does not match --construction')
    if a.case=='fm' and seed.get('fm_lower_type','node')!=a.fm_lower_type:raise ValueError('Candidate lower type does not match --fm-lower-type')
    if seed.get('construction',a.construction)!=a.construction:raise ValueError('Candidate construction mismatch')
    base=model.constrain_events(seed['parameters'])
    if a.stage=='validate':
        from concurrent.futures import ThreadPoolExecutor
        n=int(os.environ.get('VALIDATE_GRID_SIZE','81'));workers=int(os.environ.get('WORKERS','4'))
        if n<2 or workers<1:raise ValueError('VALIDATE_GRID_SIZE >=2 and WORKERS >=1 required')
        grid=np.linspace(0,2 if a.case=='fm' else 1,n);m=model.unpack(base);bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
        def scan(I):
            eq=model.equilibrium(float(I),m);cycles,diag=f.cycles(float(I),eq,m['co'],m['padded'],bounds,n=int(os.environ.get('VALIDATE_SEEDS','48')),tol=1e-9)
            for q in cycles:q.update(Cmin=float(f.inverse(np.asarray(q['min']),m)),Cmax=float(f.inverse(np.asarray(q['max']),m)))
            return dict(I=float(I),equilibria=eq,cycles=cycles,diagnostics=diag)
        rows=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for j,row in enumerate(pool.map(scan,grid)):
                rows.append(row)
                if j%10==0:print(f'VALIDATE {a.case} {j+1}/{n} I={row["I"]:.5g} cycles={len(row["cycles"])}',flush=True)
        (a.out/'candidate_branches.json').write_text(json.dumps(dict(rows=rows,method='finite grid and finite Poincare seed scan; not a completeness proof'),indent=2,default=json_numpy))
        plot_branches(model,rows,a.out/'candidate_bifurcation.png')
        print(f'VALIDATE saved {a.out}/candidate_branches.json',flush=True);return
    if a.stage=='poincare' and seed.get('stage')!='geometry':raise ValueError('Run geometry before poincare')
    if a.stage=='closed' and seed.get('stage')!='poincare':raise ValueError('Run poincare before closed')
    os.environ['FIT_EXTREMA']='1';fold=None;fold_info=None;dep=None;count_reference=[];eq_source=[];count_weight=0.
    if a.stage=='closed':
        from equilibrium_loss import model_cost
        grid=np.linspace(.4,1.2,25) if a.case=='fm' else np.linspace(.2,.8,25)
        eq_source=[(float(I),f.eqs(float(I))) for I in grid]
    if a.stage=='closed' and a.case=='fm':
        import closed_fit,cycle_count_loss
        target=float(os.environ.get('FM_FIXED_FOLD_I','1.1231503495944923'))
        fold,base,dep,fold_info=fm_fold_seed(model,base,target)
        count_weight=float(os.environ.get('FM_CYCLE_COUNT_WEIGHT','600'))
        if not np.isfinite(count_weight) or count_weight<0:raise ValueError('FM_CYCLE_COUNT_WEIGHT must be finite and nonnegative')
        count_reference=cycle_count_loss.reference_grid() if count_weight else []
    active=np.array([i for i in model.active if i!=dep],dtype=int);param_scale=np.full(model.size,.2);param_scale[model.ni]=.03;param_scale[model.shape:model.cut_start]=[.3,.3,.15,.25];param_scale[model.cut_start:]=.6
    bounds=list(zip(np.maximum(-1.5,(model.lo[active]-base[active])/param_scale[active]),np.minimum(1.5,(model.upper[active]-base[active])/param_scale[active])))
    if any(lo>0 or hi<0 or lo>=hi for lo,hi in bounds):raise ValueError('Initial candidate violates active parameter bounds')
    best=np.inf;calls=0
    def objective(y):
        nonlocal best,calls
        calls+=1;raw=base.copy();raw[active]+=param_scale[active]*y
        if a.stage=='closed' and (calls==1 or calls%20==0):print(f'CLOSED {a.case} eval={calls} started elapsed={time.monotonic()-start:.1f}s',flush=True)
        try:
            x=model.constrain_events(raw);info=None
            if fold is not None:
                x,info=(base,fold_info) if np.all(y==0) else fold.solve(x,fold_info['crossing_times'],dep,int(os.environ.get('FM_FOLD_SOLVER_MAX_NFEV','200')))
            if not model.physically_valid_events(x):raise ValueError('Nonphysical event heights')
            g=geometry(model,x);parts=dict(geometry=float(g@g))
            if a.stage=='poincare':
                r=f.p_residual(x);parts['poincare_raw']=float(r@r);cost=parts['geometry']+6*scale*parts['poincare_raw']
            elif a.stage=='closed':
                eq_loss=float(model_cost(model.unpack(x),eq_source,model.equilibrium));parts['equilibrium_sets']=eq_loss
                if a.case=='fm':
                    r=closed_fit.residual(x)
                    def count_progress(done,total,row):
                        print(f'COUNT eval={calls} {done}/{total} I={row["I"]:.6g} stable={row["stable"]} unstable={row["unstable"]} diagnostics={row["diagnostics"]}',flush=True)
                    count,report=cycle_count_loss.evaluate(x,count_reference,int(os.environ.get('FM_CYCLE_COUNT_SEEDS','32')),count_progress if calls==1 or calls%100==0 else None) if count_weight else (0.,[])
                    parts.update(cycles_raw=float(r@r),poincare_count_raw=count);cost=parts['geometry']+eq_loss+6*scale*parts['cycles_raw']+count_weight*count
                else:
                    r,report=am_cycle_residual(model,x);parts['cycles_raw']=float(r@r);cost=parts['geometry']+eq_loss+2*scale*parts['cycles_raw']
            else:cost=parts['geometry']
            if not np.isfinite(cost):raise ValueError('Nonfinite loss')
            if cost<best:
                extra=dict(stage=a.stage,loss=float(cost),loss_components=parts,calls=calls,fold_constrained=fold is not None,
                           training_settings=dict(poincare_weight_scale=scale,training_cycle_inputs=sorted({float(q['I']) for q in f.TARGETS}),
                             fm_cycle_count_weight=count_weight if a.case=='fm' and a.stage=='closed' else None,
                             fixed_fold_I=target if fold is not None else None))
                if info is not None:
                    info=dict(info);info['independent_check']=verify_fold(model,x,fold,info);extra['fold_constraint']=info
                if a.stage=='closed':extra['cycle_report']=report
                save(path,model,x,extra);best=cost
            if calls==1 or calls%100==0:print(f'{a.stage.upper()} {a.case} eval={calls} best={best:.7g} elapsed={time.monotonic()-start:.1f}s components={parts}',flush=True)
            return float(cost)
        except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:
            if calls<5 or calls%100==0:print(f'{a.stage.upper()} {a.case} rejected eval={calls}: {e}',flush=True)
            return 1e12+float(y@y)
    budget=int(os.environ.get(a.stage.upper()+'_MAXFEV','4500'))
    starts=int(os.environ.get(a.stage.upper()+'_STARTS','4' if a.stage=='geometry' else '1'))
    if budget<1 or starts<1:raise ValueError('Optimizer budget and starts must be positive')
    rng=np.random.default_rng(20261027);runs=[]
    for j in range(starts):
        y0=np.zeros(len(active)) if j==0 else np.array([np.clip(rng.normal(0,.2),lo,hi) for lo,hi in bounds])
        print(f'{a.stage.upper()} {a.case} start={j+1}/{starts} budget={budget}',flush=True)
        sol=minimize(objective,y0,method='Powell',bounds=bounds,options=dict(maxfev=budget,xtol=2e-3,ftol=3e-5))
        runs.append(dict(nfev=int(sol.nfev),fun=float(sol.fun),success=bool(sol.success),message=str(sol.message)))
    (a.out/(a.stage+'_optimizer.json')).write_text(json.dumps(dict(starts=runs,best_visited=float(best) if np.isfinite(best) else None),indent=2))
    if not np.isfinite(best):raise ValueError(f'No admissible {a.stage} candidate; previous candidate retained')


if __name__=='__main__':main()
