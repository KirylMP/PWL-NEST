"""AM fixed-fold search with finite reset stress as a hard feasibility condition.

Stage 1 seeks a feasible start; stage 2 minimizes the original cycle loss
subject to the same stress condition. Neither stage proves global feasibility.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from anchored_lower_fold import FREE,anchored_fold
from ca_reset_stress import stress
from fit_focus_fold_return import normalized_shape
from return_cycle_metrics import solve_cycle

ROOT=Path(__file__).resolve().parents[1]
PIVOT=('alpha',1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-seed',type=int,default=465,choices=(465,2311,3080,3420));p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--jitter',type=float,default=.5)
    p.add_argument('--feas-nfev',type=int,default=120);p.add_argument('--fit-iterations',type=int,default=50)
    p.add_argument('--min-margin',type=float,default=0.);p.add_argument('--bound',type=float,default=40.)
    a=p.parse_args()
    if min(a.feas_nfev,a.fit_iterations)<1 or a.seed<0 or a.jitter<0 or a.min_margin<0:p.error('invalid arguments')
    source=ROOT/f'results/random_64/AM/seed_{a.source_seed}/fit';r=json.loads((source/'report.json').read_text());source_model=json.loads((source/'candidate.json').read_text())
    base_path=ROOT/'paper/data/physical_models/fixed_fold/AM/parameters.json';base=json.loads(base_path.read_text())
    audit=json.loads((ROOT/'paper/data/physical_models/fixed_fold/fold_audit.json').read_text())['results']['am']
    free=[k for k in FREE if k!=PIVOT];scales=np.array([.05*max(abs(base[k][j]),.3) for k,j in free]);reference=np.asarray(r['best']['x'],float)
    hint=r['best']['fold'];seed_cycles=r['best']['cycles'];rows=base['training_cycles']
    cut=.25*min(base['rest_target'][0],base['physical_thresholds_C'][0],audit['neutral']['Cmin'])
    target=[normalized_shape(row['target_shape']) for row in rows]
    rng=np.random.default_rng(np.random.SeedSequence([20261106,a.seed,a.source_seed]));start=np.clip(reference+(rng.normal(0,a.jitter,5) if a.seed else 0),-a.bound+.01,a.bound-.01)
    a.output.mkdir(parents=True,exist_ok=False);cache=OrderedDict();counts={'field':0,'stress':0,'loss':0};best={'loss':float('inf')};feasible=None;max_margin=-float('inf')
    def field(x):
        key=tuple(map(float,x))
        if key in cache:cache.move_to_end(key);return cache[key]
        counts['field']+=1
        try:
            changes={k:float(v*s) for k,v,s in zip(free,x,scales)}
            d,_,checks,fold=anchored_fold(base,changes,PIVOT,cut,0.,0.,initial_fold=hint)
            T=np.asarray(d['trace']);D=np.asarray(d['determinant']);disc=T*T-4*D
            if not(T[0]<0<T[1] and T[2]<0 and D[0]>1e-8 and np.all(disc[1:]<-1e-6)):raise ValueError('focus conditions')
            counts['stress']+=1;test=stress(d);entry={'model':d,'fold':fold,'checks':checks,'stress':test,'disc':disc.tolist(),'loss':None}
        except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as e:entry={'error':str(e)}
        cache[key]=entry
        if len(cache)>24:cache.popitem(last=False)
        return entry
    def score(x):
        z=field(x)
        return float(z['stress']['min_C']-a.min_margin) if 'stress' in z else -100.
    def loss(x):
        z=field(x)
        if 'model' not in z:return 1e7
        if z['loss'] is not None:return z['loss']
        counts['loss']+=1;terms=[];found=[]
        try:
            for row,old,tgt in zip(rows,seed_cycles,target):
                name=old['route'];sol=solve_cycle(z['model'],row['I'],z['fold']['times'],extra_seeds={name:[old['times']]},route_names=(name,))
                if not sol['candidates']:sol=solve_cycle(z['model'],row['I'],z['fold']['times'],route_names=tuple(v for v in ('both','upper','lower') if v!=name))
                if not sol['candidates']:raise ValueError('unresolved training cycle')
                v=sol['candidates'][0];found.append(v)
                terms.extend([(v['Cmin']-row['target_Cmin'])/.03,(v['Cmax']-row['target_Cmax'])/.08,(v['period']-row['target_period'])/5.])
                terms.extend((normalized_shape(v['shape'])-tgt)/(.1*np.sqrt(64)))
            z['loss']=float(np.dot(terms,terms));z['cycles']=found
        except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as e:z['loss']=1e7;z['loss_error']=str(e)
        return z['loss']
    def save_if_feasible(x):
        nonlocal best
        z=field(x)
        if score(x)<0:return False
        val=loss(x)
        if val>=1e6:return False
        if val<best['loss']:
            best={'loss':val,'x':np.asarray(x).tolist(),'fold':z['fold'],'checks':z['checks'],'stress':z['stress'],
                  'discriminant':z['disc'],'cycles':z['cycles']}
            (a.output/'candidate.json').write_text(json.dumps(z['model'],indent=2,allow_nan=False)+'\n')
            (a.output/'best.json').write_text(json.dumps(best,indent=2,allow_nan=False)+'\n')
            print('FEASIBLE loss',val,'min_C',z['stress']['min_C'],'counts',counts,flush=True)
        return True
    class Found(Exception):pass
    def feasibility_objective(x):
        nonlocal feasible,max_margin
        v=score(x);max_margin=max(max_margin,v)
        print('FEAS',counts['field'],'margin',v,'best',max_margin,flush=True)
        if v>=0 and save_if_feasible(x):feasible=np.asarray(x).copy();raise Found()
        return -v
    bounds=[(-a.bound,a.bound)]*len(start)
    try:
        minimize(feasibility_objective,start,method='Nelder-Mead',bounds=bounds,
                 options={'maxfev':a.feas_nfev,'xatol':1e-4,'fatol':1e-5,'adaptive':True})
    except Found:pass
    status={'case':'AM','seed':a.seed,'source_seed':a.source_seed,'settings':vars(a)|{'output':str(a.output)},
            'base_sha256':hashlib.sha256(base_path.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((source/'candidate.json').read_bytes()).hexdigest(),
            'scope':'finite reset protocol; positivity hard constraint; numerical feasibility not invariance proof','counts':counts}
    if feasible is None:
        status.update(status='no_feasible_start',best_stress_margin=max_margin);(a.output/'status.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n');print('NO_FEASIBLE_START',max_margin,flush=True);return
    def objective(x):
        value=loss(x);save_if_feasible(x);return value
    res=minimize(objective,feasible,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':score}],
                 options={'maxiter':a.fit_iterations,'ftol':1e-5,'eps':1e-3,'disp':True})
    save_if_feasible(res.x);candidate=json.loads((a.output/'candidate.json').read_text());holdout=[]
    try:
        strict=stress(candidate,rtol=1e-10,atol=1e-12,max_step=.03)
    except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as exc:
        status.update(status='strict_stress_unresolved',detail=str(exc),counts=counts)
        (a.output/'status.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n');print('STRICT_STRESS_UNRESOLVED',exc,flush=True);return
    (a.output/'strict_stress.json').write_text(json.dumps(strict,indent=2,allow_nan=False)+'\n')
    if strict['min_C']<a.min_margin:
        status.update(status='strict_stress_rejected',min_C=strict['min_C'],counts=counts)
        (a.output/'status.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n');print('STRICT_STRESS_REJECTED',strict['min_C'],flush=True);return
    for row in base['holdout_cycles']:
        near=min(best['cycles'],key=lambda v:abs(v['I']-row['I']));name=near['route']
        sol=solve_cycle(candidate,row['I'],best['fold']['times'],extra_seeds={name:[near['times']]},route_names=(name,))
        if not sol['candidates']:sol=solve_cycle(candidate,row['I'],best['fold']['times'],route_names=tuple(v for v in ('both','upper','lower') if v!=name))
        holdout.append({'I':row['I'],'target':{k:row[k] for k in ('target_Cmin','target_Cmax','target_period')},'cycle':sol})
    status.update(status='completed',optimizer_success=bool(res.success),message=str(res.message),iterations=int(res.nit),counts=counts,
                  best_loss=best['loss'],min_C=strict['min_C'],holdout=holdout)
    (a.output/'status.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n');print('COMPLETED loss',best['loss'],'min_C',strict['min_C'],flush=True)
if __name__=='__main__':main()
