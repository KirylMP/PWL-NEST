"""Refine AM, AFM or FM in three zones with warm/fallback fold shooting.

Fits the same stationary-cycle objective as the anchored batch. Equilibrium
anchors, regional stability, and fold closure/M=1 remain hard numerical checks.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from anchored_lower_fold import FREE
from continue_physical_fixed import Orbit, nondegeneracy
from fit_lower_fixed_fold import metrics
from robust_anchored_fold import FoldSolver

ROOT=Path(__file__).resolve().parents[1]
PIVOTS={'AM':('alpha',1),'AFM':('alpha',1),'FM':('alpha',4)}
REFERENCE_SEEDS={'AM':7,'AFM':22,'FM':8}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['AM','AFM','FM'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--max-nfev',type=int,default=80)
    p.add_argument('--duration',type=float,default=400.);p.add_argument('--bound',type=float,default=4.)
    p.add_argument('--max-init-tries',type=int,default=20);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.seed<0 or a.max_nfev<1 or a.duration<=0 or a.bound<=2.1 or a.max_init_tries<1:p.error('invalid numerical settings')
    a.output.mkdir(parents=True,exist_ok=False)
    path=ROOT/'paper/data/physical_models/fixed_fold'/a.case/'parameters.json'
    base=json.loads(path.read_text());sha=hashlib.sha256(path.read_bytes()).hexdigest()
    audit=json.loads((ROOT/'paper/data/physical_models/fixed_fold/fold_audit.json').read_text())['results'][a.case.lower()]
    if sha!=audit['sha256']:raise RuntimeError('baseline hash mismatch')
    reference_path=ROOT/'results/anchored_three_main'/a.case/f'seed_{REFERENCE_SEEDS[a.case]}'/'attempt_001/candidate.json'
    reference=json.loads(reference_path.read_text());fit=reference['anchored_fit']
    if fit['source_sha256']!=sha:raise RuntimeError('reference candidate source hash mismatch')
    reference_x=np.array(fit['best_x'],float);ref_fold=fit['fold']
    pivot=PIVOTS[a.case]
    free=[v for v in FREE if v!=pivot]
    if [list(x) for x in free]!=fit['free_coefficients']:raise RuntimeError('reference parameter mapping mismatch')
    scales=np.array([.05*max(abs(base[k][j]),.3) for k,j in free])
    cut=.25*min(base['rest_target'][0],base['physical_thresholds_C'][0],audit['neutral']['Cmin'])
    solver=FoldSolver(base,pivot,cut,reference_x,ref_fold)
    rows=base['training_cycles'];lo=np.full(len(free),-a.bound);hi=np.full(len(free),a.bound)
    count=0;best={'loss':float('inf')};invalid_jac=0;failure_reasons=Counter();last_failure=''
    def candidate(x):
        changes={key:float(v*s) for key,v,s in zip(free,x,scales)}
        three,_,checks,fold=solver.solve(x,changes)
        return three,checks,fold
    def evaluate(x,record=True):
        nonlocal count,best,last_failure
        count+=1
        try:
            model,checks,fold=candidate(x)
            vals=[metrics(model,r,a.duration) for r in rows]
            if any(m is None or m['min_all']<=0 for m in vals):raise ValueError('unresolved or nonpositive training trajectory')
            rr=[]
            for row,m in zip(rows,vals):
                rr.extend([(m['Cmin']-row['target_Cmin'])/.03,
                           (m['Cmax']-row['target_Cmax'])/.08,
                           (m['period']-row['target_period'])/5.])
            rr.extend(.01*x);out=np.asarray(rr);loss=float(out@out)
            if record and loss<best['loss']:
                best={'loss':loss,'x':x.tolist(),'calls':count,'checks':checks,'fold':fold}
                (a.output/'checkpoint.json').write_text(json.dumps(best,indent=2)+'\n')
            status=True
        except (ValueError,np.linalg.LinAlgError,OverflowError,FloatingPointError) as exc:
            last_failure=str(exc);failure_reasons[last_failure]+=1
            out=np.full(3*len(rows)+len(free),100.)+.001*np.resize(x,3*len(rows)+len(free));status=False
        if count%20==0:print(a.case,a.seed,'calls',count,'best',best['loss'],'fold',solver.stats(),flush=True)
        return out,status
    ref_r,ref_ok=evaluate(reference_x,record=False)
    if not ref_ok:raise RuntimeError('reference candidate could not be reconstructed')
    ref_loss=float(ref_r@ref_r);rng=np.random.default_rng(a.seed);start=None
    for tries in range(1,a.max_init_tries+1):
        if a.seed==0 and tries==1:xx=reference_x.copy()
        elif a.seed<8:xx=np.clip(reference_x+rng.normal(0.,.35,len(free)),lo+.02,hi-.02)
        else:xx=rng.uniform(-min(3.,a.bound-.02),min(3.,a.bound-.02),len(free))
        rr,valid=evaluate(xx)
        if valid:start=xx.copy();break
        print('invalid_start',a.case,a.seed,'attempt',tries,'reason',last_failure,flush=True)
    if start is None:
        (a.output/'start_failure.json').write_text(json.dumps({'case':a.case,'seed':a.seed,
          'tries':a.max_init_tries,'reasons':dict(failure_reasons),'fold_solver':solver.stats()},indent=2)+'\n')
        raise RuntimeError('no feasible start found; see start_failure.json and log')
    initial_loss=float(rr@rr)
    print('valid_start',a.seed,'attempt',tries,'loss',initial_loss,'x',start.tolist(),flush=True)
    def fun(x):return evaluate(x)[0]
    def jac(x):
        nonlocal invalid_jac
        base_r,base_ok=evaluate(x);columns=[]
        for j in range(len(x)):
            samples={}
            for sign in (-1,1):
                xx=x.copy();xx[j]+=.01*sign
                if lo[j]<=xx[j]<=hi[j]:samples[sign]=evaluate(xx)
            minus=samples.get(-1);plus=samples.get(1)
            if minus and minus[1] and plus and plus[1]:col=(plus[0]-minus[0])/.02
            elif base_ok and plus and plus[1]:col=(plus[0]-base_r)/.01
            elif base_ok and minus and minus[1]:col=(base_r-minus[0])/.01
            else:col=np.zeros_like(base_r);invalid_jac+=1
            columns.append(col)
        return np.column_stack(columns)
    opt=least_squares(fun,start,jac=jac,bounds=(lo,hi),x_scale='jac',max_nfev=a.max_nfev,
                      ftol=1e-5,xtol=1e-5,gtol=1e-5)
    final_r,final_ok=evaluate(opt.x,record=False)
    x=np.array(best['x']);model,checks,fold=candidate(x)
    try:
        orb=Orbit(model);ind=orb.point(np.array(fold['times']),fold['I'],1e-9,require_physical=False)
        ind.pop('physical',None);ind.pop('stable',None);fold['independent']=ind
        fold['nondegeneracy']=nondegeneracy(orb,np.r_[fold['times'],fold['I']])
    except ValueError as exc:fold['independent_error']=str(exc)
    predictions=[metrics(model,row,a.duration) for row in rows]
    holdout=[{'I':row['I'],'baseline':metrics(base,row,a.duration),'candidate':metrics(model,row,a.duration),
              'target':{k:row[k] for k in ('target_Cmin','target_Cmax','target_period')}} for row in base['holdout_cycles']]
    report={'case':a.case,'arm':'three','seed':a.seed,'start_attempts':tries,'start_x':start.tolist(),
      'initial_squared_residual':initial_loss,'best_squared_residual':best['loss'],
      'baseline_squared_residual':ref_loss,'reference_description':f'previous anchored {a.case} seed {REFERENCE_SEEDS[a.case]}, reevaluated on this machine',
      'optimizer':{'success':bool(opt.success),'message':str(opt.message),'nfev':int(opt.nfev),
                   'final_x':opt.x.tolist(),'final_squared_residual':float(final_r@final_r) if final_ok else None,
                   'final_valid':final_ok,'residual_calls':count,'invalid_jac_columns':invalid_jac},
      'fold_solver':solver.stats(),'invalid_evaluation_reasons':dict(failure_reasons),
      'source_sha256':sha,'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
      'bound_normalized':a.bound,'free_coefficients':[list(v) for v in free],'scales':scales.tolist(),
      'original_target_inputs':base['fixed_equilibrium_threshold_I'],'rest_target':base['rest_target'],
      'checks':checks,'fold':fold,'training_I':[r['I'] for r in rows],
      'candidate':predictions,'target':[{k:r[k] for k in ('target_Cmin','target_Cmax','target_period')} for r in rows],
      'holdout':holdout,'duration_s':a.duration,
      'scope':'local search with fallback numerical fold shooting, not a global proof'}
    output=copy.deepcopy(model)
    for k in ('status','scope','optimizer','loss','selected','parameters','event_q','fold_constraint','training_cycles','holdout_cycles'):output.pop(k,None)
    output['status']='exploratory_unvalidated'
    output['anchored_fit']={'seed':a.seed,'source_sha256':sha,'checks':checks,'fold':fold,
                            'best_x':x.tolist(),'free_coefficients':[list(v) for v in free]}
    (a.output/'candidate.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('done',a.seed,'reference',ref_loss,'initial',initial_loss,'best',best['loss'],'status',str(opt.message),flush=True)


if __name__=='__main__':main()
