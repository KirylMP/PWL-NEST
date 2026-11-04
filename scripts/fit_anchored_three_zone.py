"""Fit a three-zone model with exact rest, equilibrium-crossing and fold anchors."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from anchored_lower_fold import FREE, anchored_fold
from continue_physical_fixed import Orbit, nondegeneracy
from fit_lower_fixed_fold import metrics

ROOT=Path(__file__).resolve().parents[1]
PIVOTS={'AM':('alpha',1),'AFM':('alpha',1),'FM':('alpha',4)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,choices=['AM','AFM','FM'])
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--max-nfev',type=int,default=20)
    p.add_argument('--duration',type=float,default=400.)
    p.add_argument('--training',choices=['three','all'],default='all')
    p.add_argument('--max-init-tries',type=int,default=20)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.seed<0 or args.max_nfev<1 or args.max_init_tries<1 or args.duration<=0:p.error('invalid numerical settings')
    args.output.mkdir(parents=True,exist_ok=False)
    path=ROOT/'paper/data/physical_models/fixed_fold'/args.case/'parameters.json'
    base=json.loads(path.read_text());sha=hashlib.sha256(path.read_bytes()).hexdigest()
    audit=json.loads((ROOT/'paper/data/physical_models/fixed_fold/fold_audit.json').read_text())['results'][args.case.lower()]
    if sha!=audit['sha256']:raise RuntimeError('baseline hash mismatch')
    pivot=PIVOTS[args.case];free=[v for v in FREE if v!=pivot]
    scale=np.array([.05*max(abs(base[k][j]),.3) for k,j in free])
    # A zero hinge is an internal algebraic device only; output has two cuts.
    dummy_cut=.25*min(base['rest_target'][0],base['physical_thresholds_C'][0],audit['neutral']['Cmin'])
    rows=base['training_cycles'] if args.training=='all' else [base['training_cycles'][i] for i in (0,3,6)]
    baseline=[metrics(base,r,args.duration) for r in rows]
    if any(m is None for m in baseline):raise RuntimeError('baseline training cycle unresolved')
    lo=np.full(len(free),-2.);hi=np.full(len(free),2.)
    def candidate(x):
        changes={key:float(v*s) for key,v,s in zip(free,x,scale)}
        three,_,checks,fold=anchored_fold(base,changes,pivot,dummy_cut,0.,0.)
        return three,checks,fold
    calls=0;best={'loss':float('inf')};nres=3*len(rows)+len(free)
    def residual(x):
        nonlocal calls,best
        calls+=1
        try:
            model,checks,fold=candidate(x)
            values=[metrics(model,row,args.duration) for row in rows]
            if any(m is None or m['min_all']<=0 for m in values):raise ValueError('unresolved or nonpositive training trajectory')
            rr=[]
            for row,m in zip(rows,values):
                rr.extend([(m['Cmin']-row['target_Cmin'])/.03,
                           (m['Cmax']-row['target_Cmax'])/.08,
                           (m['period']-row['target_period'])/5.])
            rr.extend(.01*x);out=np.asarray(rr);loss=float(out@out)
            if loss<best['loss']:
                best={'loss':loss,'x':x.tolist(),'calls':calls,'checks':checks,'fold':fold}
                (args.output/'checkpoint.json').write_text(json.dumps(best,indent=2)+'\n')
        except (ValueError,np.linalg.LinAlgError,OverflowError,FloatingPointError):
            out=np.full(nres,100.)+.001*np.resize(x,nres)
        if calls%20==0:print(args.case,'seed',args.seed,'calls',calls,'best',best['loss'],flush=True)
        return out
    rng=np.random.default_rng(args.seed);start=None
    for tries in range(1,args.max_init_tries+1):
        x=np.zeros(len(free)) if args.seed==0 and tries==1 else rng.uniform(-.6,.6,len(free))
        r=residual(x)
        if np.max(abs(r))<50:start=x.copy();break
    if start is None:raise RuntimeError('no feasible start found; inspect logs or increase max-init-tries')
    initial_loss=float(r@r)
    print('valid_start',args.case,args.seed,'attempt',tries,'loss',initial_loss,'x',start.tolist(),flush=True)
    def jac(x):
        f=residual(x);cols=[]
        for j in range(len(x)):
            step=.01 if x[j]+.01<=hi[j] else -.01
            xx=x.copy();xx[j]+=step;cols.append((residual(xx)-f)/step)
        return np.column_stack(cols)
    opt=least_squares(residual,start,jac=jac,bounds=(lo,hi),x_scale='jac',
                      max_nfev=args.max_nfev,ftol=1e-4,xtol=1e-4,gtol=1e-4)
    x=np.array(best['x']);model,checks,fold=candidate(x)
    try:
        orb=Orbit(model);verified=orb.point(np.array(fold['times']),fold['I'],1e-9,require_physical=False)
        verified.pop('physical',None);fold['independent']=verified
        fold['nondegeneracy']=nondegeneracy(orb,np.r_[fold['times'],fold['I']])
    except ValueError as exc:fold['independent_error']=str(exc)
    predictions=[metrics(model,row,args.duration) for row in rows]
    holdout=[{'I':row['I'],'baseline':metrics(base,row,args.duration),'candidate':metrics(model,row,args.duration),
              'target':{k:row[k] for k in ('target_Cmin','target_Cmax','target_period')}} for row in base['holdout_cycles']]
    report={'case':args.case,'arm':'three','seed':args.seed,'start_attempts':tries,'start_x':start.tolist(),
       'training_mode':args.training,'duration_s':args.duration,'initial_squared_residual':initial_loss,
       'best_squared_residual':best['loss'],
       'baseline_squared_residual':sum(((m['Cmin']-r['target_Cmin'])/.03)**2+
           ((m['Cmax']-r['target_Cmax'])/.08)**2+((m['period']-r['target_period'])/5.)**2 for m,r in zip(baseline,rows)),
       'optimizer':{'success':bool(opt.success),'message':str(opt.message),'nfev':int(opt.nfev),'calls':calls},
       'source_sha256':sha,'free_coefficients':[list(v) for v in free],'scales':scale.tolist(),
       'original_target_inputs':base['fixed_equilibrium_threshold_I'],'rest_target':base['rest_target'],
       'checks':checks,'fold':fold,'training_I':[r['I'] for r in rows],
       'baseline':baseline,'candidate':predictions,
       'target':[{k:r[k] for k in ('target_Cmin','target_Cmax','target_period')} for r in rows],
       'holdout':holdout,'scope':'local anchored search; sampled fold itinerary, not a global proof'}
    # Remove metadata that describe the original fit rather than this model.
    output=copy.deepcopy(model)
    for k in ('status','scope','optimizer','loss','selected','parameters','event_q','fold_constraint','training_cycles','holdout_cycles'):
        output.pop(k,None)
    output['status']='exploratory_unvalidated'
    output['anchored_fit']={'seed':args.seed,'source_sha256':sha,'checks':checks,'fold':fold,
                            'best_x':x.tolist(),'free_coefficients':[list(v) for v in free]}
    (args.output/'candidate.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('done',args.case,args.seed,'initial',initial_loss,'best',best['loss'],str(opt.message),flush=True)


if __name__=='__main__':main()
