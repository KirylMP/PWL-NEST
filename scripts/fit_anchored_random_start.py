"""Single random-start local fit with exact rest, threshold and fold anchors."""
import argparse
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
    p.add_argument('--case',choices=['AM','AFM','FM'],default='AFM');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-nfev',type=int,default=12);p.add_argument('--duration',type=float,default=400.)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    path=ROOT/'paper/data/physical_models/fixed_fold'/args.case/'parameters.json';base=json.loads(path.read_text())
    pivot=PIVOTS[args.case];free=[v for v in FREE if v!=pivot]
    scale=np.array([.05*max(abs(base[k][j]),.3) for k,j in free])
    audit=json.loads((ROOT/'paper/data/physical_models/fixed_fold/fold_audit.json').read_text())['results'][args.case.lower()]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=audit['sha256']:raise RuntimeError('source hash mismatch')
    cut_hi=min(.95*base['physical_thresholds_C'][0],.9*audit['neutral']['Cmin']);cut_lo=.25*cut_hi
    rows=base['training_cycles']; baseline=[metrics(base,r,args.duration) for r in rows]
    if any(v is None for v in baseline):raise RuntimeError('baseline training cycle unresolved')
    lo=np.r_[np.full(len(free),-2.),0.,-2.,-2.];hi=np.r_[np.full(len(free),2.),1.,2.,2.]
    def candidate(x):
        changes={key:float(v*s) for key,v,s in zip(free,x,scale)}
        cut=float(cut_lo+(cut_hi-cut_lo)*x[-3]);da=float(x[-2]);db=float(x[-1])
        return anchored_fold(base,changes,pivot,cut,da,db)
    calls=0;best={'loss':float('inf')};nres=3*len(rows)+len(lo)
    def residual(x):
        nonlocal calls,best
        calls+=1
        try:
            three,model,checks,fold=candidate(x)
            values=[metrics(model,row,args.duration) for row in rows]
            if any(m is None or m['min_all']<=0 for m in values):raise ValueError('unresolved or nonpositive training trajectory')
            rr=[]
            for row,m in zip(rows,values):rr.extend([(m['Cmin']-row['target_Cmin'])/.03,
                (m['Cmax']-row['target_Cmax'])/.08,(m['period']-row['target_period'])/5.])
            rr.extend(.01*x);out=np.asarray(rr);loss=float(out@out)
            if loss<best['loss']:
                best={'loss':loss,'x':x.tolist(),'calls':calls,'checks':checks,'fold':fold}
                (args.output/'checkpoint.json').write_text(json.dumps(best,indent=2)+'\n')
        except (ValueError,np.linalg.LinAlgError,OverflowError,FloatingPointError):
            out=np.full(nres,100.)+.001*np.resize(x,nres)
        if calls%20==0:print('calls',calls,'best',best['loss'],flush=True)
        return out
    rng=np.random.default_rng(args.seed);start=None
    for tries in range(1,101):
        x=np.r_[rng.uniform(-.25,.25,len(free)),rng.uniform(.2,.8),rng.uniform(-.1,.1,2)]
        r=residual(x)
        if np.max(abs(r))<50:start=x.copy();break
    if start is None:raise RuntimeError('no feasible random starting point in 100 attempts')
    initial_loss=float(r@r);print('valid_random_start',tries,'loss',initial_loss,'x',start.tolist(),flush=True)
    def jac(x):
        f=residual(x);cols=[]
        for j in range(len(x)):
            step=.01 if x[j]+.01<=hi[j] else -.01
            xx=x.copy();xx[j]+=step;cols.append((residual(xx)-f)/step)
        return np.column_stack(cols)
    opt=least_squares(residual,start,jac=jac,bounds=(lo,hi),x_scale='jac',
                      max_nfev=args.max_nfev,ftol=1e-4,xtol=1e-4,gtol=1e-4)
    x=np.array(best['x']);three,model,checks,fold=candidate(x)
    try:
        verified=Orbit(three).point(np.array(fold['times']),fold['I'],1e-9,require_physical=False)
        verified.pop('physical',None);fold['independent']=verified
        fold['nondegeneracy']=nondegeneracy(Orbit(three),np.r_[fold['times'],fold['I']])
    except ValueError as exc:fold['independent_error']=str(exc)
    predictions=[metrics(model,r,args.duration) for r in rows]
    heldout=[{'I':r['I'],'baseline':metrics(base,r,args.duration),'candidate':metrics(model,r,args.duration),
              'target':{k:r[k] for k in ('target_Cmin','target_Cmax','target_period')}} for r in base['holdout_cycles']]
    report={'case':args.case,'seed':args.seed,'start_attempts':tries,'start_x':start.tolist(),
            'initial_squared_residual':initial_loss,'best_squared_residual':best['loss'],
            'optimizer':{'success':bool(opt.success),'message':str(opt.message),'nfev':int(opt.nfev),'calls':calls},
            'source_sha256':audit['sha256'],'free_coefficients':[list(v) for v in free],'scales':scale.tolist(),
            'original_target_inputs':base['fixed_equilibrium_threshold_I'],'rest_target':base['rest_target'],
            'checks':checks,'fold':fold,'cut':model['physical_thresholds_C'][0],
            'training_I':[r['I'] for r in rows],'baseline':baseline,'candidate':predictions,
            'target':[{k:r[k] for k in ('target_Cmin','target_Cmax','target_period')} for r in rows],
            'holdout':heldout,'duration_s':args.duration,
            'scope':'single local random start, sampled itinerary, not a global or rigorous proof'}
    model['anchored_random_fit']={'status':'exploratory','seed':args.seed,'source_sha256':audit['sha256'],
          'checks':checks,'fold':fold,'best_x':x.tolist(),
          'warning':'Other optimizer and cycle fields are inherited and stale.'}
    (args.output/'candidate.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('done initial',initial_loss,'best',best['loss'],'status',str(opt.message),flush=True)


if __name__=='__main__':main()
