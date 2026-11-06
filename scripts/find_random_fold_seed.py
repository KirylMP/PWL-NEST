"""Independent random field initialization; fixed anchors and neutral cycle.

No fitted field or fitted cycle times are used to initialize the search.
An unsuccessful solve is not evidence that no candidate exists.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from check_focus_feasibility import anchored,constraints
from continue_physical_fixed import Orbit,REGIONS

ROOT=Path(__file__).resolve().parents[1]


def draw_start(base,seed,case,draws,bound,margin):
    rng=np.random.default_rng(np.random.SeedSequence([seed,{'AM':1,'AFM':2,'FM':3}[case]]))
    for attempt in range(1,draws+1):
        x=rng.uniform(-bound,bound,6)
        if np.min(constraints(base,x,margin))>=0:
            times=np.exp(rng.uniform(np.log(.05),np.log(50.),4))
            return x,times,attempt
    raise ValueError(f'no random field passed spectral screen in {draws} draws')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True,choices=['AM','AFM','FM']);p.add_argument('--seed',required=True,type=int);p.add_argument('--max-nfev',type=int,default=300);p.add_argument('--bound',type=float,default=40.);p.add_argument('--max-field-draws',type=int,default=1000);p.add_argument('--screen-margin',type=float,default=1e-4);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if min(a.max_nfev,a.max_field_draws)<1 or min(a.bound,a.screen_margin)<=0 or a.seed<0:p.error('invalid settings')
    base=json.loads((ROOT/'paper/data/physical_models/fixed_fold'/a.case/'parameters.json').read_text());I=base['fold_constraint']['I'];a.output.parent.mkdir(parents=True,exist_ok=True)
    try:x0,t0,draws=draw_start(base,a.seed,a.case,a.max_field_draws,a.bound,a.screen_margin)
    except ValueError as e:
        a.output.write_text(json.dumps({'case':a.case,'seed':a.seed,'status_validation':'no_spectral_start','reason':str(e)},indent=2)+'\n');print(a.case,a.seed,str(e),flush=True);return
    calls=0;best={'loss':float('inf')}
    def model(x):
        n,h,T,D,_=anchored(base,x);d=copy.copy(base);d.update(alpha=n.tolist(),beta=h.tolist(),trace=T.tolist(),determinant=D.tolist());return d
    def fun(w):
        nonlocal calls,best
        calls+=1;x=w[:6];t=w[6:]
        try:
            d=model(x);orb=Orbit(d);ret=orb.match(t,I);log=float(orb.T[list(REGIONS)]@t);c=constraints(base,x,a.screen_margin)
            vec=np.r_[ret/.01,log/5,np.minimum(c,0.)*20.]
            if not np.all(np.isfinite(vec)):raise ValueError('nonfinite')
            loss=float(vec@vec)
            if loss<best['loss']:best={'loss':loss,'x':x.tolist(),'times':t.tolist(),'return_max':float(max(abs(ret))),'logM':log,'minimum_spectral_constraint':float(min(c))}
        except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):vec=np.full(12,1e5)
        if calls%500==0:print(a.case,a.seed,'calls',calls,'best',best,flush=True)
        return vec
    x0=np.r_[x0,t0];lo=np.r_[np.full(6,-a.bound),np.full(4,1e-5)];hi=np.r_[np.full(6,a.bound),np.full(4,120.)]
    res=least_squares(fun,np.clip(x0,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=a.max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8,x_scale='jac')
    record={'case':a.case,'seed':a.seed,'source':'random independent x and random flight times; fixed underlying a,b,anchors and scales',
        'settings':vars(a)|{'output':str(a.output)},'initial_x':x0[:6].tolist(),'initial_times':x0[6:].tolist(),
        'spectral_draws':draws,'nfev':int(res.nfev),'calls':calls,'optimizer_status':str(res.message),'best':best}
    if 'x' not in best:record['best']={'loss':None};record['status_validation']='unresolved'
    else:
        x=np.array(best['x']);t=np.array(best['times']);d=model(x);orb=Orbit(d)
        if best['return_max']<1e-7 and abs(best['logM'])<1e-7 and best['minimum_spectral_constraint']>=-1e-7:
            try:record['orbit']=orb.point(t,I,1e-9,require_physical=False);record['status_validation']='validated_sampled'
            except (ValueError,OverflowError,FloatingPointError) as e:record['status_validation']='rejected_orbit';record['validation_error']=str(e)
        else:record['status_validation']='unresolved'
    a.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print(a.case,a.seed,record['status_validation'],'draws',draws,'best',best,flush=True)

if __name__=='__main__':main()
