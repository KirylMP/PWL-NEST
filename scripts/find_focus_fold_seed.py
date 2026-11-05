"""Find one joint focus/fold seed before dependent-pivot optimization.

Joint numerical solve is for initialization; it is not a fit of cycle data.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from check_focus_feasibility import anchored,constraints
from anchored_lower_fold import FREE
from continue_physical_fixed import Orbit,REGIONS

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True,choices=['AM','AFM','FM']);p.add_argument('--screen',type=Path,default=ROOT/'results/focus_screen/margin_001.json');p.add_argument('--max-nfev',type=int,default=100);p.add_argument('--margin',type=float,default=.01);p.add_argument('--start',choices=['spectrum','selected'],default='spectrum');p.add_argument('--bounded-times',action='store_true');p.add_argument('--random-seed',type=int,default=0);p.add_argument('--jitter',type=float,default=0.);p.add_argument('--focus-weight',type=float,default=3.);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    base=json.loads((ROOT/'paper/data/physical_models/fixed_fold'/a.case/'parameters.json').read_text());screen=json.loads(a.screen.read_text())['results'][a.case]['best'];reference_seed={'AM':7,'AFM':22,'FM':8}[a.case];reference_path=ROOT/'results/anchored_three_main'/a.case/f'seed_{reference_seed}'/'attempt_001/candidate.json';selected=json.loads(reference_path.read_text());report={'fold':selected['anchored_fit']['fold']};I=base['fold_constraint']['I'];x_init=screen['x'] if a.start=='spectrum' else [(selected[k][j]-base[k][j])/(.05*max(abs(base[k][j]),.3)) for k,j in FREE];time_init=report['fold']['times'];
    x0=np.r_[x_init,time_init];
    if a.jitter:
        rng=np.random.default_rng(a.random_seed);x0[:6]=np.clip(x0[:6]+rng.normal(0,a.jitter,6),-39.,39.);x0[6:]=np.clip(x0[6:]*np.exp(rng.normal(0,a.jitter*.12,4)),.01,90.)
    calls=0;best={'loss':float('inf')};a.output.parent.mkdir(parents=True,exist_ok=True)
    def model(x):
        n,h,T,D,err=anchored(base,x);d=copy.copy(base);d.update(alpha=n.tolist(),beta=h.tolist(),trace=T.tolist(),determinant=D.tolist());return d
    def fun(w):
        nonlocal calls,best
        calls+=1;x=w[:6];t=w[6:]
        try:
            d=model(x);orb=Orbit(d);ret=orb.match(t,I);log=float(orb.T[list(REGIONS)]@t);c=constraints(base,x,a.margin)
            vec=np.r_[ret/.01,log/5,np.minimum(c,0.)*a.focus_weight,.00001*x]
            if not np.all(np.isfinite(vec)):raise ValueError('nonfinite')
            loss=float(vec@vec)
            if loss<best['loss']:
                best={'loss':loss,'x':x.tolist(),'times':t.tolist(),'return_max':float(max(abs(ret))),'logM':log,'minimum_focus_constraint':float(min(c))}
        except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):vec=np.full(18,1e5)
        if calls%300==0:print(a.case,'calls',calls,'best',best,flush=True)
        return vec
    lo=np.r_[np.full(6,-40.),np.maximum(np.array(report['fold']['times'])*.1,1e-4) if a.bounded_times else np.full(4,1e-5)];hi=np.r_[np.full(6,40.),np.maximum(np.array(report['fold']['times'])*5,3.) if a.bounded_times else np.full(4,120.)]
    x0=np.clip(x0,lo+1e-6,hi-1e-6)
    res=least_squares(fun,x0,bounds=(lo,hi),max_nfev=a.max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8,x_scale='jac')
    x=np.array(best['x']);t=np.array(best['times']);d=model(x);orb=Orbit(d)
    rec={'case':a.case,'settings':vars(a)|{'screen':str(a.screen),'output':str(a.output)},'scope':'joint initial feasibility only; not optimized to LR cycle data; finite numerical evidence','nfev':int(res.nfev),'calls':calls,'status':str(res.message),'best':best,'trace':d['trace'],'determinant':d['determinant'],'discriminant':(np.array(d['trace'])**2-4*np.array(d['determinant'])).tolist()}
    if best['return_max']<1e-7 and abs(best['logM'])<1e-7 and best['minimum_focus_constraint']>=-1e-7:
        try:rec['orbit']=orb.point(t,I,1e-9,require_physical=False);rec['status_validation']='validated_sampled'
        except Exception as e:rec['status_validation']='rejected_orbit';rec['validation_error']=str(e)
    else:rec['status_validation']='unresolved'
    a.output.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');print(a.case,rec['status_validation'],best,flush=True)

if __name__=='__main__':main()
