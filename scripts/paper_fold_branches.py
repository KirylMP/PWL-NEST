"""Finite boundary-matching search near known mixed-model cycle folds."""
import argparse,concurrent.futures,json,os,sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'paper/figures/figure6_data'

def work(task):
    mode,case,target,guess,root=task
    sys.path.insert(0,str(root/'astrocyte_lab/experiments/lower_hlb'))
    from audit import load
    from fold import BoundaryFold
    d,model,x,m,inv=load(DATA/mode/case/'parameters.json');fold=BoundaryFold(model,target,first=0,last=1)
    scale=max(1.,np.max(np.abs(d.get('event_q',[1.]))))
    def f(t):
        try:
            v=fold.matching(x,t)/scale
            return v if np.isfinite(v).all() else np.full(4,1e4)
        except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError):return np.full(4,1e4)
    out=least_squares(f,guess,bounds=(np.full(4,1e-5),np.full(4,120.)),max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    if max(abs(f(out.x)))>1e-8:return None
    try:
        residual,_,z,q,mat=fold.matching(x,out.x,True);states=[]
        for j,(r,M) in enumerate(zip(fold.regions,mat)):
            y=np.r_[z[j],q[j],1.];speed=float((M@y)[0])
            if (1 if j<2 else -1)*speed<=1e-8:return None
            samples=np.array([(expm(M*t)@y)[:2] for t in np.linspace(0,out.x[j],101)])
            low=-np.inf if r==0 else m['th'][r-1];high=np.inf if r==2 else m['th'][r]
            if np.any(samples[:,0]<low-1e-7) or np.any(samples[:,0]>high+1e-7):return None
            states.extend(samples)
        C=inv(np.asarray(states)[:,0],m['shape']);period=float(sum(out.x));logm=float(sum(m['T'][r]*t for r,t in zip(fold.regions,out.x)))
        if abs(logm)<=1e-4:return None
        def rhs(t,y):
            r=int(np.searchsorted(m['th'],y[0],side='right'));v=m['co']@np.r_[1.,y[0],y[1],target,np.maximum(y[0]-m['padded'],0.)];return [v[0],v[1],m['T'][r]]
        direction=1 if logm<0 else -1
        initial=np.array([z[0],q[0],0.]);ivp=solve_ivp(rhs,(0,direction*period),initial,method='DOP853',rtol=2e-9,atol=2e-11,max_step=.05)
        closure=float(np.max(np.abs((ivp.y[:2,-1]-initial[:2])/np.array([1.,scale]))))
        if not ivp.success or closure>1e-5 or abs(ivp.y[2,-1]-direction*logm)>1e-4:return None
        return dict(I=target,Cmin=float(min(C)),Cmax=float(max(C)),period=period,log_multiplier=logm,stable=logm<0,closure=closure,matching_error=float(max(abs(residual))))
    except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError):return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,default=ROOT.parent/'PWL-NEST');p.add_argument('--workers',type=int,default=6);a=p.parse_args();root=a.model_root.resolve()
    sys.path.insert(0,str(root/'astrocyte_lab/experiments/lower_hlb'));from audit import load;from fold import BoundaryFold
    archive=json.loads((ROOT/'paper/data/two_threshold_models/sources/mixed_fold_audit.json').read_text());seeds={}
    for case in ('AM','AFM','FM'):
        d=json.loads((DATA/'fixed'/case/'parameters.json').read_text());info=d['fold_constraint'];seeds['fixed',case]=(info['I'],np.array(info['crossing_times']))
    for case in ('AM','AFM'):
        row=next(v for v in archive['results'] if v.get('model')==case.lower() and v.get('status')=='candidate_fold' and v.get('itinerary')=='both_thresholds')
        seeds['no_fold',case]=(row['I'],np.asarray(row['times']))
    tasks=[]
    for (mode,case),(I,times) in seeds.items():
        _,model,x,m,_=load(DATA/mode/case/'parameters.json');fold=BoundaryFold(model,I,first=0,last=1)
        step=1e-5;J=np.column_stack([(fold.matching(x,times+step*e)-fold.matching(x,times-step*e))/(2*step) for e in np.eye(4)]);right=np.linalg.svd(J)[2][-1]
        for delta in (-.00005,-.0002,-.001,-.003,-.007,-.015,-.03,-.06):
            for displacement in (-1.,-.25,.25,1.):
                guess=np.clip(times+displacement*right,1e-4,119.9);tasks.append((mode,case,float(I+delta),guess,root))
    print('fold branch attempts',len(tasks),flush=True);found={f'{m}_{c}':[] for m,c in seeds}
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for task,result in zip(tasks,pool.map(work,tasks)):
            if result is None:continue
            key=f'{task[0]}_{task[1]}'
            if not any(abs(r['I']-result['I'])<1e-10 and abs(r['Cmax']-result['Cmax'])<1e-5 for r in found[key]):found[key].append(result);print('CYCLE',key,result,flush=True)
    (DATA/'fold_branch_search.json').write_text(json.dumps(dict(found=found,attempts=len(tasks),scope='Finite multistart matching near stored all-threshold folds; roots require sampled itinerary and independent forward/backward DOP853 closure. Missing branches do not prove absence.'),indent=2)+'\n')
    print('DONE',{k:len(v) for k,v in found.items()},flush=True)
if __name__=='__main__':main()
