"""Continue all-threshold PWA cycles from supplied fold points.

Pseudo-arclength corrector on the affine boundary matching equations.  Each
plotted orbit has sampled itinerary checks and an independent forward/backward
DOP853 closure check. Failed continuations are reported, not extrapolated.
"""
import argparse,concurrent.futures,json,os,sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'paper/figures/figure6_data'
SCALE=np.array([10.,10.,10.,10.,.1])

def run(task):
    mode,case,I0,t0,sign,root,steps=task
    sys.path.insert(0,str(root/'astrocyte_lab/experiments/lower_hlb'))
    from audit import load
    from fold import BoundaryFold
    d,model,x,m,inv=load(DATA/mode/case/'parameters.json');fold=BoundaryFold(model,I0,first=0,last=1);qscale=max(1.,np.max(np.abs(d.get('event_q',[1.]))))
    def fn(w):
        try:
            p=w*SCALE;fold.I=float(p[4]);r=fold.matching(x,p[:4])/qscale
            if np.all(np.isfinite(r)):return r
        except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):pass
        return np.full(4,1e5)
    def jac(w):
        dh=1e-5;return np.column_stack([(fn(w+dh*v)-fn(w-dh*v))/(2*dh) for v in np.eye(5)])
    start=np.r_[t0,I0]/SCALE;w=start.copy();tangent=np.linalg.svd(jac(w),full_matrices=True)[2][-1]*sign;size=.012;points=[];reason='step limit'
    lower=np.r_[np.full(4,1e-6)/SCALE[:4],0.];upper=np.r_[np.full(4,120.)/SCALE[:4],1.35/SCALE[4]]
    for k in range(steps):
        predicted=w+size*tangent
        if np.any(predicted<=lower) or np.any(predicted>=upper):
            size*=.5
            if size<2e-5:reason='parameter or flight-time bound';break
            continue
        def augmented(v):return np.r_[fn(v),np.dot(v-predicted,tangent)]
        try:sol=least_squares(augmented,predicted,bounds=(lower,upper),max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):sol=None
        if sol is None or max(abs(augmented(sol.x)))>1e-8:
            size*=.5
            if size<2e-5:reason='corrector unresolved';break
            continue
        wnew=sol.x;actual=wnew*SCALE;fold.I=float(actual[4])
        try:
            residual,_,z,q,mats=fold.matching(x,actual[:4],True);states=[]
            for j,(r,A) in enumerate(zip(fold.regions,mats)):
                y0=np.r_[z[j],q[j],1.];speed=float((A@y0)[0])
                if (1 if j<2 else -1)*speed<=1e-8:raise ValueError('grazing or wrong crossing')
                orbit=np.array([(expm(A*t)@y0)[:2] for t in np.linspace(0,actual[j],101)])
                lo=-np.inf if r==0 else m['th'][r-1];hi=np.inf if r==2 else m['th'][r]
                if np.any(orbit[:,0]<lo-1e-7) or np.any(orbit[:,0]>hi+1e-7):raise ValueError('itinerary change')
                states.extend(orbit)
            C=inv(np.asarray(states)[:,0],m['shape']);period=float(sum(actual[:4]));logm=float(sum(m['T'][r]*dt for r,dt in zip(fold.regions,actual[:4])))
            def rhs(t,y):
                region=int(np.searchsorted(m['th'],y[0],side='right'));v=m['co']@np.r_[1.,y[0],y[1],actual[4],np.maximum(y[0]-m['padded'],0.)];return [v[0],v[1],m['T'][region]]
            direction=-1 if logm>0 else 1;initial=np.array([z[0],q[0],0.]);ivp=solve_ivp(rhs,(0,direction*period),initial,method='DOP853',rtol=2e-9,atol=2e-11,max_step=.05)
            closure=float(np.max(np.abs((ivp.y[:2,-1]-initial[:2])/np.array([1.,qscale]))))
            if not ivp.success or closure>1e-5 or abs(ivp.y[2,-1]-direction*logm)>1e-4:raise ValueError('IVP closure unresolved')
            points.append(dict(I=float(actual[4]),Cmin=float(min(C)),Cmax=float(max(C)),period=period,log_multiplier=logm,stable=bool(logm<0),closure=closure,matching_error=float(max(abs(residual))),flight_times=actual[:4].tolist(),itinerary='both_thresholds'))
        except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as exc:
            reason=str(exc);break
        new_tangent=np.linalg.svd(jac(wnew),full_matrices=True)[2][-1]
        if np.dot(new_tangent,tangent)<0:new_tangent=-new_tangent
        w=wnew;tangent=new_tangent;size=min(size*1.08,.035)
    print(mode,case,'sign',sign,'points',len(points),'unstable',sum(not p['stable'] for p in points),'reason',reason,flush=True)
    return dict(mode=mode,case=case,sign=sign,reason=reason,points=points)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,default=ROOT.parent/'PWL-NEST');p.add_argument('--workers',type=int,default=6);p.add_argument('--steps',type=int,default=80);a=p.parse_args()
    archive=json.loads((ROOT/'paper/data/two_threshold_models/sources/mixed_fold_audit.json').read_text());seeds={}
    for case in ('AM','AFM','FM'):
        d=json.loads((DATA/'fixed'/case/'parameters.json').read_text());fold=d['fold_constraint'];seeds['fixed',case]=(fold['I'],fold['crossing_times'])
    for case in ('AM','AFM'):
        r=next(v for v in archive['results'] if v.get('model')==case.lower() and v.get('status')=='candidate_fold' and v.get('itinerary')=='both_thresholds');seeds['no_fold',case]=(r['I'],r['times'])
    fm_path=DATA/'no_fold_FM_fold_search.json'
    if fm_path.exists():
        f=json.loads(fm_path.read_text()); candidates=[r for r in f['results'] if r.get('status')=='candidate_fold' and r.get('itinerary')=='both_thresholds']
        if candidates:r=candidates[0];seeds['no_fold','FM']=(r['I'],r['times'])
    tasks=[(mode,case,I,t,sign,a.model_root.resolve(),a.steps) for (mode,case),(I,t) in seeds.items() for sign in (-1,1)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(run,tasks))
    (DATA/'continued_pwa_cycles.json').write_text(json.dumps(dict(results=results,scope='Finite pseudo-arclength continuation on the both-threshold itinerary; sampled itinerary plus independent DOP853 closure, not a global cycle inventory.'),indent=2)+'\n')
if __name__=='__main__':main()
