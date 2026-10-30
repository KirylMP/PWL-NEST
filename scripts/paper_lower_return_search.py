"""Finite multistart search for cycles crossing both thresholds near onset.

Needs the model tools in PWL-NEST. Returns numerical candidates, not a global
cycle count or proof of nonexistence; saves all attempted input values.
"""
import argparse,concurrent.futures,json,os,sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'paper/figures/figure5_data'

def search(task):
    name,I,seed,model_root=task
    sys.path.insert(0,str(model_root/'astrocyte_lab/experiments/lower_hlb'))
    from audit import load
    from fold import BoundaryFold
    d,model,x,m,inv=load(DATA/'models'/f'{name}.json');fold=BoundaryFold(model,I,first=0,last=1);rng=np.random.default_rng(seed)
    n=fold.legs;initial=np.exp(rng.uniform(np.log(.2),np.log(25),n));qscale=max(1.,np.max(np.abs(d.get('event_q',[1.]))))
    def fun(t):
        try:
            y=fold.matching(x,t)/qscale;return y if np.isfinite(y).all() else np.full(n,1e4)
        except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):return np.full(n,1e4)
    result=least_squares(fun,initial,bounds=(np.full(n,.002),np.full(n,90.)),max_nfev=150,ftol=2e-9,xtol=2e-9,gtol=2e-9,x_scale='jac')
    if max(abs(fun(result.x)))>1e-8:return None
    try:
        residual,_,z,q,mats=fold.matching(x,result.x,True);points=[]
        for j,(r,M) in enumerate(zip(fold.regions,mats)):
            y0=np.r_[z[j],q[j],1.];speed=float((M@y0)[0])
            if (1 if j<2 else -1)*speed<=1e-8:return None
            ys=np.array([(expm(M*t)@y0)[:2] for t in np.linspace(0,result.x[j],81)])
            low=-np.inf if r==0 else m['th'][r-1];high=np.inf if r==2 else m['th'][r]
            if np.any(ys[:,0]<low-1e-7) or np.any(ys[:,0]>high+1e-7):return None
            points.extend(ys)
        points=np.asarray(points);C=inv(points[:,0],m['shape']);period=float(sum(result.x));logm=float(sum(m['T'][r]*dt for r,dt in zip(fold.regions,result.x)))
        def ode(t,y):
            region=int(np.searchsorted(m['th'],y[0],side='right'));f=m['co']@np.r_[1.,y[0],y[1],I,np.maximum(y[0]-m['padded'],0.)];return [f[0],f[1],m['T'][region]]
        y0=np.array([z[0],q[0],0.]);direction=-1 if logm>0 else 1
        # Start backward integration from the same point on the closed orbit.
        ivp=solve_ivp(ode,(0,direction*period),y0,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.06)
        err=float(np.max(np.abs((ivp.y[:2,-1]-y0[:2])/np.array([1.,qscale]))))
        if not ivp.success or err>1e-5 or abs(ivp.y[2,-1]-direction*logm)>1e-4:return None
        return dict(I=I,Cmin=float(min(C)),Cmax=float(max(C)),period=period,log_multiplier=logm,stable=logm<0,closure=err,seed=seed)
    except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,default=ROOT.parent/'PWL-NEST');p.add_argument('--workers',type=int,default=6);p.add_argument('--starts',type=int,default=5);a=p.parse_args()
    tasks=[];inputs={}
    for name,width in [('AFM',.008),('FM',.004)]:
        d=json.loads((DATA/'models'/f'{name}.json').read_text());base=d['fixed_equilibrium_threshold_I'][0]
        inputs[name]=[float(base+v) for v in np.linspace(-width,width,9)]
        tasks.extend((name,I,10000+i*100+j,a.model_root.resolve()) for i,I in enumerate(inputs[name]) for j in range(a.starts))
    found={name:[] for name in inputs}
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for task,r in zip(tasks,pool.map(search,tasks)):
            if r is None:continue
            group=found[task[0]]
            if not any(abs(s['I']-r['I'])<1e-12 and abs(s['Cmax']-r['Cmax'])<1e-5 for s in group):group.append(r);print('FOUND',task[0],r,flush=True)
    (DATA/'pwa_both_threshold_search.json').write_text(json.dumps(dict(inputs=inputs,starts=a.starts,found=found,scope='Finite multistart boundary matching, sampled itinerary, independent DOP853 closure; a failure does not prove absence.'),indent=2)+'\n')
    print('DONE', {k:len(v) for k,v in found.items()},flush=True)
if __name__=='__main__':main()
