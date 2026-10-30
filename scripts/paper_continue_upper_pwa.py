"""Continue unstable orbits that cross only the upper PWA threshold.

Seeds come from the end of both-threshold fold continuation. Finite numerical
continuation with sampled itineraries and independent backward IVP checks.
"""
import argparse,concurrent.futures,json,os,sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'paper/figures/figure6_data'

def work(task):
    mode,case,endpoint,root,count=task
    sys.path.insert(0,str(root/'astrocyte_lab/experiments/lower_hlb'))
    from audit import load
    from fold import BoundaryFold
    d,model,x,m,inv=load(DATA/mode/case/'parameters.json');upper=float(model.events[1]);I0=endpoint['I'];qscale=max(1.,np.max(np.abs(d.get('event_q',[1.]))));t=endpoint['flight_times'];guess=np.array([t[1],t[0]+t[2]])
    if I0<=upper:return dict(mode=mode,case=case,points=[],reason='grazing estimate not above upper crossing')
    # Stay away from both endpoint and upper BEB; the latter may require arbitrary precision.
    inputs=upper+(I0-upper)*np.linspace(.995,.005,count);rows=[]
    for I in inputs:
        fold=BoundaryFold(model,I,first=1,last=1)
        def fun(t):
            try:
                v=fold.matching(x,t)/qscale
                return v if np.isfinite(v).all() else np.full(2,1e4)
            except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError):return np.full(2,1e4)
        try:
            sol=least_squares(fun,guess,bounds=(np.full(2,1e-5),np.full(2,240.)),max_nfev=180,ftol=1e-11,xtol=1e-11,gtol=1e-11)
            if max(abs(fun(sol.x)))>1e-8:continue
            residual,_,z,q,mats=fold.matching(x,sol.x,True);states=[]
            for j,(r,M) in enumerate(zip(fold.regions,mats)):
                y0=np.r_[z[j],q[j],1.];speed=float((M@y0)[0])
                if (1 if j==0 else -1)*speed<=1e-9:raise ValueError('wrong direction')
                orbit=np.array([(expm(M*t)@y0)[:2] for t in np.linspace(0,sol.x[j],121)])
                lo=m['th'][r-1];hi=np.inf if r==2 else m['th'][r]
                if np.any(orbit[:,0]<lo-1e-7) or np.any(orbit[:,0]>hi+1e-7):raise ValueError('itinerary changed')
                states.extend(orbit)
            C=inv(np.asarray(states)[:,0],m['shape']);period=float(sum(sol.x));logm=float(sum(m['T'][r]*dt for r,dt in zip(fold.regions,sol.x)))
            if logm<=1e-4:raise ValueError('not unstable')
            def rhs(t,y):
                r=int(np.searchsorted(m['th'],y[0],side='right'));v=m['co']@np.r_[1.,y[0],y[1],I,np.maximum(y[0]-m['padded'],0.)];return [v[0],v[1],m['T'][r]]
            initial=np.array([z[0],q[0],0.]);s=solve_ivp(rhs,(0,-period),initial,method='DOP853',rtol=2e-9,atol=2e-11,max_step=.05)
            closure=float(np.max(np.abs((s.y[:2,-1]-initial[:2])/np.array([1.,qscale]))))
            if not s.success or closure>1e-5 or abs(s.y[2,-1]+logm)>1e-4:raise ValueError('backward closure unresolved')
            rows.append(dict(I=float(I),Cmin=float(min(C)),Cmax=float(max(C)),period=period,log_multiplier=logm,stable=False,closure=closure,matching_error=float(max(abs(residual))),itinerary='upper_threshold_only'));guess=sol.x
        except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError):continue
    print(mode,case,'upper-only verified',len(rows),'/',count,flush=True)
    return dict(mode=mode,case=case,points=rows,reason='finite matching scan; missed inputs unresolved')

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,default=ROOT.parent/'PWL-NEST');p.add_argument('--workers',type=int,default=6);p.add_argument('--points',type=int,default=65);a=p.parse_args()
    old=json.loads((DATA/'continued_pwa_cycles.json').read_text())['results'];tasks=[]
    for branch in old:
        points=[p for p in branch['points'] if not p['stable']]
        if points:tasks.append((branch['mode'],branch['case'],points[-1],a.model_root.resolve(),a.points))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(work,tasks))
    (DATA/'upper_pwa_cycles.json').write_text(json.dumps(dict(results=results,scope='Finite upper-threshold boundary shooting; verified sampled itinerary and backward DOP853 closure. No completeness claim.'),indent=2)+'\n')
if __name__=='__main__':main()
