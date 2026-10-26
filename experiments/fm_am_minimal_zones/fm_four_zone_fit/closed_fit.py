"""Final FM objective: extrema and periods only at actual return-map roots."""
import json
import os
import time
import sys
import numpy as np
from scipy.optimize import brentq, minimize
import fit_four as f
sys.path.insert(0,str(f.P.parent))
from equilibrium_loss import model_cost
from loss_weights import closed_components
EQ_SOURCE=[(float(I),f.eqs(float(I))) for I in np.linspace(.4,1.2,25)]
NO_CYCLE_INPUTS=tuple(float(s.strip()) for s in os.environ.get('FM_NO_CYCLE_INPUTS','1.5,2.0').split(',') if s.strip())


def validate_no_cycle_inputs():
    if not NO_CYCLE_INPUTS or not all(np.isfinite(x) for x in NO_CYCLE_INPUTS):
        raise ValueError('FM_NO_CYCLE_INPUTS must contain finite reference I values')
    rows=json.loads((f.P/'LR_cycles_reference.json').read_text())['rows']
    for I in NO_CYCLE_INPUTS:
        row=min(rows,key=lambda q:abs(q['I']-I))
        if abs(row['I']-I)>1e-8 or row['cycles']:
            raise ValueError(f'I={I} must be present in LR reference with zero detected cycles')


def extra_cycle_penalty(x):
    """Finite-grid search for unwanted cycles at LR no-cycle inputs."""
    m=f.unpack(x)
    bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
    counts=[]; diagnostics=[]
    for I in NO_CYCLE_INPUTS:
        found,diag=f.cycles(I,f.equilibrium(I,m),m['co'],m['padded'],bounds,
                            n=32,tol=1e-9)
        counts.append(len(found)); diagnostics.append(diag)
    return float(sum(n*n for n in counts)),counts,diagnostics


def residual(x, detail=False):
    m=f.unpack(x); bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
    records=[]; result=[]
    for target in f.TARGETS:
        I=target['I']; section=float(f.phi(target['C'],m['shape']))
        n=float(f.basis(target['C'],I,m)@m['N'])
        back=not target['stable']; direction=-1 if back else 1
        low=max(n+1e-7,1e-7); high=1-1e-7
        cache={}
        def calc(h):
            if h not in cache:
                cache[h]=f.ret(I,h,section,m['co'],m['padded'],bounds,
                              back=back,tol=2e-9,fast=False)
            return cache[h]
        def valid(h):
            status,v=calc(float(h))
            return (status==1 and abs(v[0])<3e-8 and v[4]-v[3]>1e-5
                    and direction*v[2]<-1e-6)
        h=float(np.clip(target['h'],low,high)); found=False
        if low<high:
            for _ in range(16):
                status,v=calc(h)
                if status!=1:break
                if valid(h):found=True;break
                derivative=np.expm1(np.clip(direction*v[2],-700,50))
                if abs(derivative)<1e-8:break
                next_h=float(np.clip(h+np.clip(-v[0]/derivative,-.04,.04),low,high))
                if abs(next_h-h)<1e-11:break
                h=next_h
            if not found:
                grid=low+(high-low)*np.r_[np.geomspace(1e-5,.1,12),np.linspace(.1,1,16)]
                vals=[calc(float(s))[1][0] if calc(float(s))[0]==1 else np.nan for s in grid]
                roots=[]
                for a,b,va,vb in zip(grid[:-1],grid[1:],vals[:-1],vals[1:]):
                    if not (np.isfinite(va) and np.isfinite(vb) and va*vb<0):continue
                    def fun(s):
                        status,v=calc(float(s));return v[0] if status==1 else np.nan
                    try:root=float(brentq(fun,a,b,xtol=2e-10))
                    except ValueError:continue
                    if valid(root):roots.append(root)
                if roots:h=min(roots,key=lambda s:abs(s-target['h']));found=True
        if found:
            _,v=calc(h); ext=f.inverse(v[3:5],m)
            rr=[(v[1]-target['T'])/8,(ext[0]-target['Cmin'])/.025,
                (ext[1]-target['Cmax'])/.15]
            record=dict(target=target,found=True,h=h,return_data=v.tolist(),
                        Cmin=float(ext[0]),Cmax=float(ext[1]),residual=rr)
        else:
            rr=[8.,8.,8.];record=dict(target=target,found=False,residual=rr)
        result.extend(rr);records.append(record)
    return records if detail else f.TARGET_NORMALIZATION*np.array(result)


def run():
    validate_no_cycle_inputs()
    seed=json.loads((f.P/'candidate.json').read_text())
    (f.P/'before_closed.json').write_text(json.dumps(seed,indent=2))
    base=np.array(seed['parameters']);scale=np.array([.2,.15,.2,.25,.2,.2,.6,.35,.6,.35,.02,.02,.03,.03,.3,.3,.15,.25])
    bounds=list(zip(np.maximum(-1.,(f.LO-base)/scale),np.minimum(1.,(f.HI-base)/scale)))
    best=np.inf;calls=0;start=time.monotonic()
    extra_weight=float(os.environ.get('FM_EXTRA_CYCLES_WEIGHT','0'))
    if not np.isfinite(extra_weight) or extra_weight<0:
        raise ValueError('FM_EXTRA_CYCLES_WEIGHT must be finite and nonnegative')
    print(f'FM cycle targets={len(f.TARGETS)} inputs={sorted(set(q["I"] for q in f.TARGETS))} residual_scale={f.TARGET_NORMALIZATION:.6g}',flush=True)
    print(f'No-cycle training inputs={NO_CYCLE_INPUTS}, extra-cycle weight={extra_weight}',flush=True)
    def objective(y):
        nonlocal best,calls
        x=base+scale*y;g=f.geometric(x);r=residual(x)
        eq_loss=model_cost(f.unpack(x),EQ_SOURCE,f.equilibrium)
        parts=closed_components(g@g,eq_loss,r@r,6.,f.POINCARE_SCALE)
        extra_raw,extra_counts,extra_diag=extra_cycle_penalty(x) if extra_weight else (0.,[],[])
        parts.update(extra_cycles_raw=extra_raw,extra_cycles_weight=extra_weight,
                     extra_cycles_weighted=extra_weight*extra_raw)
        parts['total']+=parts['extra_cycles_weighted']
        cost=parts['total'];calls+=1
        if cost<best:
            best=cost
            f.write(x,'candidate.json',dict(loss=cost,actual_equilibrium_set_loss=eq_loss,cycle_weight=6.*f.POINCARE_SCALE,loss_components=parts,extra_cycle_inputs=list(NO_CYCLE_INPUTS),extra_cycle_counts=extra_counts,extra_cycle_diagnostics=extra_diag,fit_scope='geometry/equilibria/spectrum/events + actual equilibrium set matching + actual stable and unstable return roots, extrema and periods + no-cycle input penalties',closed_orbit_training=residual(x,True),calls=calls))
        if calls==1 or calls%100==0:
            print(f'CLOSED FM eval={calls} best={best:.7g} elapsed={time.monotonic()-start:.1f}s extra_counts={extra_counts} extra_diagnostics={extra_diag} components={parts}',flush=True)
        return cost
    o=minimize(objective,np.zeros(len(base)),method='Powell',bounds=bounds,
               options=dict(maxfev=int(os.environ.get('FM_CLOSED_MAXFEV','4500')),xtol=2e-3,ftol=3e-5))
    (f.P/'closed_optimizer.json').write_text(json.dumps(dict(nfev=o.nfev,fun=float(o.fun),best_visited=best,success=bool(o.success),message=o.message),indent=2))


if __name__=='__main__':run()
