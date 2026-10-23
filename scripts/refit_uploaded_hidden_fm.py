"""Focused refit of the uploaded FM hidden-state coefficients against LR.

Training uses 3 transition protocols and 4 constant-IP3 traces; held-out
transition protocols are scored only after fitting. See the sidecar README.
"""
import json,time
import numpy as np
from scipy.optimize import least_squares
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.models import ROOT
from validate_uploaded_hidden_fm import CACHE,OUT,simulate,peaks

BASE=OUT/'input_archive/four_region_base.json'
START=np.array([-.27534184808371753,-.06788525566941761,2.31745979055761,.16130211084202645,.09749471242739638])
FIT_NAMES={'pulse_tau_20.0','periodic','promoted_tau_65.0'}

def cycle_period(t,x):
    mask=t>=max(200.,.55*t[-1]);p=peaks(t[mask],x[mask,0])
    return np.diff(p) if len(p)>=4 else np.array([])

def main():
    model=CalciumTransformPWA.load(BASE);data=json.loads((CACHE/'data.json').read_text());protocols=[]
    for q in data['train']:
        if q['name'] not in FIT_NAMES:continue
        with np.load(CACHE/f'{q["name"]}.npz') as f:protocols.append((q,f['t'].copy(),f['x'].copy()))
    for ip3 in [.53,.7,1.12,1.13]:
        with np.load(CACHE/f'fixed_{ip3:g}.npz') as f:
            q=dict(name=f'fixed_{ip3:g}',duration=400.,tau=20.,baseline=ip3,jumps=[])
            protocols.append((q,f['t'].copy(),f['x'].copy()))
    def residual(p):
        pieces=[]
        for q,tr,truth in protocols:
            initial=[1.3,.8,q['baseline']] if q['name'].startswith('fixed_') else None
            t,x=simulate(q,model,'hidden',initial,parameters=p)
            target=np.column_stack([np.interp(t,tr,truth[:,j]) for j in range(3)])
            idx=np.arange(0,len(t),5)
            pieces.extend(((x[idx,0]-target[idx,0])/(.08*np.sqrt(len(idx)))).tolist())
            pieces.extend(((x[idx,1]-target[idx,1])/(.12*np.sqrt(len(idx)))).tolist())
            cp,rp=cycle_period(t,x),cycle_period(t,target);n=min(len(cp),len(rp))
            pieces.append(2.5*(len(cp)-len(rp)))
            if n:pieces.extend(((cp[-n:]-rp[-n:])/(1.5*np.sqrt(n))).tolist())
            if q['baseline']>=1.13:pieces.append(8.*max(0.,float(np.ptp(x[t>=t[-1]-100,0]))-.02))
        return np.asarray(pieces)
    t0=time.time();r0=residual(START)
    fit=least_squares(residual,START,bounds=([-1.5,-1.5,.05,-.8,-.8],[1.5,1.5,5.,.8,.8]),
        max_nfev=5,jac='2-point',x_scale=[.3,.3,1.,.2,.2],diff_step=1e-3,
        ftol=1e-6,xtol=1e-9,gtol=1e-6)
    result=dict(start_parameters=START.tolist(),fitted_parameters=fit.x.tolist(),
        start_loss=float(r0@r0),fit_loss=float(fit.fun@fit.fun),nfev=int(fit.nfev),
        elapsed_s=time.time()-t0,heldout={},constant={},
        note='This reproducible focused refit is exploratory; the preceding run found a 1.15% loss reduction, while a second pass with stronger period weighting did not improve further.')
    for q in data['test']:
        with np.load(CACHE/f'{q["name"]}.npz') as f:tr,truth=f['t'].copy(),f['x'].copy()
        t,x=simulate(q,model,'hidden',parameters=fit.x);target=np.column_stack([np.interp(t,tr,truth[:,j]) for j in range(3)])
        result['heldout'][q['name']]={'C_RMSE':float(np.sqrt(np.mean((x[:,0]-target[:,0])**2))),
            'h_RMSE':float(np.sqrt(np.mean((x[:,1]-target[:,1])**2)))}
    for q,tr,truth in protocols:
        if not q['name'].startswith('fixed_'):continue
        t,x=simulate(q,model,'hidden',[1.3,.8,q['baseline']],parameters=fit.x)
        target=np.column_stack([np.interp(t,tr,truth[:,j]) for j in range(3)])
        cp,rp=cycle_period(t,x),cycle_period(t,target)
        result['constant'][str(q['baseline'])]={'model_period_s':float(np.median(cp)) if len(cp) else None,
            'LR_period_s':float(np.median(rp)) if len(rp) else None}
    (OUT/'refit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
