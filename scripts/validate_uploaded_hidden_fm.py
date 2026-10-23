"""Replay the uploaded four-region FM + hidden-r proposal against Li-Rinzel.

This is an isolated experiment: it loads the supplied base from the sibling
results directory and does not overwrite newer published FM candidates.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.strict_pwa import lr_field
from pwl_nest.models import ROOT

OUT=ROOT/'results/reapplied_hidden_fm_experiments_20261022'
BASE=OUT/'input_archive/four_region_base.json'
CACHE=ROOT/'results/fm_phase_continuation/cache/FM'
PARAMS=np.array([-.27534184808371753,-.06788525566941761,2.31745979055761,.16130211084202645,.09749471242739638])
DT=.2

def peaks(t,c):
    keep=t>=max(0.,t[-1]*.55)
    return t[keep][find_peaks(c[keep],prominence=.02,distance=5)[0]]

def simulate(q,model,kind,initial=None,parameters=PARAMS,hinge=None):
    data=json.loads((CACHE/'data.json').read_text())
    physical=np.asarray(data['rest'] if initial is None else initial,float)
    if kind=='LR':state=physical.copy()
    else:
        z=model.to_latent(physical)
        state=z.copy() if kind=='base' else np.array([z[0],z[1],z[0],z[2]])
    duration=float(q['duration']);t=np.arange(round(duration/DT)+1)*DT
    out=np.empty((len(t),len(state)));out[0]=state;last=0.
    a,b,alpha,d,e=parameters
    def rhs(_time,y):
        if kind=='LR':return np.r_[lr_field(y,'FM'),-(y[2]-q['baseline'])/q['tau']]
        if kind=='base':return np.r_[model.field(y),-(y[2]-q['baseline'])/q['tau']]
        u,v,r,I=y;f=model.field([u,v,I]);du=f[0]+a*(r-u);dv=f[1]+b*(r-u)
        dr=alpha*(u-r)+d*v+e*(I-1.12)
        if hinge:
            s={'C':u,'I':I,'r':r}[hinge['coordinate']]
            h=max(0.,s-hinge['knot']);du+=hinge['weights'][0]*h;dv+=hinge['weights'][1]*h;dr+=hinge['weights'][2]*h
        return [du,dv,dr,-(I-q['baseline'])/q['tau']]
    for end,delta in [*q.get('jumps',[]),(duration,0.)]:
        if end>last+1e-12:
            sol=solve_ivp(rhs,(last,end),state,method='DOP853',rtol=2e-8,atol=2e-10,max_step=.3,dense_output=True)
            if not sol.success:raise RuntimeError(sol.message)
            mask=(t>last+1e-10)&(t<=end+1e-10);out[mask]=sol.sol(t[mask]).T
            state=sol.y[:,-1];last=end
        state[-1]+=delta;out[np.isclose(t,end,atol=1e-10,rtol=0)]=state
    x=out if kind=='LR' else model.to_physical(out[:,[0,1,2] if kind=='base' else [0,1,3]])
    return t,x

def scores(t,x,tr,truth):
    ref=np.column_stack([np.interp(t,tr,truth[:,k]) for k in range(3)])
    p=peaks(t,x[:,0]);rp=peaks(t,ref[:,0])
    return dict(C_RMSE=float(np.sqrt(np.mean((x[:,0]-ref[:,0])**2))),
        h_RMSE=float(np.sqrt(np.mean((x[:,1]-ref[:,1])**2))),
        fit_events=len(p),LR_events=len(rp),
        fit_period_s=float(np.median(np.diff(p))) if len(p)>=4 else None,
        LR_period_s=float(np.median(np.diff(rp))) if len(rp)>=4 else None)

def main():
    model=CalciumTransformPWA.load(BASE);data=json.loads((CACHE/'data.json').read_text())
    result=dict(model=str(BASE.relative_to(ROOT)),thresholds_u=model.latent.thresholds.tolist(),
        thresholds_C=model.metadata['physical_thresholds_C'],regions=model.latent.region_count,
        parameters=PARAMS.tolist(),dynamic={},constant={},
        note='Uses uploaded four-region base and archived hidden-state parameters. The archive compared against an eight-region PWA, not Li-Rinzel; this replay compares both against Li-Rinzel. SIC and NEST metrics are unavailable in this current draft and are not inferred.')
    for q in data['train']+data['test']:
        with np.load(CACHE/f'{q["name"]}.npz') as f:tr,truth=f['t'].copy(),f['x'].copy()
        row={}
        for kind in ('base','hidden'):
            t,x=simulate(q,model,kind);row[kind]=scores(t,x,tr,truth)
        result['dynamic'][q['name']]=row
    for ip3 in [.53,.55,.7,1.,1.12,1.125,1.13]:
        q=dict(duration=700.,tau=20.,baseline=ip3,jumps=[]);initial=[1.3,.8,ip3]
        t,ref=simulate(q,model,'LR',initial)
        result['constant'][str(ip3)]={}
        for kind in ('base','hidden'):
            tx,x=simulate(q,model,kind,initial);m=scores(tx,x,t,ref)
            late=tx>=tx[-1]-50
            m['tail_C_mean']=float(x[late,0].mean());result['constant'][str(ip3)][kind]=m
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,row in result['dynamic'].items():
        print(name,'base',row['base']['C_RMSE'],'hidden',row['hidden']['C_RMSE'],'events',row['hidden']['fit_events'],row['hidden']['LR_events'])
    print('constant',json.dumps(result['constant'],indent=2))

if __name__=='__main__':main()
