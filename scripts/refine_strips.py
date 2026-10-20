"""Second, staged attempt, unchanged thresholds and region count.
Hard preservation of source equilibrium interval spectral subtypes.
"""
import argparse,json,time
import numpy as np
from scipy.optimize import least_squares,minimize,lsq_linear
from scipy.special import logit
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX,lr_field
from pwl_nest.calcium_strips import CalciumStrips,spectral_seed,decode,NativeStatistics,NativeDynamic
from pwl_nest.protocols import simulate
from fit_strict import protocols,stats

OUT=ROOT/'results/calcium_strips'
MODELS=ROOT/'models/calcium_strips'


def run(case):
    old=CalciumStrips.load(MODELS/f'joint_v1/{case}.json')
    training=protocols(case)
    training.insert(0,dict(name='fixed_lower_rest',initial=[.6 if case=='AM' else 1.3,.8,.3 if case=='AM' else .48],
                          duration=1600.,tau=20.,baseline=.3 if case=='AM' else .48))
    refs={};points=[]
    for p in training:
        r=simulate(case,**{k:v for k,v in p.items() if k!='name'})
        refs[p['name']]=r
        points.append(r[1][::max(1,len(r[1])//300)])
    points=np.vstack(points)
    seed,ms,ks,meta=spectral_seed(case,points)
    theta=np.array([e['C'] for e in meta['events']]);n=len(theta)+1
    meta.update(enforce_types=True,shape_parameterization='log traces; positive focus gap or node determinant/trace-square ratio',
                attempt='autonomous_then_joint_v2')
    def encode(model):
        mu,k,q,qb=model.spectral_coefficients()
        z=np.r_[np.log(abs(mu)),np.log(abs(k)),q[0]/.05,np.log(qb),
                np.log(model.matrix[0,2]),np.log(-model.matrix[1,2]),model.matrix[0,3],model.matrix[0,0]]
        for j,label in enumerate(meta['target_types']):
            if label.endswith('focus'):z[n+j]=np.log(max(1e-5,k[j]-mu[j]*mu[j]/4))
            elif label.endswith('node'):z[n+j]=logit(np.clip(4*k[j]/(mu[j]*mu[j]),.01,.99))
        return z
    seedmodel=CalciumStrips.load(MODELS/f'joint_v1/{case}_spectral_seed.json')
    starts=[encode(seedmodel),encode(old)]
    lower=np.r_[np.full(n,-9.),np.full(n,-12.),-4.,-12.,-5.,-4.,-5.,-5.]
    upper=np.r_[np.full(n,5.),np.full(n,5.),4.,2.,4.,1.,5.,5.]
    for j,label in enumerate(meta['target_types']):
        if label.endswith('node'):upper[n+j]=12.
    starts=[np.clip(z,lower+1e-6,upper-1e-6) for z in starts]
    make=lambda z:decode(case,theta,z,ms,ks,meta.copy())
    native=NativeStatistics();dynamic=NativeDynamic()
    fixed=[p for p in training if p['name'].startswith('fixed')]
    decays=[p for p in training if p['name'].startswith('decay')]
    targets={p['name']:stats(*refs[p['name']][:2]) for p in fixed}
    anchorI=np.array([e['I'] for e in meta['events']])
    evals=0;best=np.inf;bestz=starts[0];history=[];starttime=time.monotonic()
    def static_residual(z,physical=False):
        model=make(z);rr=[]
        for p in fixed:
            target=targets[p['name']]
            if physical:
                ob=native.run(model,p['baseline'],p['initial'],p['duration'],diagnostics=True)
                obs,diag=(None,None) if ob is None else ob
            else:obs=native.run(model,p['baseline'],p['initial'],p['duration'],v0_override=0.)
            if obs is None:
                rr.extend([100.]* (7 if physical else 4));continue
            mn,mx,period,ratio,minall,stopped=obs
            rr.extend([(mn-target['Cmin'])/.02,(mx-target['Cmax'])/.05,
                        (period-target['period'])/(1 if case=='AM' else 3) if target['period'] is not None and np.isfinite(period)
                        else (15. if target['period'] is not None else (mx-mn)/.02),min(0,minall)/.005])
            if physical:
                rr.extend([min(0,diag['min_h'])/.01,max(0,diag['max_h']-1)/.01,max(0,diag['max_C']-C_MAX)/.02])
        mu,k,q,qb=model.spectral_coefficients()
        transitions=(k[:-1]*theta-q[:-1])/qb
        rr.extend((transitions-anchorI)/.03)
        return rr
    # Stage A: cycle shape and equilibrium anchor inputs, independent of physical embedding.
    shape_best=np.inf;shape_z=starts[0].copy()
    for index,z in enumerate(starts):
        count=0
        def shape_residual(shape):
            nonlocal count,shape_best,shape_z
            count+=1
            full=z.copy();full[:2*n+2]=shape
            rr=np.r_[static_residual(full),.005*(shape-starts[0][:2*n+2])]
            value=float(rr@rr)
            if value<shape_best:
                shape_best,shape_z=value,full.copy()
            return rr
        print(f'{case}: shape start {index}',flush=True)
        fit=least_squares(shape_residual,z[:2*n+2],bounds=(lower[:2*n+2],upper[:2*n+2]),
                           diff_step=1e-4,max_nfev=180,x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7)
        minimize(lambda v:float(np.dot(r:=shape_residual(v),r)),shape_z[:2*n+2],method='Powell',
                 bounds=list(zip(lower[:2*n+2],upper[:2*n+2])),options=dict(maxfev=3500,xtol=1e-5,ftol=1e-6))
        print(f'{case}: best shape loss {shape_best:.6g}',flush=True)
    # Stage B: source physical embedding, then simultaneous physical/dynamic refinement.
    model=make(shape_z);mu=model.spectral_coefficients()[0]
    g=mu[0]*points[:,0]+np.maximum(0,points[:,0,None]-theta)@np.diff(mu)
    design=np.column_stack([points[:,0],points[:,1],points[:,2],np.ones(len(points))])
    emb=lsq_linear(design,lr_field(points,case)[:,0]-g,bounds=([.02,.01,-5,-5],[2.,20,5,5]),tol=1e-12)
    minusD,B,cI,c0=emb.x
    shape_z[2*n+2:]=[np.log(B),np.log(minusD),cI,c0]
    shapefitted=make(shape_z);shapefitted.save(MODELS/f'{case}_autonomous_seed.json')
    samples={}
    for p in decays:
        t,y,_=refs[p['name']];times=np.linspace(0,p['duration'],361)
        samples[p['name']]=np.column_stack([np.interp(times,t,y[:,d]) for d in [0,1]])
    def joint_residual(z):
        nonlocal evals,best,bestz
        evals+=1;rr=static_residual(z,physical=True);model=make(z)
        for p in decays:
            y=dynamic.run(model,p['initial'],p['duration'],p['tau'],p['baseline'])
            if y is None:rr.extend(np.full(726,100.));continue
            rr.extend(((y-samples[p['name']])/np.array([.03,.05])/np.sqrt(361)).ravel())
            rr.extend([min(0,y[:,0].min())/.005,max(0,y[:,0].max()-C_MAX)/.02,
                       min(0,y[:,1].min())/.01,max(0,y[:,1].max()-1)/.01])
        rr.extend(.005*(z-shape_z))
        rr=np.array(rr);value=float(rr@rr)
        if value<best:
            best,bestz=value,z.copy()
            if len(history)==0 or evals-history[-1]['evaluation']>50:
                history.append(dict(evaluation=evals,loss=value,seconds=time.monotonic()-starttime))
                print(f'{case}: joint eval={evals} loss={value:.6g}',flush=True)
        return rr
    for label,z in [('autonomous',shape_z),('previous',starts[1])]:
        print(f'{case}: physical joint start {label}',flush=True)
        least_squares(joint_residual,np.clip(z,lower+1e-6,upper-1e-6),bounds=(lower,upper),
                       diff_step=1e-4,max_nfev=150,x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7)
    minimize(lambda z:float(np.dot(r:=joint_residual(z),r)),bestz,method='Powell',
             bounds=list(zip(lower,upper)),options=dict(maxfev=3000,xtol=1e-5,ftol=1e-6))
    model=make(bestz)
    model.metadata.update(training_loss=best,accepted=False,selected_by='v2 training loss only; no v2 holdout selection',
                           optimizer_scope='two local autonomous starts, then two joint starts and bounded Powell',
                           parameters=bestz.tolist(),mu_sign=ms.tolist(),k_sign=ks.tolist())
    model.save(MODELS/f'{case}.json')
    (OUT/f'{case}_refinement.json').write_text(json.dumps(dict(shape_loss=shape_best,joint_loss=best,history=history,
        joint_evaluations=evals,training_protocols=training,metadata=model.metadata),indent=2)+'\n')
    print(f'{case}: fitted; validating',flush=True)
    # Reuse independent validation implementation through an explicit shared helper.
    from validate_strips import validate
    validate(case,model,training,refs)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['AM','FM','both'],default='both');args=parser.parse_args()
    for case in (['AM','FM'] if args.case=='both' else [args.case]):run(case)
