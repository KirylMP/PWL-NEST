"""Controlled logit-coordinate experiment; unchanged spectral C thresholds.

Training only selects parameters; historical and fresh diagnostic protocols
are evaluated only after selection. Physical PWA artifacts remain untouched.
"""
import argparse,json,time
import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.special import expit,logit
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX
from pwl_nest.calcium_strips import CalciumStrips,spectral_seed,decode,NativeStatistics,NativeDynamic
from pwl_nest.logit_pwa import LogitPWA,to_latent,to_physical,transformed_lr_field,calcium_matched_seed
from pwl_nest.protocols import simulate
from fit_strict import protocols,stats

MODELS=ROOT/'models/logit_pwa';OUT=ROOT/'results/logit_pwa'


def run(case,budget=150):
    MODELS.mkdir(exist_ok=True,parents=True);OUT.mkdir(exist_ok=True,parents=True)
    physical=CalciumStrips.load(ROOT/f'models/calcium_strips/{case}.json')
    train=protocols(case)
    train.insert(0,dict(name='fixed_lower_rest',initial=[.6 if case=='AM' else 1.3,.8,.3 if case=='AM' else .48],
                       duration=1600.,tau=20.,baseline=.3 if case=='AM' else .48))
    refs={};points=[]
    for p in train:
        kwargs={k:v for k,v in p.items() if k!='name'}
        if not p['name'].startswith('fixed'):kwargs['dt']=.1
        r=simulate(case,**kwargs);refs[p['name']]=r
        points.append(r[1][::max(1,len(r[1])//400)])
    points=np.vstack(points);latent_points=to_latent(points)
    # A calcium-matched control: only the h readout changes at h0=.8.
    alpha=float(np.mean(points[:,1]*(1-points[:,1])))
    control=calcium_matched_seed(physical,alpha)
    control.save(MODELS/f'{case}_calcium_matched_control.json')
    _,ms,ks,source=spectral_seed(case,points)
    metadata={**source,'coordinates':['C','z','I'],'transform':'C,logit(h),I',
         'class_name':'continuous_PWA_in_logit_coordinates','enforce_types':True,
         'physical_field_PWA':False,'accepted':False,
         'target_field':'exact push-forward of LR: Cdot=fC; zdot=fh/[h(1-h)]',
         'initialization':'calcium matched control and exact transformed-field least squares'}
    theta=physical.thresholds.copy();n=physical.region_count
    def encode(model):
        mu,k,q,qb=model.spectral_coefficients()
        z=np.r_[np.log(abs(mu)),np.log(abs(k)),q[0]/.05,np.log(qb),
                 np.log(model.matrix[0,2]),np.log(-model.matrix[1,2]),model.matrix[0,3],model.matrix[0,0]]
        for j,label in enumerate(metadata['target_types']):
            if label.endswith('focus'):z[n+j]=np.log(max(1e-8,k[j]-mu[j]*mu[j]/4))
            elif label.endswith('node'):z[n+j]=logit(np.clip(4*k[j]/(mu[j]*mu[j]),1e-6,1-1e-6))
        return z
    z0=encode(control.latent)
    lower=np.r_[np.full(n,-9.),np.full(n,-12.),-4.,-12.,-7.,-4.,-8.,-8.]
    upper=np.r_[np.full(n,5.),np.full(n,5.),4.,2.,3.,1.,8.,8.]
    for j,label in enumerate(metadata['target_types']):
        if label.endswith('node'):upper[n+j]=12.
    z0=np.clip(z0,lower+1e-6,upper-1e-6)
    make=lambda z:decode(case,theta,z,ms,ks,metadata.copy())
    targets_field=transformed_lr_field(latent_points,case)
    def field_residual(z):
        error=(make(z).field(latent_points)-targets_field)/np.array([.2,.2])
        return np.r_[error.ravel()/np.sqrt(len(points)),.01*(z-z0)]
    print(f'{case}: exact transformed-field fit; {n} regions, {len(z0)} parameters',flush=True)
    field=least_squares(field_residual,z0,bounds=(lower,upper),max_nfev=150,
                          ftol=1e-8,xtol=1e-8,gtol=1e-8)
    LogitPWA(make(field.x)).save(MODELS/f'{case}_transformed_field_seed.json')
    fixed=[p for p in train if p['name'].startswith('fixed')]
    decays=[p for p in train if p['name'].startswith('decay')]
    stats_target={p['name']:stats(*refs[p['name']][:2]) for p in fixed}
    source_h={p['name']:[float(refs[p['name']][1][:,1].min()),float(refs[p['name']][1][:,1].max())] for p in fixed}
    sampled={}
    for p in decays:
        t,y,_=refs[p['name']];times=np.linspace(0,p['duration'],361)
        sampled[p['name']]=np.column_stack([np.interp(times,t,y[:,d]) for d in [0,1]])
    native=NativeStatistics();dynamic=NativeDynamic()
    anchors=np.array([e['I'] for e in source['events']])
    best=np.inf;bestz=z0.copy();evaluations=0;history=[];started=time.monotonic()
    def residual(z):
        nonlocal best,bestz,evaluations
        evaluations+=1;model=make(z);rr=[]
        for p in fixed:
            target=stats_target[p['name']]
            ob=native.run(model,p['baseline'],to_latent(p['initial']),p['duration'],diagnostics=True)
            if ob is None:rr.extend([100.]*7);continue
            obs,diag=ob
            mn,mx,period,ratio,minall,stopped=obs
            rr.extend([(mn-target['Cmin'])/.02,(mx-target['Cmax'])/.05,
                       (period-target['period'])/(1 if case=='AM' else 3) if target['period'] is not None and np.isfinite(period)
                       else (15. if target['period'] is not None else (mx-mn)/.02),
                       min(0,minall)/.005,max(0,diag['max_C']-C_MAX)/.02,
                       (expit(diag['min_h'])-source_h[p['name']][0])/.1,
                       (expit(diag['max_h'])-source_h[p['name']][1])/.1])
        for p in decays:
            y=dynamic.run(model,to_latent(p['initial']),p['duration'],p['tau'],p['baseline'])
            if y is None:rr.extend(np.full(724,100.));continue
            x=np.column_stack([y[:,0],expit(y[:,1])])
            rr.extend(((x-sampled[p['name']])/np.array([.03,.05])/np.sqrt(361)).ravel())
            rr.extend([min(0,x[:,0].min())/.005,max(0,x[:,0].max()-C_MAX)/.02])
        mu,k,q,qb=model.spectral_coefficients()
        transitions=(k[:-1]*theta-q[:-1])/qb
        rr.extend((transitions-anchors)/.03)
        rr.extend(.005*(z-z0))
        rr=np.array(rr);loss=float(rr@rr)
        if loss<best:
            best,bestz=loss,z.copy()
            if not history or evaluations-history[-1]['evaluation']>=40:
                history.append(dict(evaluation=evaluations,loss=loss,seconds=time.monotonic()-started))
                print(f'{case}: evaluation={evaluations} loss={loss:.6g}',flush=True)
        return rr
    initial=[];runs=[]
    for label,z in [('calcium_matched',z0),('field',field.x)]:
        r=residual(z);initial.append(dict(label=label,loss=float(r@r)))
    for label,z in [('calcium_matched',z0),('field',field.x)]:
        fit=least_squares(residual,z,bounds=(lower,upper),max_nfev=budget,diff_step=1e-4,
                           x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7)
        residual(fit.x)
        runs.append(dict(start=label,method='least_squares',status=int(fit.status),nfev=int(fit.nfev),message=fit.message))
    fit=minimize(lambda z:float(np.dot(r:=residual(z),r)),bestz,method='Powell',
                 bounds=list(zip(lower,upper)),options=dict(maxfev=3000,xtol=1e-5,ftol=1e-6))
    residual(fit.x);runs.append(dict(method='Powell',status=int(fit.status),nfev=int(fit.nfev),message=str(fit.message)))
    model=LogitPWA(make(bestz))
    model.metadata.update(parameters=bestz.tolist(),mu_sign=ms.tolist(),k_sign=ks.tolist(),training_loss=best,
                         selected_by='training only; historical and fresh diagnostics excluded')
    model.save(MODELS/f'{case}.json')
    (OUT/f'{case}_fit.json').write_text(json.dumps(dict(initial_scores=initial,best_loss=best,history=history,
        residual_evaluations=evaluations,runs=runs,training_protocols=train,bounds=[lower.tolist(),upper.tolist()],
        field_seed_loss=float(field.fun@field.fun)),indent=2,allow_nan=False)+'\n')
    from validate_logit import validate
    validate(case,model,control,train,refs)
    print(f'{case}: complete; training loss={best:.6g}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['AM','FM','both'],default='both');parser.add_argument('--budget',type=int,default=150)
    args=parser.parse_args()
    for case in (['AM','FM'] if args.case=='both' else [args.case]):run(case,args.budget)
