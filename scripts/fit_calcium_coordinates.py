"""Compare sqrt/cbrt/log(C), retaining logit(h), source C thresholds and counts."""
import argparse,json,time
import numpy as np
from scipy.optimize import brentq,lsq_linear,least_squares,minimize
from scipy.special import expit,logit
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX,lr_field,equilibrium_h
from pwl_nest.calcium_strips import CalciumStrips,spectral_seed,NativeStatistics
from pwl_nest.logit_pwa import LogitPWA
from pwl_nest.calcium_coordinates import (CalciumCoordinate,CalciumTransformPWA,decode_coordinates,NativeCoordinateDynamic)
from pwl_nest.protocols import simulate
from fit_strict import protocols,stats

KINDS=['sqrt','cbrt','log']
ALL_KINDS=['identity']+KINDS
MODELS=ROOT/'models/calcium_coordinates';OUT=ROOT/'results/calcium_coordinates'


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def training_data(case):
    training=protocols(case)
    training.insert(0,dict(name='fixed_lower_rest',initial=[.6 if case=='AM' else 1.3,.8,.3 if case=='AM' else .48],
                          duration=1600.,tau=20.,baseline=.3 if case=='AM' else .48))
    refs={};points=[]
    for p in training:
        kwargs={k:v for k,v in p.items() if k!='name'}
        kwargs['dt']=.2 if p['name'].startswith('fixed') else .1
        refs[p['name']]=simulate(case,**kwargs)
        y=refs[p['name']][1];points.append(y[::max(1,len(y)//400)])
    points=np.vstack(points)
    source=spectral_seed(case,points)[3]
    cref=brentq(lambda c:lr_field([c,equilibrium_h(c,0.),0.],case)[0],1e-10,C_MAX)
    ctop=brentq(lambda c:lr_field([c,equilibrium_h(c,1.3),1.3],case)[0],1e-10,C_MAX)
    return training,refs,points,source,cref,ctop


def run_kind(case,kind,data,budget,*,normalize_span=False,models_root=MODELS,results_root=OUT):
    training,refs,points,source,cref,ctop=data
    directory=models_root/kind;destination=results_root/kind
    directory.mkdir(parents=True,exist_ok=True);destination.mkdir(parents=True,exist_ok=True)
    physical_theta=np.array([e['C'] for e in source['events']]);n=len(physical_theta)+1
    scale=float(CalciumCoordinate(kind,cref).forward(C_MAX)/C_MAX) if normalize_span else 1.
    co=CalciumCoordinate(kind,cref,scale);theta=co.forward(physical_theta);href=float(equilibrium_h(cref,0.));zref=float(logit(href))
    meta={**source,'class_name':'PWA_in_transformed_calcium_and_logit_h','coordinates':['u','v','I'],
          'calcium_transform':kind,'physical_thresholds_C':physical_theta.tolist(),
          'physical_field_PWA':False,'enforce_types':True,'accepted':False,
          'source_chain_rule':'udot=phi_prime(C)*fC; vdot=fh/[h*(1-h)]',
          'calcium_reference':cref,'h_reference':href,'coordinate_scale':scale,
          'normalization':'phi(Cref)=0; phi(C_MAX)=C_MAX' if normalize_span else 'phi(Cref)=0; phi_prime(Cref)=1',
          'q0_parameterization':'q0=qb*IP3_offset; source rest u=0',
          'parameter_count':2*n+6}
    dummy=CalciumTransformPWA(CalciumStrips(case,theta,np.zeros((2,n+3)),meta.copy()),co,href)
    transformed=dummy.to_latent(points);truth=dummy.source_field(transformed)
    # Monotone inverse-equilibrium chords in the transformed calcium coordinate.
    cs=np.r_[cref,physical_theta,ctop];inputs=np.r_[0.,[e['I'] for e in source['events']],1.3]
    slopes=np.diff(inputs)/np.diff(co.forward(cs));weights=abs(np.diff(inputs))
    qb=max(1e-6,float(np.dot(weights*slopes,source['source_average_determinant'])/np.dot(weights,slopes*slopes)))
    mu=np.array(source['source_average_trace']);k=qb*slopes;ms=np.sign(mu);ks=np.sign(k)
    def encode_parts(mu,k,q0,qb,B,D,cI,c0):
        z=np.r_[np.log(abs(mu)),np.log(abs(k)),q0/qb,np.log(qb),np.log(B),np.log(-D),cI,c0]
        for j,label in enumerate(source['target_types']):
            if label.endswith('focus'):z[n+j]=np.log(max(1e-7,k[j]-mu[j]*mu[j]/4))
            elif label.endswith('node'):z[n+j]=logit(np.clip(4*k[j]/(mu[j]*mu[j]),1e-5,1-1e-5))
        return z
    u,v=transformed[:,0],transformed[:,1]
    g=mu[0]*u+np.maximum(0,u[:,None]-theta)@np.diff(mu)
    embedding=lsq_linear(np.column_stack([u,v,points[:,2],np.ones(len(points))]),truth[:,0]-g,
                         bounds=([.01,.001,-8.,-8.],[3.,20.,8.,8.]),tol=1e-12)
    minusD,B,cI,c0=embedding.x
    spectral=encode_parts(mu,k,0.,qb,B,-minusD,cI,c0)
    previous=LogitPWA.load(ROOT/f'models/logit_pwa/{case}.json').latent
    matrix=previous.matrix.copy()
    matrix[:,0]+=matrix[:,1]*cref+matrix[:,2]*zref
    matrix[:,1]*=scale;matrix[:,4:]*=scale;matrix[0]/=scale
    warm=CalciumStrips(case,theta,matrix,meta.copy())
    wm,wk,wq,wqb=warm.spectral_coefficients()
    tangent=encode_parts(wm,wk,wq[0],wqb,matrix[0,2],matrix[1,2],matrix[0,3],matrix[0,0])
    lower=np.r_[np.full(n,-9.),np.full(n,-12.),-.5,-12.,-8.,-4.,-8.,-8.]
    upper=np.r_[np.full(n,5.),np.full(n,5.),.5,2.,3.,1.5,8.,8.]
    for j,label in enumerate(source['target_types']):
        if label.endswith('node'):upper[n+j]=12.
    starts=[np.clip(z,lower+1e-6,upper-1e-6) for z in [spectral,tangent]]
    make=lambda z:decode_coordinates(case,theta,z,ms,ks,meta,co,href)
    for label,z in zip(['spectral_seed','tangent_seed'],starts):make(z).save(directory/f'{case}_{label}.json')
    inverse_derivatives=np.column_stack([1/co.derivative(points[:,0]),points[:,1]*(1-points[:,1])])
    def field_residual(z):
        physical_error=(make(z).latent.field(transformed)-truth)*inverse_derivatives
        return np.r_[(physical_error/np.array([.2,.05])).ravel()/np.sqrt(len(points)),.01*(z-starts[0])]
    print(f'{case} {kind}: field seed, {n} regions/{2*n+6} parameters',flush=True)
    field=least_squares(field_residual,starts[0],bounds=(lower,upper),max_nfev=120,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    make(field.x).save(directory/f'{case}_field_seed.json')
    static=NativeStatistics();dynamic=NativeCoordinateDynamic()
    fixed=[p for p in training if p['name'].startswith('fixed')];decays=[p for p in training if p['name'].startswith('decay')]
    targets={p['name']:stats(*refs[p['name']][:2]) for p in fixed}
    htargets={p['name']:[refs[p['name']][1][:,1].min(),refs[p['name']][1][:,1].max()] for p in fixed}
    sampled={}
    for p in decays:
        t,x,_=refs[p['name']];grid=np.linspace(0,p['duration'],361)
        sampled[p['name']]=np.column_stack([np.interp(grid,t,x[:,d]) for d in [0,1]])
    anchors=np.array([e['I'] for e in source['events']]);upper_guard=float(co.forward(20.))
    count=0;best=np.inf;bestz=starts[0].copy();history=[];started=time.monotonic()
    def residual(z,regularize=True):
        nonlocal count,best,bestz
        count+=1;model=make(z);rr=[]
        for p in fixed:
            obs=static.run(model.latent,p['baseline'],model.to_latent(p['initial']),p['duration'],diagnostics=True)
            if obs is None:rr.extend([100.]*7);continue
            ob,diag=obs;mn,mx,period,ratio,minall,stopped=ob
            if minall<=co.lower or diag['max_C']>=upper_guard:
                gap=max(0.,co.lower-minall,diag['max_C']-upper_guard)
                rr.extend([50.+1000*gap]*7);continue
            try:cmin,cmax,cminall,cmaxall=map(float,co.inverse(np.array([mn,mx,minall,diag['max_C']])))
            except (ValueError,FloatingPointError):rr.extend([100.]*7);continue
            target=targets[p['name']]
            rr.extend([(cmin-target['Cmin'])/.02,(cmax-target['Cmax'])/.05,
                (period-target['period'])/(1 if case=='AM' else 3) if target['period'] is not None and np.isfinite(period)
                else (15. if target['period'] is not None else (cmax-cmin)/.02),
                min(0,cminall)/.005,max(0,cmaxall-C_MAX)/.02,
                (expit(diag['min_h']+zref)-htargets[p['name']][0])/.1,
                (expit(diag['max_h']+zref)-htargets[p['name']][1])/.1])
        for p in decays:
            y=dynamic.run(model,p['initial'],p['duration'],p['tau'],p['baseline'])
            if y is None:rr.extend(np.full(724,100.));continue
            try:x=np.column_stack([co.inverse(y[:,0]),expit(y[:,1]+zref)])
            except (ValueError,FloatingPointError):rr.extend(np.full(724,100.));continue
            rr.extend(((x-sampled[p['name']])/np.array([.03,.05])/np.sqrt(361)).ravel())
            rr.extend([min(0,x[:,0].min())/.005,max(0,x[:,0].max()-C_MAX)/.02])
        mu,k,q,qb=model.latent.spectral_coefficients()
        rr.extend(((k[:-1]*theta-q[:-1])/qb-anchors)/.03)
        rr=np.array(rr);data_loss=float(rr@rr)
        reg=.005*(z-starts[0]);r=np.r_[rr,reg] if regularize else rr
        loss=float(r@r)
        if regularize and loss<best:
            best,bestz=loss,z.copy()
            if not history or count-history[-1]['evaluation']>=50:
                history.append(dict(evaluation=count,total_loss=loss,data_loss=data_loss,seconds=time.monotonic()-started))
                print(f'{case} {kind}: eval={count}, data_loss={data_loss:.6g}',flush=True)
        return r
    candidates=[('spectral',starts[0]),('tangent',starts[1]),('field',field.x)]
    scores=[]
    for label,z in candidates:
        r=residual(z);scores.append(dict(label=label,total_loss=float(r@r)))
    # Same predeclared budget: refine the two best INITIAL training seeds.
    ranked=sorted(zip(candidates,scores),key=lambda item:item[1]['total_loss'])[:2]
    runs=[]
    for (label,z),score in ranked:
        fit=least_squares(residual,z,bounds=(lower,upper),max_nfev=budget,diff_step=1e-4,
                           x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7)
        residual(fit.x);runs.append(dict(start=label,method='least_squares',status=int(fit.status),nfev=int(fit.nfev),message=fit.message))
    fit=minimize(lambda z:float(np.dot(r:=residual(z),r)),bestz,method='Powell',bounds=list(zip(lower,upper)),
                  options=dict(maxfev=2400,xtol=1e-5,ftol=1e-6))
    residual(fit.x);runs.append(dict(method='Powell',status=int(fit.status),nfev=int(fit.nfev),message=str(fit.message)))
    model=make(bestz);r=residual(bestz,regularize=False);data_loss=float(r@r)
    model.metadata.update(parameters=bestz.tolist(),mu_sign=ms.tolist(),k_sign=ks.tolist(),training_data_loss=data_loss,
                           training_total_loss=best,accepted=False,selected_by='common physical training residual; diagnostics excluded')
    model.save(directory/f'{case}.json')
    dump(destination/f'{case}_fit.json',dict(initial_scores=scores,training_data_loss=data_loss,training_total_loss=best,
        bounds=[lower.tolist(),upper.tolist()],history=history,optimizer_runs=runs,evaluations=count,training_protocols=training,
        latent_thresholds=theta.tolist(),reference_calcium=cref,reference_h=href))
    return model,data_loss


def run(case,budget=120):
    data=training_data(case);models=[]
    for kind in ALL_KINDS:models.append(run_kind(case,kind,data,budget))
    selected,score=min((pair for pair in models if pair[0].coordinate.kind!='identity'),key=lambda pair:pair[1])
    selected.save(MODELS/f'selected_{case}.json')
    dump(OUT/f'{case}_selection.json',dict(selected=selected.coordinate.kind,training_scores={m.coordinate.kind:s for m,s in models},
        selection='best of three nonlinear C transforms on training data only; identity is a procedure-matched control'))
    from validate_calcium_coordinates import validate
    validate(case,[m for m,s in models],data[0],data[1])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['AM','FM','both'],default='both');parser.add_argument('--budget',type=int,default=120)
    args=parser.parse_args()
    MODELS.mkdir(exist_ok=True,parents=True);OUT.mkdir(exist_ok=True,parents=True)
    for case in (['AM','FM'] if args.case=='both' else [args.case]):run(case,args.budget)
