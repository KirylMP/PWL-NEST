"""FM continuation with an exact LR rest and explicit peak-shape gates.

Fits are selected on independently verified training constraints before any new
held-out readout. Legacy candidates/results remain frozen.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution,least_squares,minimize

from pwl_nest.calcium_coordinates import CalciumTransformPWA,decode_coordinates,simulate_coordinates
from pwl_nest.equilibrium_constraints import anchor_equilibrium
from pwl_nest.peak_shape import peak_shape_check,peak_shape_residual,autonomous_shape_check
from pwl_nest.protocols import simulate
from pwl_nest.rest_protocols import stable_rest
from pwl_nest.thresholds import OrderedThresholds
from fit_mixed_coordinates import MixedObjective,FAMILIES,ERRORS,SOURCE
from fit_fm_input_threshold import training_checks
from fit_strict import stats
from overnight_coordinate_grid import atomic,code_hash,source_rest
from refine_server_candidates import aggregate,diagnostics,rebuild_training_field

HARD_SHAPE_TOLERANCE=.10
FIT_SHAPE_TOLERANCE=.08


class ShapeObjective(MixedObjective):
    def __call__(self,model,precise=False,parts=False):
        residual,groups=super().__call__(model,precise=precise,parts=True)
        for p in self.spec['train']:
            tr,truth,_=self.refs[p['name']]
            if self.last_train_paths is None:
                extra=np.full(len(peak_shape_residual(tr,truth,tr,truth)),100.)
            else:
                t,x,_=self.last_train_paths[p['name']]
                extra=peak_shape_residual(t,x,tr,truth,tolerance=FIT_SHAPE_TOLERANCE)
            groups[p['name']+'_shape']=float(extra@extra);residual=np.r_[residual,extra]
        regimes=[]
        for p in self.spec['fixed']:
            reference=self.static.targets[p['name']]['observed_status']
            trace=self.last_static_replay.get(p['name'])
            observed=stats(*trace[:2])['observed_status'] if trace is not None else 'failed'
            regimes.append(500.*(reference!='unresolved' and observed!=reference))
        groups['static_regime_barrier']=float(np.dot(regimes,regimes))
        residual=np.r_[residual,regimes]
        self.last_static_shapes={}
        for p in self.spec['fixed']:
            status=self.static.targets[p['name']]['observed_status'];cycling=status=='cycling'
            trace=self.last_static_replay.get(p['name'])
            if trace is None:
                extra=np.full(6 if cycling else 2,100.)
                self.last_static_shapes[p['name']]=dict(passes=False,error='incomplete static trajectory')
            else:
                check=autonomous_shape_check(*trace[:2],*self.refs[p['name']][:2],cycling=cycling)
                self.last_static_shapes[p['name']]=check
                errors=np.asarray(check['relative_errors']);tolerances=np.asarray(check['tolerances'])
                extra=np.r_[errors/.03,10*np.maximum(abs(errors)-.8*tolerances,0.)/.01]
            groups[p['name']+'_static_shape']=float(extra@extra);residual=np.r_[residual,extra]
        return (residual,groups) if parts else residual


def shape_checks(objective,model):
    if objective.last_train_paths is None or len(objective.last_static_replay)!=len(objective.spec['fixed']):
        return dict(eligible=False,error='incomplete training trajectories',shape_passes=0)
    checks=training_checks(objective)
    shape=[]
    for p in objective.spec['train']:
        t,x,_=objective.last_train_paths[p['name']];tr,truth,_=objective.refs[p['name']]
        shape.append(dict(name=p['name'],**peak_shape_check(t,x,tr,truth,tolerance=HARD_SHAPE_TOLERANCE)))
    target=np.asarray(objective.spec['rest']);rest,number=stable_rest(model,target[2],target)
    error=float(abs(rest-target).max());field_error=float(abs(model.field(model.to_latent(target))).max())
    eigen=np.linalg.eigvals(model.region(model.to_latent(target),20.,target[2])[0][:2,:2])
    rest_ok=error<2e-10 and field_error<2e-10 and max(eigen.real)<-1e-10
    checks.update(shape=shape,shape_passes=sum(s['passes'] for s in shape),
        max_height_relative_error=max(s['max_height_relative_error'] for s in shape),
        max_width_relative_error=max(s['max_width_relative_error'] for s in shape),
        exact_rest=bool(rest_ok),rest=rest.tolist(),stable_rest_count=number,
        rest_max_abs_error=error,rest_field_max_abs_error=field_error,
        rest_eigenvalues=[[float(z.real),float(z.imag)] for z in eigen])
    static_shape=[dict(name=p['name'],**objective.last_static_shapes[p['name']]) for p in objective.spec['fixed']]
    checks.update(static_shape=static_shape,static_shape_passes=sum(s['passes'] for s in static_shape))
    checks['eligible']=bool(checks['eligible'] and rest_ok and all(s['passes'] for s in shape) and all(s['passes'] for s in static_shape))
    return checks


def parameterization(seed,spec,family):
    """Remove two derived offsets; keep the source rest in the first stable strip."""
    original=CalciumTransformPWA.load(SOURCE)
    n=seed.latent.region_count;nc=2*n+6;m=n-1;dim=FAMILIES[family]
    active=np.array([j for j in range(nc) if j not in [2*n,2*n+5]])
    lower_u=max(float(seed.coordinate.forward(spec['C_lower'])),float(seed.coordinate.forward(spec['rest'][0]))+1e-5)
    mapping=OrderedThresholds(original.latent.thresholds,lower_u,float(seed.coordinate.forward(spec['C_upper'])))
    coefficients=np.asarray(seed.metadata['parameters'])
    geometry=np.array([seed.mix_linear,seed.mix_quadratic])[:dim]
    z=np.r_[coefficients[active],mapping.encode(seed.latent.thresholds),geometry]
    full_lo=np.r_[np.full(n,-12.),np.full(n,-40.),-.5,-40.,-28.,-10.,-10.,-10.]
    full_hi=np.r_[np.full(n,5.),np.full(n,10.),.5,8.,8.,2.,10.,10.]
    for j,label in enumerate(seed.metadata['target_types']):
        if label.endswith('node'):full_hi[n+j]=12.
    lo=np.r_[full_lo[active],np.full(m,-np.log(4)),[-.08,-.02][:dim]]
    hi=np.r_[full_hi[active],np.full(m,np.log(4)),[.08,.02][:dim]]
    z=np.clip(z,lo+1e-8,hi-1e-8);na=len(active)
    center=float(seed.gate.forward(spec['rest'][1]))
    def make(p):
        coefficients=np.zeros(nc);coefficients[active]=p[:na]
        theta=mapping.decode(p[na:na+m]);meta=copy.deepcopy(seed.metadata)
        meta.update(parameters=coefficients.tolist(),accepted=False,model_family=family,
            active_fit_parameters=len(p),parameter_count=len(p),free_coefficient_parameters=na,
            threshold_parameter_count=m,mixing_parameter_count=dim,
            physical_thresholds_C=seed.coordinate.inverse(theta).tolist(),
            threshold_shifts=p[na:na+m].tolist(),raw_fit_vector=p.tolist(),
            thresholds_policy='ordered U gaps +/-ln(4); first threshold strictly above exact baseline rest',
            source_chain_rule='Udot=phi_prime*fC-(a+2*b*(V-center))*psi_prime*fh; Vdot=psi_prime*fh',
            current_validation='results/fm_anchored_shapes/; inherited legacy losses are historical',
            initialization='exact fixed LR stable rest; two constants derived; 20 s input delay',
            calcium_clipping=False)
        model=decode_coordinates(seed.case,theta,coefficients,meta['mu_sign'],meta['k_sign'],meta,
                                 seed.coordinate,seed.h_reference,seed.h_kind)
        model.mix_linear=float(p[na+m]) if dim else 0.
        model.mix_quadratic=float(p[na+m+1]) if dim>1 else 0.
        model.mix_center=center;model.__post_init__()
        model.metadata['physical_field_PWA']=bool(seed.coordinate.kind=='identity' and seed.h_kind=='identity' and not model.mix_quadratic)
        return anchor_equilibrium(model,spec['rest'])
    return z,lo,hi,active,m,make


def fresh_tests():
    specs=[dict(name=f'anchored_heldout_tau_{tau}',duration=480. if tau==160. else 300.,
                tau=tau,baseline=.45,jumps=[[20.,.7]]) for tau in [6.,24.,48.,90.,160.]]
    rng=np.random.default_rng(20261029)
    times=20.+np.r_[0.,np.cumsum(rng.uniform(18.,28.,7))]
    specs.append(dict(name='anchored_heldout_irregular',duration=320.,tau=27.,baseline=.45,
                     jumps=list(map(list,zip(times.tolist(),rng.uniform(.2,.3,len(times)).tolist())))))
    return specs


def translated_start(source,spec,z,active,m,make):
    """Translate thresholds together with equilibrium, preserving latent orbits.

    A constant-only projection can cross several switching strips. Translating
    the partition as well gives a much closer continuation of the source model.
    """
    p=z.copy();na=len(active);dim=len(p)-na-m
    p[:na]=np.asarray(source.metadata['parameters'])[active]
    if dim:p[na+m:]=np.array([source.mix_linear,source.mix_quadratic])[:dim]
    target=make(p).to_latent(spec['rest'])
    rest,_=stable_rest(source,spec['rest'][2],spec['rest'])
    shift=source.to_latent(rest)-target
    original=CalciumTransformPWA.load(SOURCE)
    mapping=OrderedThresholds(original.latent.thresholds,
        max(float(source.coordinate.forward(spec['C_lower'])),float(source.coordinate.forward(spec['rest'][0]))+1e-5),
        float(source.coordinate.forward(spec['C_upper'])))
    p[na:na+m]=mapping.encode(source.latent.thresholds-shift[0])
    return p


def prepare(out,args):
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory')
    cache=out/'cache/FM';cache.mkdir(parents=True)
    previous=Path(args.reference_cache)/'FM'
    spec=json.loads((previous/'data.json').read_text())
    spec['previous_tests_not_blind']=spec['test'];spec['test']=fresh_tests()
    for p in spec['fixed']+spec['train']:
        shutil.copyfile(previous/f'{p["name"]}.npz',cache/f'{p["name"]}.npz')
    rest=source_rest('FM',.45)
    for p in spec['test']:
        t,x,_=simulate('FM',rest,**{k:v for k,v in p.items() if k!='name'},
                       dt=.05,rtol=2e-11,atol=2e-13,max_step=.05)
        np.savez_compressed(cache/f'{p["name"]}.npz',t=t,x=x)
    atomic(cache/'data.json',spec);rebuild_training_field(cache)
    jobs=[]
    for restart in range(args.restarts):
        for family in FAMILIES:
            start=SOURCE if restart==0 else SOURCE.parents[1]/'mixed_coordinate_candidates'/f'FM_{family}.json'
            jobs.append(dict(name=f'{family}_{restart}',family=family,restart=restart,
                seed=args.seed+restart,starting_model=str(start),starting_sha256=hashlib.sha256(start.read_bytes()).hexdigest()))
    config=dict(version=1,jobs=jobs,budget=args.budget,global_iterations=args.global_iterations,
        workers=args.workers,seed=args.seed,source_model=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        parameters=dict(control=27,linear=28,quadratic=29),regions=8,rest=list(rest),
        shape_tolerance=HARD_SHAPE_TOLERANCE,fit_shape_barrier_at=FIT_SHAPE_TOLERANCE,
        geometry_bounds=dict(a=[-.08,.08],b=[-.02,.02]),threshold_gap_bounds=[-float(np.log(4)),float(np.log(4))],
        selection='independent DOP853: exact stable LR rest, all static regimes, six dynamic peak counts, every peak height and width within 10%; then training loss',
        heldout_protocols=spec['test'],heldout_used_for_field_initialization=False,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()+code_hash().encode()).hexdigest(),
        python=sys.version,numpy=np.__version__,scipy=__import__('scipy').__version__)
    atomic(out/'config.json',config)


def fit_job(out,name):
    config=json.loads((out/'config.json').read_text());job=next(j for j in config['jobs'] if j['name']==name)
    family=job['family'];directory=out/'trials'/name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():raise ValueError('Trial already completed')
    seed=CalciumTransformPWA.load(job['starting_model']);objective=ShapeObjective(out,'FM',seed)
    z,lo,hi,active,m,make=parameterization(seed,objective.spec,family);na=len(active)
    with np.load(out/'cache/FM/field.npz') as data:points=data['points']
    def field_residual(p):
        model=make(p);latent=model.to_latent(points);error=model.field(latent)-model.source_field(latent)
        physical=np.column_stack([(error[:,0]+model.mix_derivative(latent[:,1])*error[:,1])/model.coordinate.derivative(points[:,0]),
                                  error[:,1]/model.gate.derivative(points[:,1])])
        return np.r_[(physical/[.2,.05]).ravel()/np.sqrt(len(points)),.01*(p[:na]-z[:na])]
    started=time.monotonic();field=least_squares(field_residual,z,bounds=(lo,hi),max_nfev=100)
    make(field.x).save(directory/'field_seed.json')
    count=0;limit=10**9;best=np.inf;bestz=z.copy();eligible_best=np.inf;eligiblez=None;history=[];snapshots=[]
    class Limit(Exception):pass
    def residual(p):
        nonlocal count,best,bestz,eligible_best,eligiblez
        if count>=limit:raise Limit()
        count+=1;model=make(p);r=objective(model)
        full=np.r_[r,.005*(p[:na]-z[:na]),.03*p[na:na+m],.03*p[na+m:]];loss=float(full@full)
        if loss<best:
            best,bestz=loss,p.copy()
            if not history or count-history[-1]['evaluation']>=100:
                check=shape_checks(objective,model)
                history.append(dict(evaluation=count,loss=float(r@r),seconds=time.monotonic()-started,
                                    eligible=check['eligible'],shape_passes=check.get('shape_passes',0)))
                model.save(directory/'checkpoint.json')
                print(json.dumps(dict(job=name,**history[-1])),flush=True)
        # A lower unconstrained loss must never displace a feasible candidate.
        if loss<eligible_best and objective.last_train_paths is not None:
            check=shape_checks(objective,model)
            if check['eligible']:
                eligible_best,eligiblez=loss,p.copy()
                model.save(directory/'eligible_checkpoint.json')
                if not snapshots or loss<snapshots[-1][0]*.98:
                    snapshots.append((loss,p.copy()))
        return full
    candidates=[('incumbent',z),('field',field.x)]
    common=CalciumTransformPWA.load(SOURCE)
    for label,start in [('translated_common',common),('translated_incumbent',seed)]:
        try:
            p=translated_start(start,objective.spec,z,active,m,make)
            if np.all(p>=lo) and np.all(p<=hi):candidates.append((label,p))
        except ERRORS:pass
    ranks=[(float((r:=residual(p))@r),label,p) for label,p in candidates]
    # A deterministic input-gain scan retains the narrow feasible vicinity of
    # the original orbits before the wider global search. Same grid in all jobs.
    n=seed.latent.region_count;loc={int(full):i for i,full in enumerate(active)}
    base=min(ranks,key=lambda a:a[0])[2].copy()
    for shift in np.linspace(-.25,.05,13):
        p=base.copy();p[loc[2*n+1]]=np.clip(p[loc[2*n+1]]+shift,lo[loc[2*n+1]],hi[loc[2*n+1]])
        ranks.append((float((r:=residual(p))@r),'input_gain_scan',p))
    # Matched four-dimensional global search; all families share qb and cI.
    indices={'control':[loc[2*n+1],loc[2*n+4],loc[2*n+2],loc[2*n+3]],
             'linear':[loc[2*n+1],loc[2*n+4],loc[2*n+2],na+m],
             'quadratic':[loc[2*n+1],loc[2*n+4],na+m,na+m+1]}[family]
    widths={'control':[.6,.35,.6,.6],'linear':[.6,.35,.6,.08],'quadratic':[.6,.35,.08,.02]}[family]
    base=min(ranks,key=lambda a:a[0])[2]
    bounds=[(max(lo[j],base[j]-w),min(hi[j],base[j]+w)) for j,w in zip(indices,widths)]
    def unpack(s):p=base.copy();p[indices]=s;return p
    de=differential_evolution(lambda s:float((r:=residual(unpack(s)))@r),bounds,x0=base[indices],
        seed=job['seed'],popsize=8,maxiter=config['global_iterations'],polish=False,tol=0.,atol=0.)
    global_vector=unpack(de.x);ranks.append((float(de.fun),'global',global_vector))
    ranks.sort(key=lambda a:a[0]);runs=[];budget=config['budget']
    def jac(p):
        base=residual(p);cols=[]
        for j in range(len(p)):
            step=1e-4*max(1.,abs(p[j])) if j<na else 1e-3 if j<na+m else 1e-5
            if p[j]+step>hi[j]:step=-step
            q=p.copy();q[j]+=step;cols.append((residual(q)-base)/step)
        return np.column_stack(cols)
    for _,label,p in ranks[:2]:
        limit=count+budget
        try:
            fit=least_squares(residual,p,jac=jac,bounds=(lo,hi),x_scale='jac',max_nfev=200)
            status=fit.message
        except Limit:status='residual evaluation limit'
        runs.append(dict(method='least_squares',start=label,status=str(status)))
    limit=count+budget
    start=eligiblez.copy() if eligiblez is not None else bestz.copy()
    try:
        fit=minimize(lambda p:float((r:=residual(p))@r),start,method='Powell',bounds=list(zip(lo,hi)),
                     options=dict(maxfev=budget,xtol=1e-5,ftol=1e-7))
        status=fit.message
    except Limit:status='residual evaluation limit'
    runs.append(dict(method='Powell',status=str(status)))
    # Verification candidates are nominated entirely from training outcomes.
    vectors=[('unconstrained_best',bestz)]
    if eligiblez is not None:vectors.insert(0,('eligible_best',eligiblez))
    if snapshots:vectors.append(('earlier_eligible',snapshots[max(0,len(snapshots)//2-1)][1]))
    verified=[]
    for label,p in vectors:
        model=make(p);path=directory/f'{label}.json';model.save(path)
        try:
            r,parts=objective(model,precise=True,parts=True);check=shape_checks(objective,model)
            record=dict(label=label,status='verified',verified_train_loss=float(r@r),checks=check,parts=parts,model=str(path.relative_to(out)))
        except ERRORS as exc:record=dict(label=label,status='failed',error=str(exc),checks=dict(eligible=False),model=str(path.relative_to(out)))
        verified.append(record)
    valid=[r for r in verified if r['status']=='verified'];eligible=[r for r in valid if r['checks']['eligible']]
    selected=min(eligible or valid,key=lambda r:r['verified_train_loss']) if valid else verified[0]
    shutil.copyfile(out/selected['model'],directory/'candidate.json')
    result=dict(**job,**{k:v for k,v in selected.items() if k!='model'},model=str((directory/'candidate.json').relative_to(out)),
        all_verifications=verified,evaluations=count,seconds=time.monotonic()-started,parameters=len(z),regions=8,
        selection_scope='eligible training candidate' if eligible else 'diagnostic only: no eligible candidate')
    atomic(directory/'fit.json',dict(evaluations=count,history=history,runs=runs,bounds=[lo.tolist(),hi.tolist()],
        global_search=dict(indices=indices,bounds=bounds,evaluations=de.nfev,loss=float(de.fun)),source_field_nfev=field.nfev))
    atomic(directory/'result.json',result)
    print(json.dumps(dict(job=name,loss=result.get('verified_train_loss'),checks=result['checks'],evaluations=count)),flush=True)


def evaluate(out):
    config=json.loads((out/'config.json').read_text());source=CalciumTransformPWA.load(SOURCE)
    objective=ShapeObjective(out,'FM',source);trials=[json.loads(p.read_text()) for p in (out/'trials').glob('*/result.json')]
    selected=[]
    for family in FAMILIES:
        rows=[r for r in trials if r['family']==family and r['status']=='verified'];eligible=[r for r in rows if r['checks']['eligible']]
        if not rows:raise RuntimeError(f'No verified {family} candidate')
        selected.append(copy.deepcopy(min(eligible or rows,key=lambda r:r['verified_train_loss'])))
    r,parts=objective(source,precise=True,parts=True)
    original=dict(family='original',model=str(SOURCE),verified_train_loss=float(r@r),checks=shape_checks(objective,source),parts=parts)
    selected.insert(0,original)
    atomic(out/'train_selection.json',selected)
    for row in selected:
        family=row['family'];model=source if family=='original' else CalciumTransformPWA.load(out/row['model'])
        if family!='original':
            directory=out/'selected';directory.mkdir(exist_ok=True);model.save(directory/f'{family}.json')
        records=diagnostics(model,objective,objective.spec['test'],objective.refs,out/'diagnostics'/family)
        for record,p in zip(records,objective.spec['test']):
            if record['completed']:
                with np.load(out/'diagnostics'/family/f'{p["name"]}.npz') as data:
                    record['shape_check']=peak_shape_check(data['t'],data['candidate'],data['t'],data['lr'])
        row['heldout']=dict(aggregate=aggregate(records),protocols=records,
            shape_passes=sum(p.get('shape_check',{}).get('passes',False) for p in records))
        atomic(out/'summary.json',dict(config=config,selected=selected,all_trials=trials))
        print(json.dumps(dict(family=family,train=row['verified_train_loss'],eligible=row['checks']['eligible'],
                             heldout=row['heldout']['aggregate'],shape_passes=row['heldout']['shape_passes'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['all','prepare','fit','evaluate'],default='all');parser.add_argument('--job')
    parser.add_argument('--reference-cache',default='results/fm_mixed_coordinates/cache')
    parser.add_argument('--budget',type=int,default=1500);parser.add_argument('--global-iterations',type=int,default=40)
    parser.add_argument('--restarts',type=int,default=2);parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--seed',type=int,default=20261029);args=parser.parse_args();out=Path(args.output).resolve()
    if min(args.budget,args.restarts,args.workers)<1 or args.global_iterations<0:parser.error('Invalid optimization limits')
    if args.stage=='fit':
        if not args.job:parser.error('--job required')
        fit_job(out,args.job);return
    if args.stage=='evaluate':evaluate(out);return
    prepare(out,args)
    if args.stage=='prepare':return
    waiting=json.loads((out/'config.json').read_text())['jobs'];running=[]
    while waiting or running:
        while waiting and len(running)<args.workers:
            job=waiting.pop(0);directory=out/'trials'/job['name'];directory.mkdir(parents=True)
            log=(directory/'worker.log').open('w')
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--output',str(out),'--stage','fit','--job',job['name']],
                               stdout=log,stderr=subprocess.STDOUT)
            running.append((job['name'],p,log))
        for name,p,log in list(running):
            if p.poll() is not None:
                log.close();running.remove((name,p,log));print(json.dumps(dict(job=name,exit_code=p.returncode)),flush=True)
                if p.returncode:raise RuntimeError(f'{name} failed; see worker.log')
        time.sleep(.25)
    evaluate(out)


if __name__=='__main__':main()
