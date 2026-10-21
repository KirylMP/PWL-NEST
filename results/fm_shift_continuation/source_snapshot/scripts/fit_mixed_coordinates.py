"""Controlled FM pilot: vertical, tilted and quadratic switching surfaces.

All families refit a continuous PWA field in their OWN coordinates. In particular,
the quadratic case is not a nonlinear conjugation of an already fitted PWA field.
The same physical training criterion and independent DOP853 checks apply to all.
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
from scipy.optimize import differential_evolution, least_squares, minimize

from pwl_nest.calcium_coordinates import CalciumTransformPWA, decode_coordinates, simulate_coordinates
from pwl_nest.models import ROOT
from pwl_nest.peak_timing import absolute_peak_time_residual, burst_end_residual
from pwl_nest.protocols import simulate
from pwl_nest.rest_protocols import event_residual, native_events
from pwl_nest.thresholds import OrderedThresholds
from overnight_coordinate_grid import atomic, code_hash, source_rest
from refine_server_candidates import TimingObjective, aggregate, diagnostics, rebuild_training_field
from fit_fm_input_threshold import SOURCE, training_checks

FAMILIES = {'control': 0, 'linear': 1, 'quadratic': 2}
ERRORS = (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError)


class MixedObjective(TimingObjective):
    """Same burst criterion as the preceding experiment, with physical readout.

    Extrema of U need not be extrema of C. Even the control uses numerical
    physical trajectories for static calibration to keep the comparison matched.
    """
    def __call__(self, model, precise=False, parts=False):
        self.last_static_replay = {}; self.last_train_paths = None
        groups = {}; nstatic = 8*len(self.spec['fixed'])+len(model.latent.thresholds)
        try:
            for p in self.spec['fixed']:
                kw = {k:v for k,v in p.items() if k!='name'}
                if precise:
                    trace=simulate_coordinates(model, **kw, dt=.2, rtol=2e-11, atol=2e-13, max_step=.25)
                else:
                    trace=native_events(model, **kw, jumps=[], dt=.2, maxdt=.02, native=self.dynamic)
                self.last_static_replay[p['name']]=trace
            residual=self.static(model,replay=self.last_static_replay)
        except ERRORS:
            if precise:raise
            residual=np.full(nstatic,100.)
        groups['fixed_and_spectral_anchors']=float(residual@residual)
        try:
            rest,count,paths=self.paths(model,self.spec['train'],precise)
            anchor=np.r_[(rest[:2]-np.asarray(self.spec['rest'])[:2])/[.01,.05],10*(count-1)]
        except ERRORS:
            if precise:raise
            paths=None;anchor=np.full(3,100.)
        for p in self.spec['train']:
            tr,truth,_=self.refs[p['name']]
            if paths is None:
                event=np.full(self.lengths[p['name']],100.)
                timing=np.full(len(absolute_peak_time_residual(tr,truth,tr,truth)),100.)
                end=np.full(len(burst_end_residual(tr,truth,tr,truth)),100.)
            else:
                t,x,_=paths[p['name']]
                event=event_residual(t,x,tr,truth)
                timing=absolute_peak_time_residual(t,x,tr,truth)
                end=burst_end_residual(t,x,tr,truth)
            for label,r in [('events',event),('timing',timing),('burst_end',end)]:
                residual=np.r_[residual,r];groups[p['name']+'_'+label]=float(r@r)
        residual=np.r_[residual,anchor];groups['own_rest']=float(anchor@anchor)
        return (residual,groups) if parts else residual


def fresh_tests():
    specs=[dict(name=f'mixed_heldout_tau_{tau}',duration=380. if tau==190. else 260.,
                tau=tau,baseline=.45,jumps=[[20.,.7]]) for tau in [11.,42.,72.,125.,190.]]
    rng=np.random.default_rng(20261028)
    times=20.+np.r_[0.,np.cumsum(rng.uniform(13.,25.,7))]
    specs.append(dict(name='mixed_heldout_irregular',duration=300.,tau=19.,baseline=.45,
                     jumps=list(map(list,zip(times.tolist(),rng.uniform(.2,.32,len(times)).tolist())))))
    return specs


def prepare(out,args):
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
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
    config=dict(version=1,families=FAMILIES,budget_per_stage=args.budget,seed=args.seed,
        source_model=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        starting_model=str(Path(args.seed_model).resolve()),
        starting_sha256=hashlib.sha256(Path(args.seed_model).read_bytes()).hexdigest(),
        global_iterations=args.global_iterations,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()+code_hash().encode()).hexdigest(),
        coordinate='U=phi(C)-a*(V-center)-b*(V-center)^2; V=psi(h); I unchanged',
        center='source baseline equilibrium V, fixed before fitting',
        geometry_bounds=dict(a=[-.06,.06],b=[-.012,.012]),
        criterion='previous FM burst criterion, all physical static traces; no new acceptance relaxation',
        region_count=8,parameters=dict(control=29,linear=30,quadratic=31),
        selection='physical, all six training peak counts and eight static regimes; then DOP853 training loss',
        heldout_protocols=spec['test'],heldout_used_for_initialization=False,
        python=sys.version,numpy=np.__version__,scipy=__import__('scipy').__version__)
    atomic(out/'config.json',config)


def parameterization(seed,spec,family):
    if 'exact_rest_constraint' in seed.metadata:
        raise ValueError('Use fit_anchored_shapes to preserve the exact rest constraint')
    n=seed.latent.region_count;nc=2*n+6;m=n-1;dim=FAMILIES[family]
    mapping=OrderedThresholds(seed.latent.thresholds,float(seed.coordinate.forward(spec['C_lower'])),
                              float(seed.coordinate.forward(spec['C_upper'])))
    z=np.r_[seed.metadata['parameters'],np.zeros(m+dim)]
    lo=np.r_[np.full(n,-12.),np.full(n,-40.),-.5,-40.,-28.,-10.,-10.,-10.,np.full(m,-np.log(2)),[-.06,-.012][:dim]]
    hi=np.r_[np.full(n,5.),np.full(n,10.),.5,8.,8.,2.,10.,10.,np.full(m,np.log(2)),[.06,.012][:dim]]
    for j,label in enumerate(seed.metadata['target_types']):
        if label.endswith('node'):hi[n+j]=12.
    center=float(seed.gate.forward(spec['rest'][1]))
    def make(p):
        theta=mapping.decode(p[nc:nc+m]);meta=copy.deepcopy(seed.metadata)
        meta.update(accepted=False,model_family=family,active_fit_parameters=len(p),parameter_count=len(p),
            parameters=p[:nc].tolist(),threshold_shifts=p[nc:nc+m].tolist(),
            physical_thresholds_C=seed.coordinate.inverse(theta).tolist(),
            threshold_cross_section='C at V=center only; actual switching surface U=theta',
            thresholds_policy='ordered U gaps +/-ln(2); common polynomial shear',
            source_chain_rule='Udot=phi_prime*fC-(a+2*b*(V-center))*psi_prime*fh; Vdot=psi_prime*fh',
            initialization='own stable equilibrium, not fitted separately; 20 s input delay',
            raw_fit_vector=p.tolist(),calcium_clipping=False,
            current_validation='see accompanying result.json; legacy loss metadata is historical')
        model=decode_coordinates(seed.case,theta,p[:nc],meta['mu_sign'],meta['k_sign'],meta,
                                 seed.coordinate,seed.h_reference,seed.h_kind)
        model.mix_linear=float(p[nc+m]) if dim else 0.
        model.mix_quadratic=float(p[nc+m+1]) if dim>1 else 0.
        model.mix_center=center
        model.__post_init__()
        return model
    return z,lo,hi,nc,m,make


def fit_job(out,family):
    config=json.loads((out/'config.json').read_text());directory=out/'trials'/family
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():raise ValueError('Trial already completed')
    seed=CalciumTransformPWA.load(config.get('starting_model',SOURCE));objective=MixedObjective(out,'FM',seed)
    z,lo,hi,nc,m,make=parameterization(seed,objective.spec,family)
    with np.load(out/'cache/FM/field.npz') as data:points=data['points']
    # Source samples and derivative pullback must BOTH follow the trial transform.
    def field_residual(p):
        model=make(p);latent=model.to_latent(points);error=model.field(latent)-model.source_field(latent)
        physical=np.column_stack([(error[:,0]+model.mix_derivative(latent[:,1])*error[:,1])/model.coordinate.derivative(points[:,0]),
                                  error[:,1]/model.gate.derivative(points[:,1])])
        return np.r_[(physical/[.2,.05]).ravel()/np.sqrt(len(points)),.01*(p[:nc]-z[:nc])]
    started=time.monotonic();field=least_squares(field_residual,z,bounds=(lo,hi),max_nfev=60)
    make(field.x).save(directory/'field_seed.json')
    count=0;limit=3;best=np.inf;bestz=z.copy();history=[]
    class Limit(Exception):pass
    def residual(p):
        nonlocal count,best,bestz
        if count>=limit:raise Limit()
        count+=1;r=objective(make(p));full=np.r_[r,.005*(p[:nc]-z[:nc]),.03*p[nc:nc+m],.03*p[nc+m:]]
        loss=float(full@full)
        if loss<best:
            best,bestz=loss,p.copy()
            if not history or count-history[-1]['evaluation']>=100:
                history.append(dict(evaluation=count,loss=float(r@r),seconds=time.monotonic()-started))
                make(p).save(directory/'checkpoint.json')
                print(json.dumps(dict(family=family,**history[-1])),flush=True)
        return full
    global_result=None
    if config.get('global_iterations',0):
        # Equal-dimensional, equal-cap global searches. The control spends its
        # budget on existing coefficients; mixed models explore their geometry.
        indices={'control':[2*(m+1),2*(m+1)+1,2*(m+1)+4],
                 'linear':[nc+m,2*(m+1),2*(m+1)+1],
                 'quadratic':[nc+m,nc+m+1,2*(m+1)]}[family]
        widths={'control':[.08,.3,.15],'linear':[.06,.08,.3],'quadratic':[.06,.012,.08]}[family]
        bounds=[(max(lo[j],z[j]-w),min(hi[j],z[j]+w)) for j,w in zip(indices,widths)]
        def global_vector(s):
            p=z.copy();p[indices]=s;return p
        limit=count+8*len(indices)*(config['global_iterations']+1)+2
        de=differential_evolution(lambda s:float((r:=residual(global_vector(s)))@r),bounds,
            x0=z[indices],seed=config['seed'],popsize=8,maxiter=config['global_iterations'],
            polish=False,tol=0.,atol=0.)
        global_result=dict(evaluations=de.nfev,indices=indices,bounds=bounds,loss=float(de.fun),parameters=de.x.tolist())
        candidates=[('incumbent',z),('field',field.x),('global',global_vector(de.x))]
        limit=count+3
    else:
        rng=np.random.default_rng(config['seed'])
        perturb=z+rng.normal(0,.02,len(z));perturb[nc+m:]*=.1
        candidates=[('original',z),('field',field.x),('perturbed',np.clip(perturb,lo+1e-8,hi-1e-8))]
    ranks=sorted([(float((r:=residual(p))@r),label,p) for label,p in candidates],key=lambda q:q[0])
    def jac(p):
        base=residual(p);cols=[]
        for j in range(len(p)):
            step=1e-4*max(1.,abs(p[j])) if j<nc else 1e-3 if j<nc+m else 1e-5
            if p[j]+step>hi[j]:step=-step
            q=p.copy();q[j]+=step;cols.append((residual(q)-base)/step)
        return np.column_stack(cols)
    runs=[];budget=config['budget_per_stage']
    for _,label,p in ranks[:2]:
        limit=count+budget
        try:
            fit=least_squares(residual,p,jac=jac,bounds=(lo,hi),x_scale='jac',max_nfev=100)
            status=fit.message
        except Limit:status='residual evaluation limit'
        runs.append(dict(method='least_squares',start=label,status=str(status)))
    limit=count+budget
    try:
        fit=minimize(lambda p:float((r:=residual(p))@r),bestz,method='Powell',bounds=list(zip(lo,hi)),
                     options=dict(maxfev=budget,xtol=1e-5,ftol=1e-7))
        status=fit.message
    except Limit:status='residual evaluation limit'
    runs.append(dict(method='Powell',status=str(status)))
    model=make(bestz);model.save(directory/'candidate.json')
    atomic(directory/'fit.json',dict(evaluations=count,history=history,runs=runs,parameters=bestz.tolist(),
        bounds=[lo.tolist(),hi.tolist()],initial_scores=[dict(score=s,start=name) for s,name,_ in ranks],
        source_field_evaluations=field.nfev,global_search=global_result))
    verify(out,family,objective,model,seconds=time.monotonic()-started)


def verify(out,family,objective=None,model=None,seconds=None):
    directory=out/'trials'/family
    model=model or CalciumTransformPWA.load(directory/'candidate.json')
    objective=objective or MixedObjective(out,'FM',model)
    try:
        residual,parts=objective(model,precise=True,parts=True)
        checks=training_checks(objective)
        record=dict(family=family,status='verified',verified_train_loss=float(residual@residual),
            parts=parts,checks=checks,parameters=model.metadata['active_fit_parameters'],regions=model.latent.region_count,
            a=model.mix_linear,b=model.mix_quadratic,center=model.mix_center,seconds=seconds)
    except ERRORS as exc:
        record=dict(family=family,status='failed',error=str(exc),checks=dict(eligible=False),seconds=seconds)
    atomic(directory/'result.json',record);print(json.dumps(record),flush=True)
    return record


def evaluate(out):
    source=CalciumTransformPWA.load(SOURCE);objective=MixedObjective(out,'FM',source)
    r,parts=objective(source,precise=True,parts=True)
    original=dict(family='original',verified_train_loss=float(r@r),parts=parts,checks=training_checks(objective))
    rows=[original]+[json.loads((out/'trials'/f/'result.json').read_text()) for f in FAMILIES]
    # One candidate per predeclared family, frozen BEFORE any test readout.
    atomic(out/'train_selection.json',rows)
    for row in rows:
        family=row['family'];model=source if family=='original' else CalciumTransformPWA.load(out/'trials'/family/'candidate.json')
        records=diagnostics(model,objective,objective.spec['test'],objective.refs,out/'diagnostics'/family)
        row['heldout']=dict(aggregate=aggregate(records),protocols=records)
        atomic(out/'summary.json',dict(config=json.loads((out/'config.json').read_text()),candidates=rows))
        print(json.dumps(dict(family=family,train=row.get('verified_train_loss'),heldout=row['heldout']['aggregate'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['all','prepare','fit','verify','evaluate'],default='all')
    parser.add_argument('--family',choices=list(FAMILIES))
    parser.add_argument('--reference-cache',default=str(ROOT/'results/fm_one_input_threshold/cache'))
    parser.add_argument('--budget',type=int,default=800)
    parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--seed-model',default=str(SOURCE))
    parser.add_argument('--global-iterations',type=int,default=0)
    args=parser.parse_args();out=Path(args.output).resolve()
    if args.budget<1 or args.global_iterations<0:parser.error('Invalid optimization budget')
    if args.stage in ['fit','verify'] and not args.family:parser.error('--family required')
    if args.stage=='fit':fit_job(out,args.family);return
    if args.stage=='verify':verify(out,args.family);return
    if args.stage=='evaluate':evaluate(out);return
    prepare(out,args)
    if args.stage=='prepare':return
    running=[]
    for family in FAMILIES:
        directory=out/'trials'/family;directory.mkdir(parents=True)
        log=(directory/'worker.log').open('w')
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--output',str(out),
                            '--stage','fit','--family',family],stdout=log,stderr=subprocess.STDOUT)
        running.append((family,p,log))
    while running:
        for family,p,log in list(running):
            if p.poll() is not None:
                log.close();running.remove((family,p,log))
                print(json.dumps(dict(family=family,exit_code=p.returncode)),flush=True)
                if p.returncode:raise RuntimeError(f'{family} failed; see worker.log')
        time.sleep(.25)
    evaluate(out)


if __name__=='__main__':main()
