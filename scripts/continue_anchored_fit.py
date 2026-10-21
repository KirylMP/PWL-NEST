"""Continue the frozen FM fit on explicitly promoted failures, then new inputs."""
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
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.peak_shape import peak_shape_check
from pwl_nest.protocols import simulate
from fit_anchored_shapes import ShapeObjective,shape_checks,fit_job
from overnight_coordinate_grid import atomic
from refine_server_candidates import rebuild_training_field,diagnostics,aggregate

SEED=Path('models/anchored_shape_candidates/FM_control.json')
PREVIOUS=Path('results/fm_anchored_shapes/cache/FM')


def prepare(out,args):
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    cache=out/'cache/FM';cache.mkdir(parents=True)
    previous=Path(args.reference_cache);seed=Path(args.seed_model)
    spec=json.loads((previous/'data.json').read_text())
    old_tests=copy.deepcopy(spec['test'])
    if args.promote is None:promoted=[p for p in old_tests if p['tau'] in (48.,160.)]
    else:
        names=set(args.promote);promoted=[p for p in old_tests if p['name'] in names]
        if names!={p['name'] for p in promoted}:raise ValueError('Unknown protocol to promote')
    spec['train']+=promoted
    spec['previous_tests_not_blind']+=old_tests
    spec['promoted_prior_failures']+=promoted
    phase=args.new_protocols=='phase';irregular=args.new_protocols=='irregular'
    prefix='phase' if phase else 'continuation'
    taus=[9.,29.,58.,105.,170.] if phase else [8.,33.,54.,115.,180.]
    spec['test']=[dict(name=f'{prefix}_heldout_tau_{tau}',duration=(510. if phase else 540.) if tau==taus[-1] else 360.,
        tau=tau,baseline=.45,jumps=[[20.,.7]]) for tau in taus]
    rng=np.random.default_rng(20261101 if phase else 20261030)
    times=20.+np.r_[0.,np.cumsum(rng.uniform(16.,30.,7))]
    spec['test'].append(dict(name=f'{prefix}_heldout_irregular',duration=340.,tau=31. if phase else 22.,baseline=.45,
        jumps=list(map(list,zip(times.tolist(),rng.uniform(.2,.3,len(times)).tolist())))))
    if irregular:
        spec['test']=[dict(name=f'width_heldout_tau_{tau}',duration=480. if tau==150. else 360.,
            tau=tau,baseline=.45,jumps=[[20.,.7]]) for tau in [12.,52.,150.]]
        for j,tau in enumerate([19.,28.,40.]):
            rng=np.random.default_rng(20261102+j);times=20.+np.r_[0.,np.cumsum(rng.uniform(16.,30.,7))]
            spec['test'].append(dict(name=f'width_heldout_irregular_{tau}',duration=340.,tau=tau,baseline=.45,
                jumps=list(map(list,zip(times.tolist(),rng.uniform(.2,.28,len(times)).tolist())))))
    for p in spec['fixed']+spec['train']:
        shutil.copyfile(previous/f'{p["name"]}.npz',cache/f'{p["name"]}.npz')
    for p in spec['test']:
        t,x,_=simulate('FM',spec['rest'],**{k:v for k,v in p.items() if k!='name'},dt=.05,
            rtol=2e-11,atol=2e-13,max_step=.05)
        np.savez_compressed(cache/f'{p["name"]}.npz',t=t,x=x)
    atomic(cache/'data.json',spec);rebuild_training_field(cache)
    jobs=[dict(name=f'control_{i}',family='control',restart=i,seed=args.search_seed+i,
        search_space=['default','thresholds','kinetics'][i%3] if args.diverse_directions else 'default',
        starting_model=str(seed),starting_sha256=hashlib.sha256(seed.read_bytes()).hexdigest()) for i in range(args.restarts)]
    atomic(out/'config.json',dict(continuation=True,jobs=jobs,budget=args.budget,
        global_iterations=args.global_iterations,workers=args.workers,regions=8,parameters=27,
        constraints=f'exact LR rest, 8 static morphology/regime checks, all {len(spec["train"])} training peak counts/heights; dynamic width gate={not args.no_dynamic_width_gate}',
        source_model=str(seed),reference_cache=str(previous),timing_weight=args.timing_weight,
        phase_ceiling=args.phase_ceiling,search_seed=args.search_seed,
        dynamic_width_gate=not args.no_dynamic_width_gate,
        dynamic_width_residual_weight=args.dynamic_width_residual_weight,
        heldout_protocols=spec['test'],
        promoted=[p['name'] for p in promoted],heldout_used_for_field_initialization=False,
        selection='minimum independently verified eligible training loss; if none, keep incumbent and report diagnostic best',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path('scripts/fit_anchored_shapes.py'),Path('scripts/fit_mixed_coordinates.py'),Path('src/pwl_nest/peak_shape.py')]},
        python=sys.version,numpy=np.__version__,scipy=__import__('scipy').__version__))


def evaluate(out):
    config=json.loads((out/'config.json').read_text());source=Path(config.get('source_model',SEED))
    seed=CalciumTransformPWA.load(source);objective=ShapeObjective(out,'FM',seed)
    r,parts=objective(seed,precise=True,parts=True)
    baseline=dict(name='incumbent',model=str(source),verified_train_loss=float(r@r),checks=shape_checks(objective,seed))
    trials=[json.loads(p.read_text()) for p in (out/'trials').glob('*/result.json')]
    valid=[p for p in trials if p['status']=='verified']
    eligible=[p for p in valid if p['checks']['eligible']]
    chosen=min(eligible or valid,key=lambda p:p['verified_train_loss'])
    improves=bool(eligible and (not baseline['checks']['eligible'] or chosen['verified_train_loss']<baseline['verified_train_loss']))
    recommendation=dict(name=chosen['name'] if improves else 'incumbent',eligible_new_fit=improves,
        diagnostic_best=chosen['name'],diagnostic_best_loss=chosen['verified_train_loss'],
        verified_train_loss=chosen['verified_train_loss'] if improves else baseline['verified_train_loss'],
        reason='all expanded training checks passed and training loss improved' if improves else 'no eligible improvement; retain previous model')
    atomic(out/'recommendation_before_heldout.json',recommendation)
    rows=[baseline,copy.deepcopy(chosen)]
    for row in rows:
        path=source if row['name']=='incumbent' else out/row['model']
        model=CalciumTransformPWA.load(path)
        records=diagnostics(model,objective,objective.spec['test'],objective.refs,out/'diagnostics'/row['name'])
        for record,p in zip(records,objective.spec['test']):
            if record['completed']:
                with np.load(out/'diagnostics'/row['name']/f'{p["name"]}.npz') as data:
                    record['shape_check']=peak_shape_check(data['t'],data['candidate'],data['t'],data['lr'])
        row['heldout']=dict(aggregate=aggregate(records),protocols=records,
            shape_passes=sum(p.get('shape_check',{}).get('passes',False) for p in records))
        atomic(out/'summary.json',dict(recommendation=recommendation,compared=rows,trials=trials))
        print(json.dumps(dict(name=row['name'],eligible=row['checks']['eligible'],train=row['verified_train_loss'],
            heldout=row['heldout']['aggregate'],shape_passes=row['heldout']['shape_passes'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--stage',choices=['all','prepare','fit','evaluate'],default='all')
    parser.add_argument('--job');parser.add_argument('--budget',type=int,default=1200)
    parser.add_argument('--global-iterations',type=int,default=25);parser.add_argument('--restarts',type=int,default=3)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--seed-model',default=str(SEED))
    parser.add_argument('--reference-cache',default=str(PREVIOUS))
    parser.add_argument('--promote',nargs='*',help='Names of previous tests to move into training')
    parser.add_argument('--new-protocols',choices=['continuation','phase','irregular'],default='continuation')
    parser.add_argument('--search-seed',type=int,default=20261030)
    parser.add_argument('--diverse-directions',action='store_true')
    parser.add_argument('--phase-ceiling',type=float,help='Optional hard ceiling on training mean absolute peak timing error, seconds')
    parser.add_argument('--timing-weight',type=float,default=1.,help='Multiplier on absolute peak-time residuals; hard gates stay fixed')
    parser.add_argument('--dynamic-width-residual-weight',type=float,default=1.,help='Soft fit weight for dynamic peak width')
    parser.add_argument('--no-dynamic-width-gate',action='store_true',help='Keep dynamic width in reports and soft loss, but remove its hard eligibility gate')
    args=parser.parse_args();out=Path(args.output).resolve()
    if min(args.budget,args.restarts,args.workers)<1 or args.global_iterations<0:parser.error('Invalid optimization limits')
    if not np.isfinite(args.timing_weight) or args.timing_weight<=0:parser.error('Invalid timing weight')
    if not np.isfinite(args.dynamic_width_residual_weight) or args.dynamic_width_residual_weight<0:parser.error('Invalid dynamic width residual weight')
    if args.phase_ceiling is not None and (not np.isfinite(args.phase_ceiling) or args.phase_ceiling<=0):parser.error('Invalid phase ceiling')
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
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--output',str(out),'--stage','fit','--job',job['name']],stdout=log,stderr=subprocess.STDOUT)
            running.append((job['name'],p,log))
        for name,p,log in list(running):
            if p.poll() is not None:
                log.close();running.remove((name,p,log));print(json.dumps(dict(job=name,exit_code=p.returncode)),flush=True)
                if p.returncode:raise RuntimeError(f'{name} failed; see worker.log')
        time.sleep(.25)
    evaluate(out)


if __name__=='__main__':main()
