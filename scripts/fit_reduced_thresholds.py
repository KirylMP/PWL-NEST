"""Refit fewer C thresholds using training trajectories only; verify with DOP853."""
import argparse
import copy
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import least_squares,minimize

from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.reduced_thresholds import ReducedFamily
from overnight_coordinate_grid import prepare,atomic
from refine_server_candidates import rebuild_training_field
from am_flow_objective import FlowRestObjective

SOURCES=dict(AM_cube='models/am_baseline_refit/AM_sqrt_tails__cube.json',
             AM_log='models/am_baseline_refit/AM_sqrt_tails__log.json',
             FM='models/phase_shape_candidates/FM_width_soft_candidate.json')


def run(args):
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory')
    out.mkdir(parents=True)
    case='AM' if args.source.startswith('AM') else 'FM'
    source_path=ROOT/SOURCES[args.source];source=CalciumTransformPWA.load(source_path)
    cache_source=(ROOT/'results/am_baseline_refit/cube/cache/AM' if case=='AM' else
                  ROOT/'results/server_timing_refinement/cache/FM')
    if cache_source.exists():shutil.copytree(cache_source,out/'cache'/case)
    else:prepare(out,case)
    rebuild_training_field(out/'cache'/case)
    objective=FlowRestObjective(out,case,source,native_maxdt=.02,spectral_anchors=False)
    config=dict(source=str(source_path.relative_to(ROOT)),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        C_thresholds=args.thresholds,keep_input_thresholds=args.keep_input,seed=args.seed,
        budget_per_stage=args.budget,optimized_starts=args.starts,regions=(args.thresholds+1)*(len(source.input_thresholds)+1 if args.keep_input else 1),
        selection='independently verified training residual; no diagnostics',
        loss='RestObjective physical regimes, own rest and event features; spectral boundary anchors omitted for BOTH full and reduced models',
        constraints='exact stable LR background equilibrium, continuous field, ordered C thresholds, positive u-v and negative v-v coupling; original spectral sequence not imposed')
    atomic(out/'config.json',config)
    reference,parts=objective(source,precise=True,parts=True)
    atomic(out/'full_model_score.json',dict(loss=float(reference@reference),parts=parts))
    with np.load(out/'cache'/case/'field.npz') as f:points=f['points']
    # The field initializer contains train samples only, with bounded size.
    points=points[::max(1,len(points)//1800)]
    families=[];screen=[]
    thresholds=np.asarray(source.metadata['physical_thresholds_C'])
    for subset in combinations(range(len(thresholds)),args.thresholds):
        for target in ['incumbent','Li_Rinzel']:
            try:
                family=ReducedFamily(source,thresholds[list(subset)],objective.spec['rest'],points,
                    objective.spec['C_lower'],objective.spec['C_upper'],target=target,keep_input=args.keep_input)
                z=np.zeros(family.size);model=family.make(z)
                r=objective(model);score=float(r@r)
                record=dict(indices=list(subset),target=target,helper_loss=score,parameters=family.size,
                    status='screened',family_index=len(families))
                families.append(family)
            except (ValueError,RuntimeError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:
                record=dict(indices=list(subset),target=target,status='failed',error=str(e))
            screen.append(record)
    atomic(out/'screen.json',screen)
    ranked=sorted([r for r in screen if r['status']=='screened'],key=lambda r:r['helper_loss'])
    if not ranked:raise RuntimeError('No viable reduced initializer')
    # All candidates selected here use training scores. Held-out paths are not evaluated.
    candidates=[]
    def verify(model,label):
        path=out/'candidates'/f'{label}.json';path.parent.mkdir(exist_ok=True)
        model.save(path)
        try:
            r,groups=objective(model,precise=True,parts=True)
            record=dict(label=label,status='verified',train_loss=float(r@r),parts=groups,
                        model=str(path.relative_to(out)))
        except (ValueError,RuntimeError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:
            record=dict(label=label,status='failed',error=str(e),model=str(path.relative_to(out)))
        candidates.append(record);atomic(out/'verified_candidates.json',candidates)
        print(json.dumps(record),flush=True)
    for rank,entry in enumerate(ranked[:args.starts]):
        family=families[entry['family_index']]
        directory=out/'optimization'/f'start_{rank}';directory.mkdir(parents=True)
        initial=np.zeros(family.size)
        verify(family.make(initial),f'seed_{rank}')
        baseline=objective(family.make(initial));length=len(baseline)
        count=0;limit=args.budget;best=float(baseline@baseline);bestz=initial.copy();history=[]
        class Limit(Exception):pass
        def residual(z):
            nonlocal count,best,bestz
            if count>=limit:raise Limit()
            count+=1
            try:r=objective(family.make(z))
            except (ValueError,RuntimeError,FloatingPointError,OverflowError,np.linalg.LinAlgError):r=np.full(length,100.)
            score=float(r@r)
            if score<best:
                best=score;bestz=z.copy()
                if not history or count-history[-1]['evaluation']>=100:
                    history.append(dict(evaluation=count,loss=best))
                    family.make(bestz).save(directory/'unverified_checkpoint.json')
                    print(json.dumps(dict(source=args.source,start=rank,evaluation=count,helper_loss=best)),flush=True)
            return r
        started=time.monotonic()
        try:
            least_squares(residual,initial,bounds=(family.lower,family.upper),diff_step=1e-4,
                          x_scale='jac',max_nfev=100,ftol=1e-6,xtol=1e-6)
        except Limit:pass
        limit=count+args.budget
        radius=.04
        lo=np.maximum(family.lower,bestz-radius);hi=np.minimum(family.upper,bestz+radius)
        try:
            minimize(lambda z:float(np.dot(r:=residual(z),r)),bestz,method='Powell',
                bounds=list(zip(lo,hi)),options=dict(maxfev=args.budget,xtol=1e-5,ftol=1e-6))
        except Limit:pass
        atomic(directory/'fit.json',dict(initializer=entry,evaluations=count,helper_loss=best,
            history=history,seconds=time.monotonic()-started,vector=bestz.tolist()))
        verify(family.make(bestz),f'fit_{rank}')
    valid=[r for r in candidates if r['status']=='verified']
    if not valid:raise RuntimeError('No reduced candidate passed independent training replay')
    winner=min(valid,key=lambda r:r['train_loss'])
    selected=CalciumTransformPWA.load(out/winner['model'])
    selected.metadata.update(verified_current_train_loss=winner['train_loss'],
        selected_by=config['selection'],source_model=config['source'],experiment='threshold_reduction',accepted=False)
    selected.save(out/'selected.json')
    atomic(out/'selection.json',dict(selected=winner,full_model_loss=float(reference@reference),
        C_threshold_count=len(selected.latent.thresholds),input_threshold_count=len(selected.input_thresholds),
        parameter_count=selected.metadata['active_fit_parameters'],candidates=candidates))
    print(json.dumps(dict(source=args.source,selected_loss=winner['train_loss'],full_loss=float(reference@reference))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=list(SOURCES),required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--thresholds',type=int,required=True)
    p.add_argument('--keep-input',action='store_true')
    p.add_argument('--budget',type=int,default=1000)
    p.add_argument('--starts',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261027)
    a=p.parse_args()
    if a.thresholds<1 or a.budget<1 or a.starts<1:p.error('Positive thresholds/budget/starts required')
    run(a)
