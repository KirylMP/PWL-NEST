"""Matched three-dimensional global warm starts, then full joint refinement."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import differential_evolution
from pwl_nest.calcium_coordinates import CalciumTransformPWA, decode_coordinates
from fit_fm_input_threshold import BurstObjective, INPUT_BOUNDS, training_checks
from overnight_coordinate_grid import atomic, code_hash
import fit_positive_coordinates as fitting


def run(args):
    out = Path(args.output).resolve()
    directory = out / 'trials' / f'global_{args.family}'
    directory.mkdir(parents=True)
    previous = [json.loads(p.read_text()) for p in (out / 'trials').glob('*/result.json')
                if not p.parent.name.startswith('global_')]
    rows = [r for r in previous if r['family'] == args.family]
    eligible = [r for r in rows if r['checks']['eligible']]
    parent = min(eligible or rows, key=lambda r: r['verified_train_loss'])
    base = CalciumTransformPWA.load(out / parent['model'])
    objective = BurstObjective(out, 'FM', base)
    z = np.array(base.metadata['parameters'])
    n = base.latent.region_count
    if args.family == 'one_I_knot':
        start = np.r_[base.input_thresholds, base.metadata['extra_parameters']]
        a, lam = start[1:]
        bounds = [INPUT_BOUNDS, (max(-2., a - .08), min(2., a + .08)),
                  (max(-np.log(4), lam - .7), min(np.log(4), lam + .7))]

        def make(p):
            eta, a, lam = p
            B, D = base.latent.matrix[:, 2]
            qb = base.latent.spectral_coefficients()[3]
            weights = np.array([[a], [(qb * np.expm1(lam) + D * a) / B]])
            latent = copy.deepcopy(base.latent)
            latent.metadata.update(extra_parameters=[float(a), float(lam)], input_thresholds=[float(eta)])
            return CalciumTransformPWA(latent, base.coordinate, base.h_reference, base.h_kind, [eta], weights)
    else:
        indices = np.array([2 * n, 2 * n + 1, 2 * n + 4])
        start = z[indices]
        bounds = list(zip(np.maximum(start - [.03, .3, .08], [-.5, -40., -10.]),
                          np.minimum(start + [.03, .3, .08], [.5, 8., 10.])))

        def make(p):
            parameters = z.copy();parameters[indices] = p
            metadata = copy.deepcopy(base.metadata);metadata['parameters'] = parameters.tolist()
            return decode_coordinates(base.case, base.latent.thresholds, parameters,
                base.metadata['mu_sign'], base.metadata['k_sign'], metadata,
                base.coordinate, base.h_reference, base.h_kind)
    atomic(directory / 'search_config.json', dict(parent=parent['model'],
        parent_sha256=hashlib.sha256((out / parent['model']).read_bytes()).hexdigest(),
        family=args.family, bounds=np.array(bounds).tolist(), start=start.tolist(),
        global_maxiter=args.iterations, population_size_factor=8, joint_budget=args.budget, seed=args.seed,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes() + code_hash().encode()).hexdigest(),
        selection='training only; equal DE and joint budgets for both families'))
    calls = 0;best_native = float('inf');history = []

    def score(p):
        nonlocal calls, best_native
        calls += 1
        try:
            model = make(p);r = objective(model);loss = float(r @ r)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return 1e12
        if loss < best_native:
            best_native = loss
            history.append(dict(evaluation=calls, loss=loss, parameters=np.asarray(p).tolist()))
            model.save(directory / 'global_checkpoint.json')
            print(json.dumps(history[-1]), flush=True)
        return loss
    started = time.monotonic()
    result = differential_evolution(score, bounds, x0=start, popsize=8, maxiter=args.iterations,
        rng=np.random.default_rng(args.seed), polish=False, tol=1e-7)
    warm = make(result.x)
    warm.save(directory / 'global_seed.json')
    with np.load(out / 'cache/FM/field.npz') as d:
        points = d['points']
    data = ([], {}, points, {}, objective.spec['C_lower'], objective.spec['C_upper'])
    fitting.OUT, fitting.MODELS = directory / 'fit', directory / 'models'
    fitting.OUT.mkdir();fitting.MODELS.mkdir()
    joint = fitting.fit('FM', 'global_input', copy.deepcopy(warm), data, objective, args.budget,
        rng_seed=args.seed, checkpoint_path=directory / 'unverified_checkpoint.json',
        input_threshold_bounds=INPUT_BOUNDS if args.family == 'one_I_knot' else None)
    joint.save(directory / 'joint_candidate.json')
    candidates = [(copy.deepcopy(parent), base, 'parent')]
    for label, model in [('global_seed', warm), ('joint', joint)]:
        r, parts = objective(model, precise=True, parts=True)
        candidates.append((dict(verified_train_loss=float(r @ r), train_parts=parts,
                                checks=training_checks(objective)), model, label))
    eligible = [r for r in candidates if r[0]['checks']['eligible']]
    record, model, chosen = min(eligible or candidates, key=lambda r: r[0]['verified_train_loss'])
    model.metadata.update(accepted=False, physical_field_PWA=True,
        objective='absolute peak times + burst end + original criterion',
        selection='training only; I-threshold global warm start + joint refinement',
        input_knot_locations_fitted=args.family == 'one_I_knot')
    model.save(directory / 'candidate.json')
    final = dict(name=directory.name, family=args.family, status='verified',
        model=str((directory / 'candidate.json').relative_to(out)),
        input_thresholds=model.input_thresholds.tolist(), parameters=32 if args.family == 'one_I_knot' else 29,
        regions=16 if args.family == 'one_I_knot' else 8, verified_train_loss=record['verified_train_loss'],
        train_parts=record['train_parts'], checks=record['checks'], chosen=chosen,
        global_evaluations=calls, joint_budget=args.budget, seconds=time.monotonic() - started)
    atomic(directory / 'result.json', final)
    atomic(directory / 'global_history.json', history)
    atomic(directory / 'alternatives.json', [dict(label=label, **rec) for rec, _, label in candidates])
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--family', choices=['control', 'one_I_knot'], required=True)
    parser.add_argument('--budget', type=int, default=1500)
    parser.add_argument('--iterations', type=int, default=25)
    parser.add_argument('--seed', type=int, default=20261027)
    args = parser.parse_args()
    if args.budget < 1 or args.iterations < 1:
        parser.error('Positive fitting budgets are required')
    run(args)
