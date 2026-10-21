"""Matched FM controls vs one fitted I hinge; explicit burst-end calibration."""
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
from scipy.signal import find_peaks
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.protocols import simulate
from pwl_nest.peak_timing import burst_end_residual
from overnight_coordinate_grid import atomic, prepare, source_rest, code_hash
from refine_server_candidates import TimingObjective, diagnostics, aggregate
from fit_strict import stats
import fit_positive_coordinates as fitting

SOURCE = ROOT / 'models/server_candidates/FM_identity__identity.json'
INPUT_BOUNDS = (.46, 1.12)


class BurstObjective(TimingObjective):
    def __call__(self, model, precise=False, parts=False):
        residual, groups = super().__call__(model, precise=precise, parts=True)
        for p in self.spec['train']:
            tr, truth, _ = self.refs[p['name']]
            if self.last_train_paths is None:
                extra = np.full(len(burst_end_residual(tr, truth, tr, truth)), 100.)
            else:
                t, x, _ = self.last_train_paths[p['name']]
                extra = burst_end_residual(t, x, tr, truth)
            residual = np.r_[residual, extra]
            groups[p['name'] + '_burst_end'] = float(extra @ extra)
        return (residual, groups) if parts else residual


def add_input_hinge(seed, knot):
    model = copy.deepcopy(seed)
    model.input_thresholds = np.array([knot])
    model.input_weights = np.zeros((2, 1))
    model.metadata.update(input_thresholds=[knot], extra_parameters=[0., 0.],
                          input_knot_locations_fitted=True, accepted=False)
    model.__post_init__()
    return model


def fresh_tests():
    specs = [dict(name=f'heldout_tau_{tau}', duration=360. if tau == 170. else 240.,
                  tau=tau, baseline=.45, jumps=[[20., .7]]) for tau in [9., 28., 57., 95., 170.]]
    rng = np.random.default_rng(20261024)
    times = 20. + np.r_[0., np.cumsum(rng.uniform(14., 24., 7))]
    amplitudes = rng.uniform(.2, .32, len(times))
    specs.append(dict(name='heldout_irregular', duration=300., tau=23., baseline=.45,
                      jumps=list(map(list, zip(times.tolist(), amplitudes.tolist())))))
    return specs


def setup(args, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    if args.reference_cache:
        shutil.copytree(Path(args.reference_cache) / 'FM', out / 'cache/FM')
    else:
        prepare(out, 'FM')
    cache = out / 'cache/FM'
    spec = json.loads((cache / 'data.json').read_text())
    original_train = copy.deepcopy(spec['train'])
    # Known failures are explicitly promoted; they cease to be held-out tests.
    promoted = [dict(name=f'promoted_tau_{tau}', duration=200. if tau == 65. else 220.,
                     tau=tau, baseline=.45, jumps=[[20., .7]]) for tau in [35., 65., 80.]]
    spec['train'] += promoted
    spec['previous_tests_not_blind'] = copy.deepcopy(spec['test'])
    spec['test'] = fresh_tests()
    spec['promoted_prior_failures'] = promoted
    rest = source_rest('FM', .45)
    for p in promoted + spec['test']:
        t, x, _ = simulate('FM', rest, **{k: v for k, v in p.items() if k != 'name'},
                           dt=.25 if p in promoted else .05, rtol=2e-11, atol=2e-13, max_step=.05)
        np.testing.assert_allclose(x[t < 20.], np.broadcast_to(rest, x[t < 20.].shape), atol=1e-8)
        np.savez_compressed(cache / f'{p["name"]}.npz', t=t, x=x)
    # Rebuild initialization points ONLY from training; old field.npz is ignored.
    points = []
    for p in spec['fixed'] + spec['train']:
        with np.load(cache / f'{p["name"]}.npz') as d:
            points.append(d['x'][::max(1, len(d['t']) // 200)])
    np.savez_compressed(cache / 'field.npz', points=np.vstack(points))
    spec['field_initialization_protocols'] = [p['name'] for p in spec['fixed'] + spec['train']]
    spec['heldout_used_for_field_initialization'] = False
    atomic(cache / 'data.json', spec)
    last_inputs = []
    for p in original_train[:2]:
        with np.load(cache / f'{p["name"]}.npz') as d:
            peaks = find_peaks(d['x'][:, 0], prominence=.02)[0]
            last_inputs.append(float(d['x'][peaks[-1], 2]))
    knots = [min(last_inputs), float(np.mean(last_inputs)), max(last_inputs)]
    jobs = []
    for i, knot in enumerate(knots):
        for family in ['control', 'one_I_knot']:
            jobs.append(dict(name=f'{family}_{i}', family=family, seed=args.seed + i,
                             initial_knot=knot if family == 'one_I_knot' else None))
    config = dict(version=1, budget=args.budget, jobs=jobs, workers=args.workers, input_bounds=INPUT_BOUNDS,
        source_model=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes() + code_hash().encode()).hexdigest(),
        python=sys.version, numpy=np.__version__, scipy=__import__('scipy').__version__,
        field_initialization='fixed + dynamic TRAIN ONLY; no test trajectory points',
        selection='physical + all training peak counts + known static regimes; then independent training loss',
        promoted_prior_failures=promoted, heldout_protocols=spec['test'])
    atomic(out / 'config.json', config)
    return config


def training_checks(objective):
    rows = []
    paths = {**objective.last_static_replay, **objective.last_train_paths}
    physical = all((x[:, 0] > 0).all() and (x[:, 0] < C_MAX).all()
                   and (x[:, 1] > 0).all() and (x[:, 1] < 1).all() for _, x, _ in paths.values())
    fixed_ok = True
    for p in objective.spec['fixed']:
        observed = stats(*paths[p['name']][:2])['observed_status']
        expected = objective.static.targets[p['name']]['observed_status']
        matches = expected == 'unresolved' or observed == expected
        rows.append(dict(protocol=p['name'], kind='fixed', expected=expected, observed=observed, matches=matches))
        fixed_ok = fixed_ok and matches
    count_ok = True
    for p in objective.spec['train']:
        def count(trace):
            t, x, _ = trace
            peaks = find_peaks(x[:, 0], prominence=.02)[0]
            return int(np.sum(t[peaks] >= 20.))
        expected, observed = count(objective.refs[p['name']]), count(paths[p['name']])
        rows.append(dict(protocol=p['name'], kind='dynamic', expected=expected, observed=observed, matches=expected == observed))
        count_ok = count_ok and expected == observed
    return dict(physical=bool(physical), fixed_regimes_match=bool(fixed_ok),
                all_dynamic_peak_counts_match=bool(count_ok), eligible=bool(physical and fixed_ok and count_ok), protocols=rows)


def fit_job(out, name):
    config = json.loads((out / 'config.json').read_text())
    job = next(j for j in config['jobs'] if j['name'] == name)
    directory = out / 'trials' / name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        raise ValueError('Trial already completed')
    seed = CalciumTransformPWA.load(SOURCE)
    if job['initial_knot'] is not None:
        seed = add_input_hinge(seed, job['initial_knot'])
    objective = BurstObjective(out, 'FM', seed)
    with np.load(out / 'cache/FM/field.npz') as d:
        points = d['points']
    data = ([], {}, points, {}, objective.spec['C_lower'], objective.spec['C_upper'])
    fitting.OUT, fitting.MODELS = directory / 'fit', directory / 'models'
    fitting.OUT.mkdir(exist_ok=True);fitting.MODELS.mkdir(exist_ok=True)
    started = time.monotonic()
    model = fitting.fit('FM', 'input_threshold', seed, data, objective, config['budget'],
        rng_seed=job['seed'], checkpoint_path=directory / 'unverified_checkpoint.json',
        input_threshold_bounds=INPUT_BOUNDS if job['initial_knot'] is not None else None)
    model.metadata.update(accepted=False, source_server_candidate=str(SOURCE.relative_to(ROOT)),
        objective='absolute peak times + last peak + excess postburst calcium + original criterion',
        model_family=job['family'], actual_region_count=model.latent.region_count * (len(model.input_thresholds) + 1))
    model.save(directory / 'candidate.json')
    r, parts = objective(model, precise=True, parts=True)
    result = dict(**job, status='verified', verified_train_loss=float(r @ r), train_parts=parts,
                  checks=training_checks(objective), model=str((directory / 'candidate.json').relative_to(out)),
                  input_thresholds=model.input_thresholds.tolist(), parameters=model.metadata['active_fit_parameters'],
                  regions=model.metadata['actual_region_count'], seconds=time.monotonic() - started)
    atomic(directory / 'result.json', result)
    print(json.dumps(result), flush=True)


def evaluate(out):
    config = json.loads((out / 'config.json').read_text())
    baseline = CalciumTransformPWA.load(SOURCE)
    objective = BurstObjective(out, 'FM', baseline)
    r, parts = objective(baseline, precise=True, parts=True)
    base = dict(name='original', family='original', model=str(SOURCE), verified_train_loss=float(r @ r),
                train_parts=parts, checks=training_checks(objective), input_thresholds=[], parameters=29, regions=8)
    verified = [json.loads(p.read_text()) for p in (out / 'trials').glob('*/result.json')]
    selected = [base]
    for family in ['control', 'one_I_knot']:
        rows = [r for r in verified if r['family'] == family]
        if not rows:
            continue
        eligible = [r for r in rows if r['checks']['eligible']]
        row = min(eligible or rows, key=lambda r: r['verified_train_loss'])
        row['selection_scope'] = 'eligible training candidate' if eligible else 'diagnostic only: no eligible candidate'
        selected.append(row)
    # Only NOW evaluate held-out references; no held-out metric selects a model.
    for row in selected:
        model = baseline if row['family'] == 'original' else CalciumTransformPWA.load(out / row['model'])
        records = diagnostics(model, objective, objective.spec['test'], objective.refs, out / 'diagnostics' / row['family'])
        row['heldout'] = dict(aggregate=aggregate(records), protocols=records)
        if row['family'] != 'original':
            directory = out / 'selected';directory.mkdir(exist_ok=True)
            model.save(directory / f'{row["family"]}.json')
        atomic(out / 'summary.json', dict(config=config, selected=selected, all_trials=verified))
    return selected


def run(args):
    out = Path(args.output).resolve()
    if args.stage == 'fit':
        fit_job(out, args.job)
        return
    if args.stage == 'evaluate':
        evaluate(out)
        return
    config = setup(args, out)
    if args.stage == 'prepare':
        return
    waiting = list(config['jobs']);running = {}
    while waiting or running:
        while waiting and len(running) < args.workers:
            job = waiting.pop(0);directory = out / 'trials' / job['name'];directory.mkdir(parents=True)
            log = (directory / 'worker.log').open('w')
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--output', str(out),
                                     '--stage', 'fit', '--job', job['name']], stdout=log, stderr=subprocess.STDOUT)
            running[job['name']] = (proc, log, time.monotonic())
        for name, (proc, log, started) in list(running.items()):
            if time.monotonic() - started > 900.:
                proc.terminate()
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            if proc.poll() is not None:
                log.close();print(json.dumps(dict(job=name, exit_code=proc.returncode)), flush=True)
                del running[name]
        time.sleep(.25)
    evaluate(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['all', 'prepare', 'fit', 'evaluate'], default='all')
    parser.add_argument('--job')
    parser.add_argument('--reference-cache')
    parser.add_argument('--budget', type=int, default=1500)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261024)
    args = parser.parse_args()
    if args.workers < 1 or args.budget < 1 or (args.stage == 'fit' and not args.job):
        parser.error('Invalid worker count, budget, or job')
    run(args)
