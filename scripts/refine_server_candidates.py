"""Targeted continuation of frozen server candidates with optional peak timing.

Old grid results remain frozen. Selection uses independently computed training
loss only. Both previous tests and genuinely new protocols are diagnostics.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA, simulate_coordinates
from pwl_nest.peak_timing import absolute_peak_time_residual, interpolated_peak_times
from overnight_coordinate_grid import RestObjective, atomic, prepare, source_rest, code_hash
from check_rest_and_phase import metrics
from pwl_nest.protocols import simulate
import fit_positive_coordinates as fitting


class TimingObjective(RestObjective):
    def __init__(self, out, case, seed, sigma_s=1.):
        super().__init__(out, case, seed)
        self.sigma_s = sigma_s
        self.last_train_paths = None

    def paths(self, model, protocols, precise=False):
        result = super().paths(model, protocols, precise)
        if protocols is self.spec['train']:
            self.last_train_paths = result[2]
        return result

    def __call__(self, model, precise=False, parts=False):
        self.last_train_paths = None
        residual, groups = super().__call__(model, precise=precise, parts=True)
        for p in self.spec['train']:
            tr, truth, _ = self.refs[p['name']]
            if self.last_train_paths is None:
                n = len(interpolated_peak_times(tr, truth))
                extra = np.full(n, 100.)
            else:
                t, x, _ = self.last_train_paths[p['name']]
                extra = absolute_peak_time_residual(t, x, tr, truth, sigma_s=self.sigma_s)
            residual = np.r_[residual, extra]
            groups[p['name'] + '_absolute_peak_times'] = float(extra @ extra)
        return (residual, groups) if parts else residual


def fresh_protocols(case):
    baseline = .3 if case == 'AM' else .45
    pulse = .35 if case == 'AM' else .7
    specs = [dict(name=f'new_tau_{tau}', duration=300. if tau == 140. else 220.,
                  tau=tau, baseline=baseline, jumps=[[20., pulse]]) for tau in [13., 35., 80., 140.]]
    rng = np.random.default_rng(20261023)
    times = 20. + np.r_[0., np.cumsum(rng.uniform(14., 23., 6))]
    amplitudes = rng.uniform(.12, .18, len(times)) if case == 'AM' else rng.uniform(.2, .3, len(times))
    specs.append(dict(name='new_irregular', duration=260., tau=17., baseline=baseline,
                      jumps=list(map(list, zip(times.tolist(), amplitudes.tolist())))))
    return specs


def diagnostics(model, objective, specs, references, directory):
    records = []
    directory.mkdir(parents=True, exist_ok=True)
    for p in specs:
        try:
            _, _, paths = objective.paths(model, [p], precise=True)
            t, x, _ = paths[p['name']]
            tr, truth, _ = references[p['name']]
            np.testing.assert_allclose(t, tr, atol=1e-10, rtol=0)
            record = dict(name=p['name'], completed=True, **metrics(t, truth, x, 20.))
            record['interpolated_peak_times_LR'] = interpolated_peak_times(t, truth).tolist()
            record['interpolated_peak_times_candidate'] = interpolated_peak_times(t, x).tolist()
            np.savez_compressed(directory / f'{p["name"]}.npz', t=t, lr=truth, candidate=x)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            record = dict(name=p['name'], completed=False, error=str(exc))
        records.append(record)
    return records


def aggregate(records):
    complete = all(r['completed'] for r in records)
    return dict(C_RMSE=float(np.mean([r['raw_full_RMSE'][0] for r in records])) if complete else None,
                physical=all(r.get('physical', False) for r in records),
                peak_count_matches=sum(r.get('peaks', {}).get('counts', [0, -1])[0]
                                       == r.get('peaks', {}).get('counts', [0, -1])[1] for r in records),
                protocol_count=len(records))


def rebuild_training_field(cache):
    """Discard cached field seeds that may contain held-out trajectory points."""
    cache = Path(cache)
    spec = json.loads((cache / 'data.json').read_text())
    training = spec['fixed'] + spec['train']
    points = []
    for p in training:
        with np.load(cache / f'{p["name"]}.npz') as data:
            points.append(data['x'][::max(1, len(data['t']) // 200)])
    np.savez_compressed(cache / 'field.npz', points=np.vstack(points))
    spec.update(field_initialization_protocols=[p['name'] for p in training],
                heldout_used_for_field_initialization=False)
    atomic(cache / 'data.json', spec)


def run(args):
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()) and not args.replay_trials:
        raise ValueError('Use a new output directory; frozen runs are never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / 'models/server_candidates' / f'{name}.json' for name in args.models]
    seeds = [CalciumTransformPWA.load(p) for p in sources]
    base_objective = getattr(args, 'base_objective', False)
    objective_label = ('original RestObjective: fixed regimes, spectral anchors, own rest, event features and weak waveform'
                       if base_objective else 'original rest/event criterion + every absolute peak time / sigma_s; quadratic peak interpolation')
    config = dict(models=[str(p.relative_to(ROOT)) for p in sources],
        source_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in sources],
        budget=args.budget, restarts=args.restarts, sigma_s=args.timing_sigma, seed=args.seed,
        objective=objective_label, base_objective=base_objective,
        native_maxdt=args.native_maxdt,powell_radius=args.powell_radius,
        flow_helper=args.flow_helper,
        selection='all training independently recomputed with DOP853; tests excluded',
        field_initialization='frozen cache; no optimization' if args.replay_trials else 'fixed and dynamic training paths only',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes() + code_hash().encode()).hexdigest(),
        python=sys.version, numpy=np.__version__, scipy=__import__('scipy').__version__)
    if args.replay_trials:
        previous_config = json.loads((out / 'config.json').read_text())
        for key in ['models', 'source_sha256', 'budget', 'restarts', 'sigma_s', 'seed', 'objective']:
            if previous_config[key] != config[key]:
                raise ValueError(f'Saved trial configuration differs: {key}')
        for key,default in dict(base_objective=False,native_maxdt=.005,powell_radius=None,flow_helper=False).items():
            if previous_config.get(key,default)!=config[key]:
                raise ValueError(f'Saved trial numerical configuration differs: {key}')
        atomic(out / 'replay_config.json', config)
    else:
        atomic(out / 'config.json', config)
    for case in dict.fromkeys(model.case for model in seeds):
        if not (out / 'cache' / case / 'data.json').exists():
            if args.reference_cache:
                shutil.copytree(Path(args.reference_cache) / case, out / 'cache' / case)
            else:
                prepare(out, case)
        if not args.replay_trials:
            rebuild_training_field(out / 'cache' / case)
    results = []
    for name, source, original in zip(args.models, sources, seeds):
        case = original.case
        if args.flow_helper:
            from am_flow_objective import FlowRestObjective
            objective=FlowRestObjective(out,case,original,native_maxdt=args.native_maxdt)
        else:
            objective = RestObjective(out, case, original, native_maxdt=args.native_maxdt) if base_objective else TimingObjective(out, case, original, args.timing_sigma)
        objective.native_maxdt=args.native_maxdt
        r, groups = objective(original, precise=True, parts=True)
        baseline_loss = float(r @ r)
        best_loss = baseline_loss
        incumbent = copy.deepcopy(original)
        selected_trial = None
        item = dict(name=name, baseline_train_loss=baseline_loss, baseline_train_parts=groups, trials=[])
        print(json.dumps(dict(model=name, baseline_train_loss=baseline_loss, parts=groups)), flush=True)
        with np.load(out / 'cache' / case / 'field.npz') as data:
            points = data['points']
        data = ([], {}, points, {}, objective.spec['C_lower'], objective.spec['C_upper'])
        for restart in range(args.restarts):
            directory = out / 'trials' / name / f'r{restart:04d}'
            fitting.OUT = directory / 'fit'
            fitting.MODELS = directory / 'models'
            if not args.replay_trials:
                fitting.OUT.mkdir(parents=True)
                fitting.MODELS.mkdir()
            started = time.monotonic()
            try:
                if args.replay_trials:
                    model = CalciumTransformPWA.load(directory / 'candidate.json')
                else:
                    model = fitting.fit(case, 'baseline' if base_objective else 'timing', copy.deepcopy(incumbent), data, objective, args.budget,
                        rng_seed=args.seed + restart * 100000,
                        checkpoint_path=directory / 'unverified_checkpoint.json',powell_radius=args.powell_radius)
                    model.metadata.update(objective=objective_label,
                        source_server_candidate=str(source.relative_to(ROOT)),
                        accepted=False, selected_by='independent DOP853 training loss only',
                        current_validation='see the result.json alongside this model; inherited legacy losses are historical')
                    if not base_objective:
                        model.metadata['absolute_peak_timing_sigma_s']=args.timing_sigma
                    else:
                        model.metadata.pop('absolute_peak_timing_sigma_s',None)
                    model.save(directory / 'candidate.json')
                residual, parts = objective(model, precise=True, parts=True)
                loss = float(residual @ residual)
                record = dict(restart=restart, status='verified', verified_train_loss=loss,
                              train_parts=parts, seconds=time.monotonic() - started,
                              model=str((directory / 'candidate.json').relative_to(out)))
                if loss < best_loss:
                    incumbent, best_loss, selected_trial = model, loss, record['model']
            except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                record = dict(restart=restart, status='failed', error=str(exc), seconds=time.monotonic() - started)
            atomic(directory / ('replay_result.json' if args.replay_trials else 'result.json'), record)
            item['trials'].append(record)
            print(json.dumps(dict(candidate_name=name, **record)), flush=True)
        selected = out / 'selected' / f'{name}.json'
        selected.parent.mkdir(exist_ok=True)
        incumbent.save(selected)
        fresh = fresh_protocols(case)
        refs = dict(objective.refs)
        rest = source_rest(case, objective.spec['rest'][2])
        for p in fresh:
            refs[p['name']] = simulate(case, rest, **{k: v for k, v in p.items() if k != 'name'},
                                       dt=.05, rtol=2e-11, atol=2e-13, max_step=.05)
        item.update(selected_train_loss=best_loss, selected_trial=selected_trial,
                    selected_model=str(selected.relative_to(out)), old_protocols=objective.spec['test'], new_protocols=fresh)
        # Evaluate after training selection; these measurements never choose a model.
        for label, model in [('baseline', original), ('selected', incumbent)]:
            for split, specs in [('previous_tests', objective.spec['test']), ('new_tests', fresh)]:
                records = diagnostics(model, objective, specs, refs, out / 'diagnostics' / name / label / split)
                item[f'{label}_{split}'] = dict(aggregate=aggregate(records), protocols=records)
        results.append(item)
        atomic(out / 'summary.json', results)
        print(json.dumps(dict(model=name, selected_train_loss=best_loss,
            previous_tests=item['selected_previous_tests']['aggregate'],
            new_tests=item['selected_new_tests']['aggregate'])), flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--models', nargs='+', default=['FM_identity__identity'])
    parser.add_argument('--budget', type=int, default=1500)
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--timing-sigma', type=float, default=1.)
    parser.add_argument('--base-objective', action='store_true',
                        help='Use the original RestObjective unchanged, without extra absolute peak-time residuals')
    parser.add_argument('--native-maxdt',type=float,default=.005,
                        help='Native dynamic integration step in seconds; does not change the loss definition')
    parser.add_argument('--powell-radius',type=float,
                        help='Optional local Powell box relative to max(1, abs(parameter)); global feasibility bounds stay unchanged')
    parser.add_argument('--flow-helper',action='store_true',
                        help='Use cached affine flows and full fixed-duration trajectories; requires --base-objective')
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--reference-cache', help='Optional cache/ from the frozen first grid run')
    parser.add_argument('--replay-trials', action='store_true',
                        help='Recompute saved trial training losses and finish diagnostics; no new fitting')
    args = parser.parse_args()
    if (args.budget < 1 or args.restarts < 0 or not np.isfinite(args.timing_sigma) or args.timing_sigma <= 0
        or not np.isfinite(args.native_maxdt) or args.native_maxdt<=0
        or (args.powell_radius is not None and (not np.isfinite(args.powell_radius) or args.powell_radius<=0))):
        parser.error('Invalid optimization limits or timing sigma')
    if args.flow_helper and not args.base_objective:parser.error('--flow-helper requires --base-objective')
    run(args)
