"""Local multi-start fit of a four-zone model with a fixed-input cycle fold.

The added lower correction vanishes above its threshold. For each trial a
three-zone shooting system solves four crossing times and one coefficient to
enforce orbit closure and log(M)=0 at the target IP3. The trial is rejected if
the sampled fold itinerary is invalid or enters the added zone. This is local
shooting, not a proof of global bifurcation structure.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

from continue_physical_fixed import Orbit, REGIONS, BOUND, nondegeneracy
from fit_lower_fixed_fold import make_model, metrics, rhs

ROOT = Path(__file__).resolve().parents[1]
PIVOTS = {'AM': ('alpha', 4), 'AFM': ('alpha', 0), 'FM': ('alpha', 4)}
COEFFICIENTS = [(k, j) for k in ('alpha', 'beta') for j in range(5)]


def recompute(model):
    n, h, a, b = np.array(model['alpha']), np.array(model['beta']), model['a'], model['b']
    mats = [np.array([[-a*(n[1]+sum(n[3:3+j])), a],
                      [b*(h[1]+sum(h[3:3+j])), -b]]) for j in range(3)]
    model['trace'] = [float(np.trace(m)) for m in mats]
    model['determinant'] = [float(np.linalg.det(m)) for m in mats]
    return model


def fold_solution(base, changes, pivot, cut):
    d = copy.deepcopy(base)
    for (key, index), delta in changes.items():
        d[key][index] += delta
    pkey, pind = pivot
    seed = np.r_[base['fold_constraint']['crossing_times'], d[pkey][pind]]
    fold_I = base['fold_constraint']['I']
    scale = np.r_[np.maximum(seed[:4], .2), max(abs(seed[4]), .5)]
    def equations(z):
        d[pkey][pind] = float(z[4]); recompute(d)
        try:
            orb = Orbit(d)
            return np.r_[orb.match(z[:4], fold_I),
                         np.dot(np.asarray(d['trace'])[list(REGIONS)], z[:4])]
        except (ValueError, OverflowError, FloatingPointError):
            return np.full(5, 1e3)
    lo = np.r_[np.full(4, 1e-5), seed[4]-10*scale[4]]
    hi = np.r_[np.full(4, 120.), seed[4]+10*scale[4]]
    result = least_squares(equations, seed, bounds=(lo, hi), x_scale=scale,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=90)
    residual = equations(result.x)
    if not result.success or np.max(np.abs(residual)) > 2e-7:
        raise ValueError('fold equations unresolved')
    orb = Orbit(d)
    _, z, q, mats = orb.match(result.x[:4], fold_I, True)
    minimum = float('inf')
    speeds = []
    for i, (region, A) in enumerate(zip(REGIONS, mats)):
        y = np.array([z[i], q[i], 1.])
        speed = float((A @ y)[0]); speeds.append(speed)
        if (1 if i < 2 else -1)*speed < 1e-7:
            raise ValueError('fold non-transverse')
        samples = np.array([(expm(A*t) @ y)[0] for t in np.linspace(0, result.x[i], 81)])
        minimum = min(minimum, float(samples.min()))
        low = -np.inf if region == 0 else orb.cuts[region-1]
        high = np.inf if region == 2 else orb.cuts[region]
        if np.any(samples < low-1e-7) or np.any(samples > high+1e-7):
            raise ValueError('fold itinerary violated')
    # Margin exceeds the sampled/optimizer tolerance, but is not an interval proof.
    if minimum <= cut + 1e-4:
        raise ValueError('fold enters new region')
    return d, {'times': result.x[:4].tolist(), 'I': fold_I, 'Cmin_sampled': minimum,
               'matching_max': float(max(abs(residual[:4]))), 'logM': float(residual[4]),
               'speeds': speeds, 'pivot': [pkey, pind], 'pivot_value': float(result.x[4])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, choices=['AM', 'AFM', 'FM'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--arm', choices=['three', 'four'], default='four',
                   help='three disables the added lower correction for a matched refit control')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training', choices=['three', 'all'], default='all')
    p.add_argument('--max-nfev', type=int, default=30)
    p.add_argument('--duration', type=float, default=400.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = ROOT/'paper/data/physical_models/fixed_fold'/args.case/'parameters.json'
    base = json.loads(path.read_text()); source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = json.loads((ROOT/'paper/data/physical_models/fixed_fold/fold_audit.json').read_text())['results'][args.case.lower()]
    if audit['sha256'] != source_sha: raise RuntimeError('baseline hash mismatch')
    pivot = PIVOTS[args.case]
    free = [x for x in COEFFICIENTS if x != pivot]
    scales = np.array([.05*max(abs(base[k][j]), .3) for k,j in free])
    fold_min = audit['neutral']['Cmin']
    cut_hi = min(.95*base['physical_thresholds_C'][0], .9*fold_min)
    cut_lo = .25*cut_hi
    rows = base['training_cycles'] if args.training == 'all' else [base['training_cycles'][i] for i in (0,3,6)]
    baseline = [metrics(base, row, args.duration) for row in rows]
    if any(v is None for v in baseline):
        raise RuntimeError('baseline cycle unresolved; increase duration or inspect trajectories')
    np.random.seed(args.seed)
    x0 = np.zeros(len(free)+3); x0[-3] = .8
    if args.seed:
        x0[:len(free)] = np.random.default_rng(args.seed).normal(0., .08, len(free))
        x0[-2:] = np.random.default_rng(args.seed+5000).normal(0., .03, 2)
    lo = np.r_[np.full(len(free), -2.), 0., -2., -2.]
    hi = np.r_[np.full(len(free), 2.), 1., 2., 2.]
    def unpack(x):
        changes = {key: float(v*s) for key,v,s in zip(free, x, scales)}
        da, db = (float(x[-2]), float(x[-1])) if args.arm == 'four' else (0., 0.)
        return changes, float(cut_lo+(cut_hi-cut_lo)*x[-3]), da, db
    def candidate(x):
        changes, cut, da, db = unpack(x)
        three, fold = fold_solution(base, changes, pivot, cut)
        return three, make_model(three, cut, da, db), fold
    nres = 3*len(rows)+4+len(x0)
    calls = 0; best = {'loss': float('inf')}
    def residual(x):
        nonlocal calls, best
        calls += 1
        try:
            three, model, fold = candidate(x)
            values = [metrics(model, row, args.duration) for row in rows]
            if any(m is None for m in values): raise ValueError('training cycle unresolved')
            rr = []
            for row,m in zip(rows,values):
                rr += [(m['Cmin']-row['target_Cmin'])/.03,
                       (m['Cmax']-row['target_Cmax'])/.08,
                       (m['period']-row['target_period'])/5.]
            rr += list(np.asarray(rhs(model, 0., base['rest_target']))/[.03,.01])
            rr += [5*max(0., model['trace'][0]+.001), 5*max(0., .001-model['determinant'][0])]
            rr += list(.01*x)
            out = np.asarray(rr)
            loss = float(out@out)
            if loss < best['loss']:
                best = {'loss': loss, 'x': x.tolist(), 'fold': fold, 'calls': calls}
                (args.output/'checkpoint.json').write_text(json.dumps(best, indent=2)+'\n')
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            out = np.full(nres, 100.) + .001*np.resize(x,nres)
        if calls % 20 == 0: print(args.case, 'seed', args.seed, 'calls', calls, 'best', best['loss'], flush=True)
        return out
    # Fixed absolute finite differences are larger than integration noise.
    def jac(x):
        f = residual(x); cols=[]
        for i in range(len(x)):
            step = .01 if x[i]+.01 <= hi[i] else -.01
            y=x.copy(); y[i]+=step
            cols.append((residual(y)-f)/step)
        return np.column_stack(cols)
    print(args.case, 'seed', args.seed, 'baseline', baseline, flush=True)
    baseline_x=np.zeros_like(x0); baseline_x[-3]=.8
    baseline_loss=float(np.dot(r0:=residual(baseline_x),r0))
    result=least_squares(residual,x0,jac=jac,bounds=(lo,hi),max_nfev=args.max_nfev,
                         x_scale='jac',ftol=1e-4,xtol=1e-4,gtol=1e-4)
    if not np.isfinite(best['loss']): raise RuntimeError('no valid candidate found')
    x=np.array(best['x']); three, model, fold=candidate(x)
    # Independent integration and finite-difference fold diagnostics at the end.
    orb=Orbit(three)
    try:
        verified=orb.point(np.array(fold['times']),fold['I'],1e-9,require_physical=False)
        # The continuation helper's legacy C ceiling is not a physical bound.
        verified.pop('physical', None)
        fold['independent']=verified
        fold['nondegeneracy']=nondegeneracy(orb,np.r_[fold['times'],fold['I']])
    except ValueError as exc:
        fold['independent_error']=str(exc)
    predictions=[metrics(model,row,args.duration) for row in rows]
    holdout=[{'I':row['I'], 'baseline':metrics(base,row,args.duration),
              'candidate':metrics(model,row,args.duration),
              'target':{k:row[k] for k in ('target_Cmin','target_Cmax','target_period')}}
             for row in base['holdout_cycles']]
    model['lower_zone_refit']={'status':'exploratory; not validated', 'source_sha256':source_sha,
       'seed':args.seed, 'arm':args.arm, 'parameters_x':x.tolist(), 'free_coefficients':free,
       'coefficient_scales':scales.tolist(), 'fold':fold,
       'warning':'Other optimizer and cycle fields are inherited from the base and are stale.'}
    report={'case':args.case,'seed':args.seed,'arm':args.arm,'training_mode':args.training,'duration_s':args.duration,
       'training_I':[row['I'] for row in rows], 'baseline':baseline, 'candidate':predictions,
       'target':[{k:row[k] for k in ('target_Cmin','target_Cmax','target_period')} for row in rows],
       'holdout':holdout,'fold':fold,'baseline_squared_residual':baseline_loss,'best_squared_residual':best['loss'],
       'optimizer':{'success':bool(result.success),'message':str(result.message),'nfev':int(result.nfev), 'calls':calls},
       'rest_rhs':rhs(model,0.,base['rest_target']),'lower_trace':model['trace'][0],
       'lower_determinant':model['determinant'][0]}
    (args.output/'candidate.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(args.case,'seed',args.seed,'baseline',baseline_loss,'best',best['loss'],'status',result.message,flush=True)


if __name__=='__main__': main()
