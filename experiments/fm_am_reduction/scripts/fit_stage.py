#!/usr/bin/env python3
"""Fit the protocol's pre-cycle objective for a planned FM/AM run.

This implements spectral threshold partitioning and the mixed nullcline,
realized-spectrum, and equilibrium fit (protocol sections 4, 5.1–5.3). Cycle
return losses and independent structural validation are deliberately separate
future stages; outputs are always marked preliminary.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment, brentq

from plan import PACKAGE, ROOT, read_json


def phi(c: np.ndarray, eta: np.ndarray, c_ref: float) -> np.ndarray:
    p, e, logn, k = eta
    n = float(np.exp(logn))
    t = np.asarray(c, dtype=float)/c_ref
    wp, we = np.exp(p), np.exp(e)
    denom = wp+we+1.
    wp, we, wl = wp/denom, we/denom, 1./denom
    if abs(k) < 1e-5:
        v=t-1.
        exp_part=v + .5*k*v**2 + (k*k/6)*v**3 + (k**3/24)*v**4
    else:
        exp_part = np.expm1(np.clip(k*(t-1), -700, 700))/k
    return wp*np.expm1(np.clip(n*np.log(t), -700, 700))/n + we*exp_part + wl*np.log(t)


def hinge_design(z: np.ndarray, ip3: np.ndarray, thresholds_z: np.ndarray) -> np.ndarray:
    z, ip3 = np.broadcast_arrays(np.asarray(z), np.asarray(ip3))
    return np.column_stack([np.ones(z.size), z.ravel(), ip3.ravel(),
                            *[np.maximum(z.ravel()-t, 0.) for t in thresholds_z]])


def quadrature_weights(nodes: np.ndarray) -> np.ndarray:
    """Trapezoid weights in the physical sampling coordinate."""
    if len(nodes) == 1:
        return np.ones(1)
    gaps = np.diff(nodes)
    return np.r_[gaps[0]/2, (gaps[:-1]+gaps[1:])/2, gaps[-1]/2]


def partition_thresholds(equilibria: list[dict], m: int, grid_size: int,
                         i_range: tuple[float, float], c_bounds: tuple[float, float],
                         scales=(1., .15)) -> tuple[np.ndarray, dict]:
    samples = [q for q in equilibria if i_range[0] <= q["I"] <= i_range[1]
               and c_bounds[0] < q["C"] < c_bounds[1]]
    if len(samples) < m+1:
        raise ValueError("Too few spectral samples for requested region count")
    i_levels = sorted({float(q["I"]) for q in samples})
    level_weights = quadrature_weights(np.array(i_levels))
    sample_weight = {id(q): level_weights[i_levels.index(q["I"])] for q in samples}
    samples.sort(key=lambda q: q["C"])
    c = np.array([q["C"] for q in samples])
    t = np.array([q["trace"] for q in samples])
    d = np.array([q["determinant"] for q in samples])
    w = np.array([sample_weight[id(q)] for q in samples])
    # Group equal/near-equal C values so a threshold cannot split samples at
    # the same equilibrium coordinate. The grid cap keeps the DP bounded.
    cuts = np.r_[0, np.flatnonzero(np.diff(c) > 1e-10)+1, len(c)]
    groups = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        wg = w[lo:hi]
        groups.append((float(np.average(t[lo:hi], weights=wg)),
                       float(np.average(d[lo:hi], weights=wg)), float(wg.sum()),
                       float(np.average(c[lo:hi], weights=wg))))
    # Optional deterministic compression to a finite C grid.
    if len(groups) > grid_size:
        edges = np.linspace(0, len(groups), grid_size+1).round().astype(int)
        reduced = []
        for a, b in zip(edges[:-1], edges[1:]):
            block = groups[a:b]
            weights = np.array([x[2] for x in block])
            reduced.append((float(np.average([x[0] for x in block], weights=weights)),
                            float(np.average([x[1] for x in block], weights=weights)),
                            float(weights.sum()), float(np.average([x[3] for x in block], weights=weights))))
        groups = reduced
    g = len(groups)
    if g < m+1:
        raise ValueError("Threshold partition grid is smaller than region count")
    y_t = np.array([q[0] for q in groups]); y_d = np.array([q[1] for q in groups])
    weights = np.array([q[2] for q in groups]); locations = np.array([q[3] for q in groups])
    scale_t, scale_d = scales
    wt = weights/weights.sum()
    # Segment loss equals the weighted residual around its optimal mean.
    prefix_w = np.r_[0., np.cumsum(wt)]
    prefix_t = np.r_[0., np.cumsum(wt*y_t)]
    prefix_d = np.r_[0., np.cumsum(wt*y_d)]
    prefix_t2 = np.r_[0., np.cumsum(wt*(y_t/scale_t)**2)]
    prefix_d2 = np.r_[0., np.cumsum(wt*(y_d/scale_d)**2)]
    cost = np.full((g, g+1), np.inf)
    for lo in range(g):
        for hi in range(lo+1, g+1):
            sw = prefix_w[hi]-prefix_w[lo]
            mt = (prefix_t[hi]-prefix_t[lo])/sw
            md = (prefix_d[hi]-prefix_d[lo])/sw
            cost[lo, hi] = ((prefix_t2[hi]-prefix_t2[lo])-sw*(mt/scale_t)**2
                            +(prefix_d2[hi]-prefix_d2[lo])-sw*(md/scale_d)**2)
    dp = np.full((m+2, g+1), np.inf)
    back = np.full((m+2, g+1), -1, dtype=int)
    dp[0, 0] = 0.
    for regions in range(1, m+2):
        for hi in range(regions, g+1):
            options = dp[regions-1, :hi] + cost[:hi, hi]
            lo = int(np.argmin(options))
            dp[regions, hi] = options[lo]
            back[regions, hi] = lo
    boundaries = []
    hi = g
    for regions in range(m+1, 1, -1):
        lo = int(back[regions, hi])
        boundaries.append(.5*(locations[lo-1]+locations[lo]))
        hi = lo
    thresholds = np.array(boundaries[::-1])
    if len(thresholds) != m or np.any(np.diff(thresholds) <= 0):
        raise RuntimeError("Dynamic-programming threshold placement failed")
    return thresholds, {"weighted_partition_loss": float(dp[m+1, g]),
                        "spectral_samples": len(samples), "partition_groups": g,
                        "i_range": list(i_range), "weighting": "uniform dI per sampled equilibrium branch"}


class NonisolatedEquilibria(ValueError):
    pass


def model_roots(theta: np.ndarray, thresholds_c: np.ndarray, ip3: float,
                c_ref: float, c_min: float, c_max: float) -> list[tuple[float, float]]:
    """All isolated roots on positive C, including roots outside the physical box."""
    m=len(thresholds_c);ncoef=m+3;eta=theta[-4:]
    alpha,beta=theta[:ncoef],theta[ncoef:2*ncoef];diff=alpha-beta
    zt=phi(thresholds_c,eta,c_ref);knots=np.r_[-np.inf,zt,np.inf]
    slopes=diff[1]+np.r_[0.,np.cumsum(diff[3:])]
    intercepts=diff[0]+diff[2]*ip3-np.r_[0.,np.cumsum(diff[3:]*zt)]
    roots=[]
    for lo,hi,slope,intercept in zip(knots[:-1],knots[1:],slopes,intercepts):
        if abs(slope)<1e-14:
            if abs(intercept)<1e-12:raise NonisolatedEquilibria('coincident nullclines on a region')
            continue
        zr=-intercept/slope
        if not lo-1e-10<=zr<=hi+1e-10:continue
        left,right=c_min,c_max
        for _ in range(60):
            if float(phi(np.array([left]),eta,c_ref)[0])<=zr:break
            left*=.01
        for _ in range(60):
            if float(phi(np.array([right]),eta,c_ref)[0])>=zr:break
            right*=2
        c=float(brentq(lambda c:float(phi(np.array([c]),eta,c_ref)[0])-zr,left,right,xtol=1e-12))
        if roots and abs(c-roots[-1][0])<1e-8:continue
        h=float((hinge_design(np.array([zr]),np.array([ip3]),zt)@beta)[0])
        roots.append((c,h))
    return roots


class MixedObjective:
    def __init__(self, run_dir: Path, config: dict):
        self.run = read_json(run_dir/"run.json")
        self.m = int(self.run["m"])
        self.c_ref = float(config["coordinate_transform"]["C_ref"])
        self.thresholds = np.load(run_dir/"optimizer"/"thresholds.npz")["C"]
        self.spec = config
        refdir = run_dir/"reference"
        with np.load(refdir/"nullclines.npz") as d:
            C, I = d["C"], d["I"]
            keep_c = np.unique(np.rint(np.linspace(0, len(C)-1, 56)).astype(int))
            keep_i = np.unique(np.rint(np.linspace(0, len(I)-1, 40)).astype(int))
            cc, ii = np.meshgrid(C[keep_c], I[keep_i], indexing="ij")
            ix_c, ix_i = np.ix_(keep_c, keep_i)
            mask = d["valid_N"][ix_c, ix_i].ravel()
            self.c_null = cc.ravel()[mask]
            self.i_null = ii.ravel()[mask]
            self.n_null = d["N"][ix_c, ix_i].ravel()[mask]
            self.h_null = d["H"][ix_c, ix_i].ravel()[mask]
        wc = quadrature_weights(C[keep_c]); wi = quadrature_weights(I[keep_i])
        weights = (wc[:, None]*wi[None, :]).ravel()[mask]
        self.null_weights = weights/weights.sum()
        raw = json.loads((refdir/"equilibria.json").read_text())["equilibria"]
        lo, hi = config["data"]["nullcline_I_domain"]
        self.equilibria = [q for q in raw if lo <= q["I"] <= hi]
        levels = sorted({float(q["I"]) for q in self.equilibria})
        stride = max(1, len(levels)//24)
        self.eq_levels = levels[::stride]
        self.eq_by_i = {x: [q for q in self.equilibria if abs(q["I"]-x) < 1e-10]
                        for x in self.eq_levels}
        self.eq_normalizer = max(1, sum(map(len, self.eq_by_i.values())))
        spectral_lo, spectral_hi = .3, 1.2
        self.spectral = [q for q in raw if spectral_lo <= q["I"] <= spectral_hi]
        i_levels = sorted({float(q["I"]) for q in self.spectral})
        weights_i = dict(zip(i_levels, quadrature_weights(np.array(i_levels))))
        self.spec_weights = np.array([weights_i[q["I"]] for q in self.spectral])
        self.spec_weights /= self.spec_weights.sum()
        self.c_bounds = (1e-5, self.run["parameters"]["C_0"]/(1+float(self.run["parameters"]["c_1"]))-1e-5)
        self.eq_slots = max(self.m+1, max(map(len, self.eq_by_i.values()), default=1))

    def residual(self, x: np.ndarray) -> np.ndarray:
        ncoef = self.m+3
        alpha, beta = x[:ncoef], x[ncoef:2*ncoef]
        loga, logb = x[2*ncoef:2*ncoef+2]
        eta = x[-4:]
        a, b = np.exp(loga), np.exp(logb)
        z = phi(self.c_null, eta, self.c_ref)
        zt = phi(self.thresholds, eta, self.c_ref)
        X = hinge_design(z, self.i_null, zt)
        n_hat, h_hat = X@alpha, X@beta
        sc = np.sqrt(self.null_weights)
        r_null = np.r_[(n_hat-self.n_null)/self.spec["data"]["nullcline_scale_N"],
                       (h_hat-self.h_null)/self.spec["data"]["nullcline_scale_H"]]*np.r_[sc, sc]

        c_samples = np.array([q["C"] for q in self.spectral])
        region = np.searchsorted(self.thresholds, c_samples, side="right")
        n_slopes = alpha[1]+np.r_[0., np.cumsum(alpha[3:])]
        h_slopes = beta[1]+np.r_[0., np.cumsum(beta[3:])]
        tr = -a*n_slopes[region]-b
        det = a*b*(n_slopes[region]-h_slopes[region])
        r_spec = np.r_[(tr-np.array([q["trace"] for q in self.spectral]))/
                       self.spec["data"]["spectral_scale_trace"],
                       (det-np.array([q["determinant"] for q in self.spectral]))/
                       self.spec["data"]["spectral_scale_determinant"]]
        sw = np.r_[np.sqrt(self.spec_weights), np.sqrt(self.spec_weights)]
        r_spec *= sw*np.sqrt(self.spec["loss_weights"]["spectral"])

        r_eq = []
        for ip3, refs in self.eq_by_i.items():
            candidates = model_roots(x, self.thresholds, ip3, self.c_ref, *self.c_bounds)
            nref, nmod = len(refs), len(candidates)
            slots = nref+self.eq_slots
            miss = float(self.spec["data"]["equilibrium_missing_penalty"])
            extra = float(self.spec["data"]["equilibrium_extra_penalty"])
            cost = np.full((nref+nmod, nmod+nref), 1e6)
            for i, ref in enumerate(refs):
                for j, (c, h) in enumerate(candidates):
                    cost[i,j] = ((c-ref["C"])/self.spec["data"]["equilibrium_scale_C"])**2 + ((h-ref["h"])/self.spec["data"]["equilibrium_scale_h"])**2
                cost[i, nmod+i] = miss
            for j in range(nmod):
                cost[nref+j, j] = extra
            if nmod and nref:
                cost[nref:, nmod:] = 0.
            rows, cols = linear_sum_assignment(cost)
            terms = np.zeros(2*slots)
            slot = 0
            for i, j in zip(rows, cols):
                if i < nref and j < nmod:
                    c, h = candidates[j]; ref = refs[i]
                    terms[2*slot:2*slot+2] = [(c-ref["C"])/self.spec["data"]["equilibrium_scale_C"],
                                              (h-ref["h"])/self.spec["data"]["equilibrium_scale_h"]]
                    slot += 1
                elif i < nref and j >= nmod:
                    terms[2*slot] = math.sqrt(miss); slot += 1
                elif i >= nref and j < nmod:
                    terms[2*slot] = math.sqrt(extra); slot += 1
            r_eq.extend(terms)
        r_eq = np.asarray(r_eq)*np.sqrt(self.spec["loss_weights"]["equilibria"]/self.eq_normalizer)
        return np.r_[r_null, r_spec, r_eq]

    def parts(self, x: np.ndarray) -> dict:
        r = self.residual(x)
        # Recompute the residual group boundaries for transparent scoring.
        nnull = 2*len(self.c_null)
        nspec = 2*len(self.spectral)
        return {"nullclines": float(r[:nnull]@r[:nnull]),
                "realized_spectra_weighted": float(r[nnull:nnull+nspec]@r[nnull:nnull+nspec]),
                "equilibria_weighted": float(r[nnull+nspec:]@r[nnull+nspec:]),
                "precycle_total": float(r@r)}


def initial_vector(objective: MixedObjective, seed: int,
                   warm_model: dict | None = None) -> np.ndarray:
    objective = getattr(objective,'geometry',objective)
    m, c_ref = objective.m, objective.c_ref
    eta = np.array([0., 0., 0., 0.])
    rng = np.random.default_rng(seed)
    if seed and warm_model is None:
        kind = seed % 4
        eta = [np.array([7., -7., 0., 0.]), np.array([-7., -7., 0., 0.]),
               np.array([4., -4., rng.uniform(-1.5, 1.5), 0.]),
               np.array([rng.uniform(-3,3), rng.uniform(-3,3), rng.uniform(-1.5,1.5), rng.uniform(-2,2)])][kind]
    if warm_model is not None:
        transform = warm_model["transform"]
        eta = np.array([transform["p"], transform["e"], np.log(transform["n"]), transform["k"]])
    zt = phi(objective.thresholds, eta, c_ref)
    X = hinge_design(phi(objective.c_null, eta, c_ref), objective.i_null, zt)
    if warm_model is None:
        target_n, target_h = objective.n_null, objective.h_null
        loga, logb = np.log(1.), np.log(.2)
    else:
        old_eta_info = warm_model["transform"]
        old_eta = np.array([old_eta_info["p"], old_eta_info["e"],
                            np.log(old_eta_info["n"]), old_eta_info["k"]])
        old_x = hinge_design(phi(objective.c_null, old_eta, c_ref), objective.i_null,
                             phi(np.asarray(warm_model["physical_thresholds_C"]), old_eta, c_ref))
        target_n = old_x@np.asarray(warm_model["alpha"])
        target_h = old_x@np.asarray(warm_model["beta"])
        loga, logb = np.log(warm_model["a"]), np.log(warm_model["b"])
    alpha = np.linalg.lstsq(X, target_n, rcond=None)[0]
    beta = np.linalg.lstsq(X, target_h, rcond=None)[0]
    rng = np.random.default_rng(seed)
    if seed != 0:
        alpha += rng.normal(0, .03, len(alpha))
        beta += rng.normal(0, .03, len(beta))
        loga += rng.normal(0, .7)
        logb += rng.normal(0, .7)
    return np.r_[alpha, beta, loga, logb, eta]


def optimize_trial(objective: MixedObjective, x0: np.ndarray, lower: np.ndarray,
                   upper: np.ndarray, max_nfev: int, checkpoint_dir=None):
    """Worker entry point. Kept at module scope for multiprocessing pickling."""
    started = time.perf_counter()
    label=checkpoint_dir.name if checkpoint_dir is not None else "trial"
    print(f"{label}: precycle stage started",flush=True)
    def checkpoint(label,result,epsilon=None):
        if checkpoint_dir is None:return
        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        target=checkpoint_dir/(label+'.json');temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(x=result.x.tolist(),epsilon=epsilon,
            nfev=int(result.nfev),status=int(result.status),success=bool(result.success),
            history=history),indent=2,allow_nan=False)+'\n')
        temporary.replace(target)
    geometry = getattr(objective, 'geometry', objective)
    result = least_squares(geometry.residual, x0, bounds=(lower, upper), method="trf",
                         x_scale="jac", max_nfev=max_nfev,
                         ftol=1e-8, xtol=1e-8, gtol=1e-8)
    history=[dict(stage='precycle',nfev=int(result.nfev),loss=float(2*result.cost),success=bool(result.success))]
    checkpoint('precycle_checkpoint',result)
    if hasattr(objective,'epsilon_schedule'):
        for epsilon in objective.epsilon_schedule:
            objective.epsilon=epsilon
            print(f'{label}: cycle continuation epsilon={epsilon}',flush=True)
            result=least_squares(objective.residual,result.x,jac=objective.jac,
                bounds=(lower,upper),method='trf',x_scale='jac',
                max_nfev=objective.cfg['max_nfev_per_epsilon'],ftol=1e-7,xtol=1e-7,gtol=1e-7)
            details=objective.parts(result.x)
            history.append(dict(stage='cycles',epsilon=epsilon,nfev=int(result.nfev),
                loss=details['full_total'],success=bool(result.success),
                return_domain_admissible=details['return_domain_admissible']))
            checkpoint('epsilon_'+str(epsilon),result,epsilon)
        result.return_report=objective.report
    result.parts_value=objective.parts(result.x)
    result.history=history
    result.elapsed_seconds = time.perf_counter()-started
    result.bound_hits = int(np.sum(np.minimum(result.x-lower, upper-result.x)<1e-6))
    return result


def fit_run(run_dir: Path, config: dict, restarts: int, max_nfev: int, seed: int,
            workers: int, warm_model: dict | None = None) -> None:
    run = read_json(run_dir/"run.json")
    out = run_dir/"optimizer"
    threshold_path = out/"thresholds.npz"
    if not threshold_path.exists():
        eq = json.loads((run_dir/"reference"/"equilibria.json").read_text())["equilibria"]
        i_range = (0.3, 1.2)
        c_max = run["parameters"]["C_0"]/(1+run["parameters"]["c_1"])
        thresholds, info = partition_thresholds(eq, run["m"],
            config["optimization"]["threshold_placement_grid_size"], i_range,
            (0.03, c_max-1e-4),
            scales=(config["data"]["spectral_scale_trace"], config["data"]["spectral_scale_determinant"]))
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(threshold_path, C=thresholds)
        (out/"partition.json").write_text(json.dumps({"physical_thresholds_C": thresholds.tolist(),
            **info, "selection_scope": "training spectral branch only"}, indent=2, allow_nan=False)+"\n")
    else:
        info = read_json(out/"partition.json")
        thresholds = np.load(threshold_path)["C"]
    full_cycles = bool(config.get('cycle_fit',{}).get('enabled',False))
    if full_cycles:
        from full_cycle_fit import FullCycleObjective, prepare_cycle_reference
        if not (run_dir/'reference'/'poincare.json').exists():prepare_cycle_reference(run_dir,config)
        objective=FullCycleObjective(run_dir,config)
    else:objective = MixedObjective(run_dir, config)
    m = int(run["m"]); ncoef = m+3
    lo = np.r_[np.full(2*ncoef, -25.), -8., -8., -12., -12., -5., -10.]
    hi = np.r_[np.full(2*ncoef, 25.), 5., 3., 12., 12., 4., 10.]
    trials_dir = out/"trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    (out/"search_setup.json").write_text(json.dumps(dict(
        bounds_lower=lo.tolist(),bounds_upper=hi.tolist(),max_nfev=max_nfev,
        trials=restarts,workers=workers,seed=seed,ftol=1e-8,xtol=1e-8,gtol=1e-8,
        jacobian="variational returns; geometry central differences" if full_cycles else "numerical 2-point",nullcline_nodes=len(getattr(objective,'geometry',objective).c_null),
        spectral_samples=len(getattr(objective,'geometry',objective).spectral),equilibrium_I_nodes=getattr(objective,'geometry',objective).eq_levels),indent=2)+"\n")
    winner = None
    starts = []
    for trial in range(restarts):
        trial_seed = seed+trial
        start_model = warm_model if trial == 0 else None
        x0 = np.clip(initial_vector(objective, trial_seed if trial else 0, start_model), lo+1e-7, hi-1e-7)
        starts.append((trial, trial_seed, x0))

    print(f"Submitting {restarts} restarts on {workers} processes; max_nfev={max_nfev}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(optimize_trial, objective, x0, lo, hi, max_nfev,trials_dir/f"trial_{trial:03d}"): (trial, trial_seed)
                   for trial, trial_seed, x0 in starts}
        for future in as_completed(futures):
            trial, trial_seed = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failed = trials_dir/f"trial_{trial:03d}"
                failed.mkdir(exist_ok=True)
                (failed/"failure.json").write_text(json.dumps({"seed": trial_seed, "error": repr(exc)}))
                print(f"{run['run_id']} trial {trial+1}/{restarts} FAILED: {exc}", flush=True)
                continue
            parts = result.parts_value
            record = {"run_id": run["run_id"], "trial": trial, "seed": trial_seed,
                      "workers": workers, "success": bool(result.success), "status": int(result.status),
                      "message": result.message, "nfev": int(result.nfev),
                      "cost_half_sum_squares": float(result.cost), "parts": parts,
                      "optimality": float(result.optimality),
                      "elapsed_seconds": result.elapsed_seconds, "bound_hits": result.bound_hits, "continuation": result.history}
            trial_dir = trials_dir/f"trial_{trial:03d}"
            trial_dir.mkdir(exist_ok=True)
            model = {
                "schema_version": 1, "run_id": run["run_id"], "case": run["case"],
                "K_ER": run["K_ER"], "threshold_count_m": m, "regions": m+1,
                "physical_thresholds_C": thresholds.tolist(),
                "alpha": result.x[:ncoef].tolist(), "beta": result.x[ncoef:2*ncoef].tolist(),
                "a": float(np.exp(result.x[2*ncoef])), "b": float(np.exp(result.x[2*ncoef+1])),
                "transform": {"p": float(result.x[-4]), "e": float(result.x[-3]),
                              "n": float(np.exp(result.x[-2])), "k": float(result.x[-1]),
                              "C_ref": objective.c_ref},
                "fit_scope": "full five-term objective with epsilon continuation" if full_cycles else "pre-cycle objective",
                "cycle_loss_included": full_cycles, "independent_validation_completed": False,
                "protocol_status": "cycle_fit_pending_independent_validation" if full_cycles else "precycle_only",
            }
            (trial_dir/"candidate.json").write_text(json.dumps(model, indent=2, allow_nan=False)+"\n")
            (trial_dir/"result.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
            if full_cycles:
                (trial_dir/'cycle_returns.json').write_text(json.dumps(result.return_report,indent=2,allow_nan=False)+'\n')
            score=parts.get('full_total',parts.get('precycle_total'))
            ranking=(not parts.get('return_domain_admissible',True), score)
            if winner is None or ranking < winner[0]:
                winner = (ranking, trial_dir, model, record)
                (out/"best_so_far.json").write_text(json.dumps({"candidate": model,
                    "optimizer_result": record}, indent=2, allow_nan=False)+"\n")
            print(f"{run['run_id']} trial {trial+1}/{restarts}: "
                  f"loss={score:.7g} admissible={not ranking[0]}", flush=True)
    if winner is None:
        raise RuntimeError("No optimizer trials ran")
    selected = {"selected_trial": str(winner[1].relative_to(run_dir)),
                "threshold_partition": info, "training_loss_parts": winner[3]["parts"],
                "candidate": winner[2],
                "optimizer_result": winner[3],
                "return_domain_admissible": not winner[0][0],
                "status": ("no_admissible_return_domain_candidate" if winner[0][0] else "cycle fit completed; inspect independent validation") if full_cycles else "precycle only"}
    if full_cycles and winner[0][0]:print("No admissible return-domain candidate; saved selection is diagnostic only.",flush=True)
    (out/"selected.json").write_text(json.dumps(selected, indent=2, allow_nan=False)+"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["baseline", "threshold_reduction", "parameter_sweep"])
    parser.add_argument("--case", choices=["FM", "AM"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--workers", type=int, default=1,
                        help="independent optimizer processes; set to available CPU cores")
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if not args.all and args.stage is None:
        parser.error("choose --all or specify --stage")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    protocol = read_json(PACKAGE/"config"/"protocol.json")
    stages = protocol["optimization"]["stage_order"] if args.all else [args.stage]
    cases = [args.case] if args.case else ["FM", "AM"]
    plan_path = ROOT/"runs"/"fm_am_reduction"/"plan"/"index.json"
    if not plan_path.exists():
        parser.error("no plan found; run scripts/plan.py first")
    run_ids = sorted(read_json(plan_path)["runs"])
    for stage in stages:
        for case in cases:
            matching = [x for x in run_ids if x.startswith(f"{stage}/{case}_")]
            matching.sort(key=lambda x: int(read_json(ROOT/"runs"/"fm_am_reduction"/x/"run.json")["m"]),
                          reverse=True)
            for run_id in matching:
                run_dir = ROOT/"runs"/"fm_am_reduction"/run_id
                if not (run_dir/"reference"/"equilibria.json").exists():
                    raise FileNotFoundError(f"Reference data missing for {run_id}; prepare it first")
                run = read_json(run_dir/"run.json")
                warm_model = None
                if run.get("warm_start_from_previous_m"):
                    previous_m = int(run["m"])+1
                    previous_prefix = f"{case}_m{previous_m:02d}_K{str(run['K_ER']).replace('.', 'p')}"
                    previous_stage = "baseline" if previous_m == 7 else "threshold_reduction"
                    previous_dir = ROOT/"runs"/"fm_am_reduction"/previous_stage/previous_prefix
                    previous_selected = previous_dir/"optimizer"/"selected.json"
                    if not previous_selected.exists():
                        raise FileNotFoundError(f"Warm-start candidate missing: {previous_selected}")
                    warm_model = read_json(previous_selected)["candidate"]
                fit_run(run_dir, protocol, args.restarts or run["restarts"],
                        args.max_nfev, args.seed if args.seed is not None else run["seed"],
                        args.workers, warm_model)


if __name__ == "__main__":
    main()
