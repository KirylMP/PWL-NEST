# FM reduction experiments — v5

The supplied scientific specification is `protocol/FM_REDUCTION_PROTOCOL.md`.
The supported FM entry point is `scripts/run_fm.py`. All 26 parameters are free
on the final joint stages. AM, threshold reduction and K sweeps remain separate
future experiments while the FM implementation is being debugged.

The objective keeps the five protocol terms and weights 1,10,10,30,30. Cycle
error is averaged over I; the extra-cycle term is **not** averaged. Periods and
amplitudes are independent diagnostics, not additional training terms. A matched
cycle count does not establish a matched orbit, period, or bifurcation branch.

## What v5 fixes

- A physical section follows a candidate equilibrium. Both LR and PWA use the
  same physical C, h0 and u at every evaluation. Derivatives include movement of
  C and h0 on **both** maps, including the derivative of the normalization of G.
  LR needs only two section sensitivities, which are chained to all parameters.
- The section starts at the lower of the two physical nullclines, clipped at
  h=0, to cover positive crossings of both fields. The anchor is the physical
  candidate equilibrium nearest the reference equilibrium. Source cycle ranges
  supply a coverage guard. Multiple candidate equilibria are independently
  checked for cycles missed by the selected section.
- Independent regional slopes and graph values replace correlated hinge
  coefficients as numerical coordinates. This is an invertible parameter
  conversion. Numerical bounds are saved in those coordinates. Parameter
  transformation derivatives are included in the Jacobian.
- Early geometry and broad-smoothing stages can hold the transform fixed;
  subsequent stages release all 26 parameters. The default epsilon sequence is
  .5 -> .1 -> .01 -> .001. The broad stage supplies useful gradients where
  narrow Gaussian smoothing was saturated. LR training counts are checked before
  fitting; the shared final epsilon is reduced further if needed, with at most
  one source-profile grid refinement. The actual schedule is saved.
- `ftol` and `xtol` are disabled in the v5 solver. `success` means checked
  stationarity on the **common final grid**, together with a passed numerical
  cycle audit. Solver termination, gradient stationarity, numerical admissibility
  and scientific acceptance are recorded separately. A bounded, verified descent
  search can recover from a stalled step; its evaluations are logged.
- Requested I values are filled to the configured maximum coarse gap (default
  .05). Warm-model equilibrium threshold events are bracketed. Later stages add
  midpoints, giving at most .025 gaps by default; known nonsmooth event centers
  are avoided. Source and incumbent cycle profiles add points around observed
  cycle-count transitions before any optimizer starts. This is a recorded
  training-collocation refinement, not use of the independent validation grid.
- u is refined between solver stages around sign changes, extrema and domain
  boundaries. Its layout stays fixed inside each solver call. Every final
  candidate is rescored at the same epsilon, I grid and common nested u grid.
  An additional joint polish optimizes on that common grid before its final
  stationarity and numerical-resolution checks.
- The old half-reference-edge quota is removed. Actual regular intervals,
  observed-cycle coverage and convergence under u refinement/epsilon halving
  are checked separately. Unknown returns are not reported as zero cycles.
- A supplied incumbent is rescored as a separate contender. A warm start on an
  identical threshold basis preserves its exact coefficients: least-squares
  reprojection can lose slopes between close thresholds on a coarse sample grid.

This is a numerical implementation update. It does not prove global optimality
or completeness of all cycle searches. Residuals remain only locally smooth
when event domains, anchors or matching assignments change. Candidates failing
coverage/resolution checks are saved as diagnostic results, not accepted fits.

V5 changes the section/normalization policy and hence the finite-epsilon
numerical objective. The LR smoothed count now depends on the common moving
section, so its section derivative is included. The five terms and their weights
are retained. Historical loss values must be recomputed under v5 before being
compared; the supplied incumbent is automatically rescored for this purpose.

## Server run

From the repository root, using the existing Python environment:

```bash
source .venv/bin/activate
python -u experiments/fm_am_reduction/scripts/run_fm.py \
  --name fm_v5_16 --trials 16 --workers 16 \
  --max-nfev 1500 --cycle-max-nfev 20 \
  --start-from runs/fm_am_reduction/fm_full_v4_64/optimizer/best_so_far.json
```

Use a fresh name. The supplied file may be `candidate.json`, `best_so_far.json`
or `selected.json`. The run creates data, logs, source snapshots, optimization
checkpoints and figures automatically. CPU processes parallelize trials; BLAS
threads are limited to one. CUDA is not used. The extra incumbent evaluation is
not an additional optimized trial.

A full v5 evaluation is substantially more expensive than v4: source returns now
move with the section and the collocation grid is refined. `--cycle-max-nfev`
applies **per epsilon**, with up to six additional recovery evaluations. Grid
preparation, shared-grid rescoring, cycle audits and validation have separate
costs. A final common-grid polish has its own budget (`--common-polish-nfev`,
defaulting to `--cycle-max-nfev`). An exhausted small budget is explicitly
nonstationary. The continuation history and total evaluation count include all
optimizer phases and recovery evaluations.

Defaults:

- initial u grid: 65 points, from 1e-5 to .98;
- maximum per-stage u nodes: 129;
- common selection u grid: 129 or the next complete nested refinement;
- independent validation: 257 section nodes, shifted 81-point I grid;
- final counter tolerance: .025 for grid/epsilon convergence and observed counts;
- stationarity tolerance: 1e-6 in the saved regional coordinates.

Options `--cycle-epsilon`, `--max-I-gap`, `--return-nodes`, `--max-stage-nodes`,
`--selection-nodes`, `--profile-nodes`, `--count-tolerance`,
`--training-I-refinements`, `--common-polish-nfev` and solver budgets are
saved in `protocol.json`. Independent validation must have more section nodes
than common selection. Changing grids changes the unaveraged extra-cycle term's
effective weight; comparisons must use a common configuration.

## Results and logs

Everything is in `runs/fm_am_reduction/<name>/` (ignored by Git):

- `logs/run_*.log`: stdout, stderr, phase, inventory and audit progress;
- `run.json`, `protocol.json`, `source/`: actual settings and source hashes;
- `reference/source_cycle_inventory.json`: training LR cycle observations,
  physical coverage intervals, grids, solver settings and unresolved cases;
- `training_grid_history.json`, `epsilon_calibration.json`: observed transitions,
  the common grid construction and source counter convergence checks;
- `optimizer/search_setup.json`: actual regional bounds, scales and tolerances;
- `optimizer/trials/trial_*/iterations.jsonl`: objective evaluations, step sizes,
  return statuses/anchors, Jacobian spectra and conditioning;
- epsilon checkpoints and `collocation_history.json`: stage grids and iterates;
- `training_audit_progress.json`, `training_audit.json`: coverage, common/refined
  counts, epsilon-halving checks, observed stable/unstable cycles;
- `optimizer/trials/incumbent/`: supplied model, rescored without optimization;
- `optimizer/best_so_far.json`, `selected.json`: common-grid ranking. Inspect
  `numerical_audit_passed`, `converged_stationary` and all loss components;
- `analysis/`: direct trajectories, spectra, nullclines and loss figures;
- `analysis/structure/`: independent multi-section stable/unstable root scans,
  neighbor-I continuation, structural signatures, periods/amplitudes and plots.

`numerical_audit_passed` means the reported cycle score passed the numerical
checks. It does **not** mean that the model matches LR. `accepted` remains false
until scientific diagnostics are reviewed. Missing periods/cycles never receive
zero matching error. A finite grid cannot exclude arbitrarily narrow regimes.

```bash
tail -f runs/fm_am_reduction/fm_v5_16/logs/run_*.log
```

## Recheck a v5 candidate without fitting

```bash
python -u experiments/fm_am_reduction/scripts/cycle_training_audit.py \
  --run-dir runs/fm_am_reduction/fm_v5_16 \
  --out runs/fm_am_reduction/fm_v5_16/analysis/training_audit_rerun
```

The candidate defaults to `selected.json`, or to `best_so_far.json` if the search
is still running. Supply `--candidate path/to/model.json` to override. It must
use the run's threshold partition. The output directory must be fresh.

Independent structural validation can also be rerun in a fresh directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u \
  experiments/fm_am_reduction/scripts/validate_structure.py \
  --run-dir runs/fm_am_reduction/fm_v5_16 \
  --output-dir runs/fm_am_reduction/fm_v5_16/analysis/structure_rerun \
  --workers 16 --points 81 --nodes 257 --max-time 600
```

## Historical runs

`full_cycle_fit.DEFAULTS` and `audit_cycle_counts.py` retain the v3/v4 fixed-source
section semantics for reproducibility. They do not evaluate the new moving
objective. Legacy `fit_stage.py --stage ...` jobs without cycle configuration
still reproduce pre-cycle experiments. The original protocol document is kept
unchanged. Implementation mapping and older corrections are in
`PROTOCOL_AUDIT.md`; numerical findings are in `reports/`.

Regression checks:

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover \
  -s experiments/fm_am_reduction/tests -v
```
