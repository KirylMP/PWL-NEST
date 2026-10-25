# FM reduction protocol

`protocol/FM_REDUCTION_PROTOCOL.md` is the supplied specification. The supported
entry point for the complete FM experiment is `scripts/run_fm.py`. AM, threshold
reduction and K sweeps stay on hold while FM is being debugged. Legacy
`fit_stage.py --stage ...` commands without a cycle configuration reproduce the
old pre-cycle experiment; they do not run the complete protocol.

From the repository root, with Python >= 3.11:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -u experiments/fm_am_reduction/scripts/run_fm.py \
  --name fm_full_64 --trials 64 --workers 16 \
  --max-nfev 1500 --cycle-max-nfev 60
```

The run prepares LR data, places seven fixed C thresholds by spectral partition,
fits all 26 parameters to the initial geometry/spectrum/equilibrium objective,
then fits all 26 to the complete objective at epsilon .15, .05 and .02. The
solution at each epsilon starts the next. Candidate selection always uses .02,
the same section grid and the same scales. Periods and amplitudes are validation
quantities, not training terms.

Default training I nodes are .55, .65, .8, 1.0, 1.1 and 1.14; the last
node supplies a nonoscillatory reference regime with a regular transient return.
The five weights are 1,10,10,30,30. `L_cyc` averages over training I nodes;
`L_extra` is summed without averaging. Gaussian-smoothed reference counts are
not rounded. A separate numerical feasibility barrier prevents missing return
domains being interpreted as zero cycles. It is exactly zero on admissible
return domains; inadmissible results are labeled and ranked behind admissible
ones. The definition of the numerical coverage threshold is saved in config.

This is materially more expensive than the old fit: `--max-nfev` is the initial
stage budget, and `--cycle-max-nfev` is a separate budget for each epsilon.
Each cycle evaluation integrates a family of first-return problems and their
sensitivities. CPU processes parallelize restarts. CUDA is not used.

For a previous FM candidate, add this optional first warm start:

```bash
  --start-from runs/fm_am_reduction/fm_debug_64/optimizer/selected.json
```

Other restarts remain independent. Existing run directories are never reused;
choose a fresh `--name`. Data preparation, logs and reports are automatic.

## Output

Everything is under `runs/fm_am_reduction/<name>/` (ignored by Git):

- `logs/run_*.log`: unbuffered stdout, stderr, errors and continuation progress.
- `run.json`, `protocol.json`, `source/`: parameters, grids, budgets and source
  snapshots/hashes, including the supplied protocol.
- `reference/poincare.json`: LR return maps, sections, raw smooth counts inputs
  and regular-domain statuses. `reference/` also stores nullclines/equilibria.
- `optimizer/search_setup.json`: actual bounds and optimization settings.
- `optimizer/trials/trial_*/`: initial and epsilon checkpoint vectors, candidate,
  full loss components, convergence history, return tables or explicit failures.
- `optimizer/best_so_far.json`, `selected.json`: current and final selections.
  Check `return_domain_admissible`; a diagnostic fallback is not an accepted fit.
- `analysis/`: direct-trajectory diagrams and unresolved integration records.
- `analysis/structure/`: independent multi-section first-return root enumeration,
  stable/unstable cycles, neighboring-I continuation, equilibrium/cycle signatures,
  matching errors, transition intervals and bifurcation PNG/PDF figures.

Follow progress from another terminal:

```bash
tail -f runs/fm_am_reduction/fm_full_64/logs/run_*.log
```

Cycle validation uses a shifted I grid, more section nodes and tighter tolerances.
It enumerates fixed-point roots of oriented first returns, checks their
multipliers, deduplicates orbits across sections, then corrects neighboring-I
predictions. Thus repelling cycles can be found without waiting for a trajectory
to converge to them. Nonhyperbolic/suspect cases and uncovered domains remain
explicitly unresolved. A finite grid cannot prove global completeness.
`L_str_val_full_grid` is null if any grid points are unresolved; the regular-subset
score is reported separately. Missing cycles never get zero period/amplitude
error. All validation results are excluded from model selection.

Independent validation can be rerun without fitting:

```bash
python experiments/fm_am_reduction/scripts/validate_structure.py \
  --run-dir runs/fm_am_reduction/fm_full_64 \
  --workers 16 --points 161 --nodes 65 --max-time 900
```

Training-grid refinements use `--cycle-I`, `--return-nodes` and
`--return-max-time` in a new run. Recompare final candidates on a common grid;
changing M changes the effective unaveraged extra-cycle term. See
[PROTOCOL_AUDIT.md](PROTOCOL_AUDIT.md) for the implementation mapping and numerical
limits. Regression checks:

```bash
python -m unittest discover -s experiments/fm_am_reduction/tests -v
```
