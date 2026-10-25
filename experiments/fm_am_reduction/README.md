# FM/AM reduction experiments

This folder holds the server experiment plan for the transformed-coordinate,
thresholded Li–Rinzel family. The FM document supplied for this work is kept at
[`protocol/FM_REDUCTION_PROTOCOL.md`](protocol/FM_REDUCTION_PROTOCOL.md).

## Planned sequence

1. Fit the seven-threshold baseline for FM and AM, with the canonical
   `K_ER` values in `config/cases.json`.
2. At each canonical `K_ER`, compare `m = 7, 6, 5, 4, 3` thresholds, using the
   immediately higher-complexity result as a warm start when available.
3. Sweep `K_ER` with `m = 7` first. The config also records an optional
   reduced-threshold follow-up matrix can be made after stage 2 identifies
   useful values of `m`, by setting `parameter_sweep.m_values` in a new
   protocol config version.

The FM protocol defines `K_ER = 0.05`; AM uses the repository's Li–Rinzel
reference `K_ER = 0.1`. All other reference parameters are shared and are
recorded explicitly. The K sweep is an experiment variable, not an instruction
to alter the reference model files.

## Server setup

From the repository root on Ubuntu:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python experiments/fm_am_reduction/scripts/plan.py --all
python experiments/fm_am_reduction/scripts/prepare_reference_data.py --all
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python experiments/fm_am_reduction/scripts/fit_stage.py --all --workers 16
```

Generated plans and reference data go under `runs/fm_am_reduction/` and are
ignored by Git. Each run has its own immutable `run.json`, dataset files, and
logs; no model or result from another run is overwritten. Run one stage/case
with, for example:

```bash
python experiments/fm_am_reduction/scripts/plan.py --stage baseline --case FM
python experiments/fm_am_reduction/scripts/prepare_reference_data.py --stage baseline --case FM
OPENBLAS_NUM_THREADS=1 python experiments/fm_am_reduction/scripts/fit_stage.py --stage baseline --case FM --workers 16
```

## Implementation status

Plan expansion, Li–Rinzel reference datasets, dynamic-programming spectral
threshold placement, and multi-start fitting of the protocol's initial
`L_null + lambda L_spec + mu L_eq` objective are implemented. Stage dependencies
warm-start `m=6…3` from the preceding region count; each still includes
independent starts. The candidate output explicitly records that it is
preliminary.

The full protocol remains incomplete: the smoothed Poincaré-return cycle loss,
continuation of cycles, independent structural validation, period/amplitude
validation, and complexity acceptance gates are not implemented yet. Do not
interpret a pre-cycle fit as the final candidate or as a result of the complete
protocol. Existing repository `fit_*` scripts fit different model families
and are intentionally not used as substitutes.

`fit_stage.py --workers 16` runs independent optimizer restarts in separate
CPU processes. Keep BLAS/OpenMP thread counts at 1 when using process workers
to avoid oversubscribing the cores. The current fitter uses NumPy/SciPy on CPU;
it does not use CUDA, so a GPU will not speed up these runs.

The supplied document is FM-specific. AM/K extensions are captured in
`config/cases.json`, while protocol-defined scales and objective weights are
recorded in `config/protocol.json`. Any change to the objective should use a
new protocol/config version and a separate run directory.

## FM debugging run after implementation audit

Use a fresh experiment; existing runs and prepared caches retain the old definition.
The default 64 starts run on 16 CPU processes. This command prepares only FM,
fits the corrected pre-cycle objective, and generates diagnostic figures:

```bash
python -u experiments/fm_am_reduction/scripts/run_fm.py \
  --name fm_debug_64 --trials 64 --workers 16 --max-nfev 1500
```

Everything is in `runs/fm_am_reduction/fm_debug_64/`:
`logs/run_*.log`, source snapshot, `optimizer/trials/`, `optimizer/selected.json`,
`optimizer/best_so_far.json`, and `analysis/`. Figures include `bifurcation.png`
and PDF, `spectra.png`, and `nullclines.png`. `analysis/scan.json`, `orbits.csv`,
and `summary.json` contain raw diagnostics and unresolved cases. Analysis can
be rerun without fitting:

```bash
python experiments/fm_am_reduction/scripts/analyze_candidate.py \
  --run-dir runs/fm_am_reduction/fm_debug_64 --workers 16 --points 161 --duration 2000
```

See [PROTOCOL_AUDIT.md](PROTOCOL_AUDIT.md) for fixed errors and remaining gaps.
These figures are a finite-grid diagnostic; repelling cycles and the cycle loss
from the protocol are still missing. Focus on FM until that part is debugged.
