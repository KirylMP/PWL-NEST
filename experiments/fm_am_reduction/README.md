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
   reduced-threshold follow-up matrix, to run after stage 2 identifies useful
   values of `m`.

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
```

Generated plans and reference data go under `runs/fm_am_reduction/` and are
ignored by Git. Each run has its own immutable `run.json`, dataset files, and
logs; no model or result from another run is overwritten. Run one stage/case
with, for example:

```bash
python experiments/fm_am_reduction/scripts/plan.py --stage baseline --case FM
python experiments/fm_am_reduction/scripts/prepare_reference_data.py --stage baseline --case FM
```

## Implementation status

The plan expansion and Li–Rinzel reference dataset generation are implemented.
The full optimizer from the supplied protocol is **not yet implemented**: in
particular, the spectral partition optimizer, joint 26-parameter fitting,
Poincaré-return cycle loss/continuation, threshold-count comparison, and
independent structural validation are still pending. Existing repository
`fit_*` scripts fit different model families and are intentionally not called
as substitutes. Do not interpret prepared reference data as a fitted result.

The supplied document is FM-specific. AM/K extensions are captured in
`config/cases.json`, while protocol-defined scales and objective weights are
recorded in `config/protocol.json`. Any change to the objective should use a
new protocol/config version and a separate run directory.
