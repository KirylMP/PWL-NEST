# Server multistart: fixed-fold transformed-coordinate models

`scripts/run_transformed_server_batch.py` first collects a requested number of
numerically distinct fixed-fold candidates for each of AM, AFM and FM in log
coordinates, then optimizes all of them and evaluates four held-out inputs. It
uses at most `--workers` child processes, with BLAS/OpenMP thread counts fixed
to one. The simulation and analysis use NumPy/SciPy, not NEST.

## Search and admissibility

Each mode starts with its previously found neutral-orbit candidate.
Subsequent seeded starts perturb seven free normalized field coefficients and
solve numerically for one dependent coefficient and four crossing times, with
the fold input fixed. Noise standard deviations cycle over 0.1, 1/3, 1 and 3
times `--sigma`, default 0.03. They sample around a known candidate and within
a local fold-root trust region; they are not a uniform or global search.
Changing that trust region or the sampling protocol changes the experiment.

Before counting a distinct fold candidate, the runner verifies the fixed rest
and threshold-equilibrium anchors, thresholds, fixed fold input, regional
spectra, orbit matching, log(M), crossing directions, and 301 samples per
segment. It independently integrates the orbit and accumulates divergence; it
also estimates nondegeneracy in the segment matching system at several finite
difference steps. Coefficients and flight times are compared for numerical
duplicates using `--dedup-tol`. Tests are finite and are not proofs of first
exit, fold nondegeneracy, uniqueness, or global branch completeness. The
exponential calcium readout ensures nonnegative calcium but does not enforce an
LR upper-pool bound.

For each candidate the sampler checks all eight free field coefficients as
possible dependent parameters in the fold equations. For each one it forms
the 5×5 finite-difference Jacobian with respect to that scaled coefficient and
the four log flight times. Full rank is the local implicit-function-theorem
condition for solving those five unknowns near the current candidate while
holding the other coefficients fixed. The sampler uses the best-conditioned
pivot and records every coefficient's relative smallest singular value.
`pivot_fixability.csv` reports how often each coefficient passes the local
rank check and how often it was selected. This is a local numerical diagnostic;
it does not guarantee a root after large parameter changes or global
identifiability.

The quota counts screened folds, **not** successful seven-cycle fits or
four-cycle holdouts. A valid fold can have no numerically discoverable stable
cycle at a training input, can fail numerical fitting, or can fail held-out
verification. Each outcome is saved. Up to `--max-attempts` are allowed per
combination. If fewer than the quota are found, the runner reports the deficit
and optimizes those it did find; rejected starts do not prove nonexistence.

Independent verification defaults to all training and holdout inputs. To rerun
only the four holdouts for one saved candidate with per-input progress, use
`python -u scripts/verify_transformed_fold.py CANDIDATE --subset holdout --progress`.
This writes a separate `*_verification_holdout.json` file and preserves the
original fit and full-verification outputs. Subset mode skips the duplicate
fold IVP/nondegeneracy audit; it records the fold values from the candidate and
relies on the batch's independent `fold_validation` for that check.

The 7-point fitting objective and safeguards are described in
`CYCLE_OPTIMIZATION.md`. The implicit fold solve, root trust region and fixed
itinerary may prevent exploration of other branches. Optimizer status such as
`xtol` is not evidence of a global minimum. The search uses up to
`--fit-nfev` evaluations per start, but may stop earlier.

## Outputs and restart

All files are written beneath the explicit `--output` directory. It must be
new or empty. Each attempt retains a `status.json`, a log and, where found,
coefficients; distinct candidates are recorded in `accepted.json`. Fitting
outputs have per-start logs, candidate, report, independent validation, and
`status.json`. `summary.csv`, `pivot_fixability.csv` and `batch_result.json` are
written at completion.
The summary script also works while the batch is running. It chooses the best
completed candidate by **training loss**, not by holdout performance.

`--resume` reuses completed work for the same output and numerical settings;
`--workers` may change. Attempts and seeds already assigned are not repeated.
After phase 1 the set of accepted starts is frozen for phase 2. To retry fits
with a failed or unresolved status add `--retry-failed` along with `--resume`.
To change quota or numerical search settings, start a new output directory.
Results and their upstream SHA256 hashes are kept, but the code does not pack
or push the results automatically.

Example minimal smoke test (output must not already exist):

```bash
export PYTHONPATH="$PWD/scripts:$PWD/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -p 'test_transformed_*' -v
python scripts/run_transformed_server_batch.py --output results/transformed_smoke_01 \
  --valid-per-combo 1 --workers 6 --fold-nfev 100 --fit-nfev 1 --max-attempts 8 --timeout 300
python scripts/summarize_transformed_server_batch.py results/transformed_smoke_01
```

`fit-nfev 1` only tests wiring. For research use increase it, e.g. to 80.
Timing depends strongly on acceptance and convergence; no runtime estimate is
guaranteed. `valid-per-combo 32` means up to 96 screened folds in total, 32
for each of three models; use 64 for up to 192. The log/manifest gives the
actual number. NEST is not invoked and no benchmark is run.
