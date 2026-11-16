# Server multistart: fixed-fold transformed-coordinate models

`scripts/run_transformed_server_batch.py` first collects a requested number of
numerically distinct fixed-fold candidates for each AM/AFM/FM × log/square
combination, then optimizes all of them and evaluates four held-out inputs. It
uses at most `--workers` child processes, with BLAS/OpenMP thread counts fixed
to one. The simulation and analysis use NumPy/SciPy, not NEST.

## Search and admissibility

Each combination starts with its previously found neutral-orbit candidate.
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
exit, fold nondegeneracy, uniqueness, or global branch completeness. Neither
the square nor exponential calcium readout ensures an LR upper-pool bound.

The quota counts screened folds, **not** successful seven-cycle fits or
four-cycle holdouts. A valid fold can have no numerically discoverable stable
cycle at a training input, can fail numerical fitting, or can fail held-out
verification. Each outcome is saved. Up to `--max-attempts` are allowed per
combination. If fewer than the quota are found, the runner reports the deficit
and optimizes those it did find; rejected starts do not prove nonexistence.

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
`status.json`. `summary.csv` and `batch_result.json` are written at completion.
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
guaranteed. `valid-per-combo 32` means up to 192 screened folds in total, 32
for each of six combinations; use 64 for up to 384. The log/manifest gives the
actual number. NEST is not invoked and no benchmark is run.
