# Exploratory lower-zone fit with fixed fold

The three source models have two calcium thresholds and a numerically found
fold. We insert a third threshold `cut` below a sampled minimum of the fold
orbit with a 10% safety margin. The two nullclines receive independent
corrections proportional to `(cut-C)_+`; the field above the cut is unchanged.
Thus the old fold and its local dynamics persist provided the fold orbit stays
above the cut. The stored minimum is sampled, not an interval proof.

Only cut and two correction coefficients are optimized. The fold is preserved
by construction, while rest residual and the sign constraints on the new-zone
trace/determinant are *penalties*, not exact constraints. The optimizer can
therefore return a candidate that fails them, has negative calcium, worsens the
fit, fails to converge, or misses other bifurcations. This is a pilot, not an
automatically validated final model. The old training/optimizer metadata in
candidate JSON is inherited provenance and must not be interpreted as new.

The objective uses Cmin, Cmax and period for the seven saved training cycles;
`--training three` is a fast smoke option. It uses SciPy event detection at
constant IP3, not NEST. A separate set of four saved holdout points is computed
only after fitting. Duration and convergence criteria can discard slow cycles;
each `null` is a failed or unresolved cycle. One deterministic start per
case is used. Compare the baseline and candidate on the same rows and inspect
the optimizer status, rest residual, lower-zone spectrum, all `min_all` and
the held-out points. Check the fold, other equilibria and entire branches
independently before scientific claims or network benchmarks.

Dependencies: Python, NumPy, SciPy. Use a fresh output path every time.
For example:

```bash
python -m unittest discover -s tests -p 'test_lower_fixed_fold.py' -v
OPENBLAS_NUM_THREADS=1 python scripts/fit_lower_fixed_fold.py \
  --cases AM AFM FM --training three --max-nfev 2 --duration 240 \
  --output results/lower_fold_smoke
OPENBLAS_NUM_THREADS=1 python scripts/fit_lower_fixed_fold.py \
  --cases AM AFM FM --training all --max-nfev 80 --duration 400 \
  --output results/lower_fold_main
```

Cases are processed sequentially and saved after each case completes. On
interruption, completed cases remain; the in-progress case is not checkpointed.
The output directory cannot be reused. Optimizer `max-nfev` counts iterations
as defined by SciPy, not all numerical Jacobian residual calls. No claim about
NEST runtime can be derived from this experiment.
