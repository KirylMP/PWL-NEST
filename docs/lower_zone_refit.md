# Fixed-fold local refit, with and without a lower zone

This exploratory experiment starts from the three frozen physical models.
For every trial, nine original alpha/beta coefficients may vary in a bounded
neighborhood, along with the lower cut and two local correction coefficients.
The remaining original coefficient and four crossing times are solved from
four closure equations and log(M)=0 at the original fold input. A trial is
rejected when the solve fails, the sampled itinerary changes, a crossing is
not transverse, or the sampled fold enters the new lower zone. Finite sampling
and numerical solution are not a proof. Other folds may exist.

`--arm three` sets the two added correction coefficients to zero, so the
additional cut is invisible. `--arm four` can fit the correction. Both arms
refit the same old coefficients, use the same loss, seed and budget. This is a
controlled local search, not a proof that either arm is globally optimal.
One pivot coefficient is solved by the fold constraint and is not bounded by
the same local parameter box; inspect `fold.pivot_value`. Old candidate JSON
fields, including `status`, `optimizer`, and `training_cycles`, are inherited
from the baseline and do not validate the refit. Only `lower_zone_refit` and
the new report describe the result.

The objective matches Cmin, Cmax and period on seven saved training points,
with soft rest and lower-zone stability penalties. They are not exact
constraints. The last cycle is accepted only when the two final periods agree
within the configured tolerance. A `null` diagnostic is an unresolved cycle,
never an accuracy of zero. Holdout cycles are calculated only at the end.
Out-of-sample waveforms, equilibrium positions, other cycles and branch
completeness remain to be checked before adoption or NEST timing.

The program reports the best valid evaluated point, including finite-difference
probes; this may not be the final optimizer iterate. A checkpoint records the
best parameters seen, but it cannot resume a run. Each invocation requires a
new output directory. With many coefficients, each optimizer iteration uses
many forward integrations; the full experiment can take hours. Limit parallel
jobs to available cores and set OPENBLAS_NUM_THREADS=1. The base paper branch
does not contain the prior server results, which are not needed to run this.

```bash
python -m unittest discover -s tests -p 'test_lower*fold.py' -v
OPENBLAS_NUM_THREADS=1 python scripts/fit_lower_refit_fixed_fold.py \
  --case AM --arm three --seed 0 --training three --max-nfev 1 \
  --duration 240 --output results/lower_refit_smoke/AM/three/seed_0
python scripts/summarize_lower_refit.py results/lower_refit_smoke
```

The sampled lower-fold C minimum, shooting residuals and independent orbit
closure are recorded; finite-difference curvature and input derivative are
diagnostics only. The independent check is for the three-zone continuation;
the four-zone orbit is the same if the cut clearance check is correct. Inspect
all errors, unsuccessful fits, rest residuals, stability and `min_all`, and
recheck numerical tolerances before interpreting results.
