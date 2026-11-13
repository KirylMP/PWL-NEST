# Fixed-fold transformed-coordinate experiment

Starting from the exploratory log and square fits, this experiment constructs
continuous three-zone fields for AM, AFM and FM. Each fit has calcium
`C=exp(u)` µM or `C=u**2` µM. The zero-input rest state and input values of
the two threshold equilibria are fixed; hidden coordinates at the two crossing
equilibria are allowed to change. The corresponding calcium thresholds are
unchanged from the selected physical models. Regional types are constrained to
stable lower node/focus, unstable middle focus, stable upper focus.

At the previously prescribed fold input, the search solves four segment
matching equations and log(M)=0 for the itinerary (middle,upper,middle,lower).
It uses numerical least-squares with a proximity regularizer, then polishes
matching and log(M) to numerical precision. It is not yet the original
parameter-reduced fold-constrained optimization of cycle extrema, period and
waveform. Fits are preliminary and not selected for the paper.

Selected neutral-orbit candidates are AM log 6 / square 3, AFM log 2 / square
3, FM log 7 / square 0. Their residuals are below 4e-13; all have proper
switching directions and 301 sampled points per segment within their zones.
`seed_*_verification.json` records DOP853 closure at three maximal time steps,
the accumulated divergence, and finite-difference singularity/curvature/input
derivative of the segment matching equations. These finite checks are evidence
for folds, not exact proofs of first exits or global branch completeness.
Near the AM lower threshold the normal speed is particularly small. The
piecewise Jacobian is discontinuous across switching surfaces, so independently
integrated divergence is only approximately zero, with its own numerical error.

The 7 training and 4 held-out trajectories use DOP853 600 s from each fixed
rest state. A cycle is counted when at least three recent maxima of u are found
and the last two periods agree within 3%. These are finite-time checks, not a
proof of stability. Extrema are measured on a 256-sample orbit grid. For square
readout the interpretation assumes u>0 along the sampled orbit, which holds for
the presented test cycles; the field permits u to change sign in general.

All held-out trajectories were oscillatory in these finite tests, but amplitude
and period errors vary widely. The FM square neutral orbit reaches approximately
2.58 µM calcium, outside the 2/1.185 µM bound implied by the corresponding LR
closed-pool parameterization; nonnegative calcium is not sufficient for full
physical admissibility. In the 400 s varying-input AM simulation, the log fit
emits only one upward C=0.2 µM event versus eight for the physical fit. Thus a
fixed fold does not alone ensure a good transient response or a useful fit.
Multiple basins or long transients may contribute but have not been resolved.

Reproduce, from repository root with NumPy, SciPy and Matplotlib:

```bash
export PYTHONPATH="$PWD/scripts:$PWD/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/verify_transformed_fold.py results/fold_anchors/AM/square/seed_3.json
python scripts/diagnose_fixed_fold.py
python scripts/plot_fixed_fold_fits.py
```

Multistart search and polishing example:

```bash
python scripts/fold_anchor_multistart.py --case AM --kind square --seed 3 --nfev 1200
python scripts/fold_anchor_multistart.py --case AM --kind square --seed 3 --nfev 5000 --exact --resume results/fold_anchors/AM/square/seed_3.json
```

The paths above overwrite results of that seed; choose a separate output
location or preserve the originals if performing new research runs.
