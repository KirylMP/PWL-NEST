# Exploratory log and square coordinate fits

This is an exploratory comparison, not a replacement for the selected paper
models. It fits continuous three-zone affine fields in `u,q` for AM, AFM and
FM, with calcium `exp(u)` µM or `u**2` µM. Both coordinates share thresholds
transformed from the selected physical models and the same seven LR training
and four held-out inputs. The three equilibrium anchors at I=0,I1,I2 and the
stable/unstable/stable regional spectral conditions are imposed. In FM, the
I1 anchor is the specified LR equilibrium, not an arbitrary root of its
multi-valued equilibrium equation.

Fitting minimizes normalized errors in the transformed LR vector field on
seven reference trajectories by constrained least squares. This is NOT the
previous cycle-extrema, period and waveform loss. It does NOT enforce the
prescribed fold. The selected physical models were trained with a different
objective, so this is not a controlled comparison of their full optimization
protocols. Do not call the provisional fits fixed-fold models.

Test cycles are detected in 600 s DOP853 trajectories using the last peaks,
within 3% agreement of the last two periods. Minima and maxima use 256 samples
per orbit; finite integration, the peak test and interpolation do not prove
the cycle or its stability. For square readout, zero crossing can create
additional calcium peaks not aligned with maxima of u. All detected sampled
training/holdout cycles in this run had u>0, but this is not a global guarantee.
Numerical log readout can underflow for large negative u even though the
mathematical exponential is strictly positive. Positivity of C does not imply
other physical constraints, such as an upper calcium bound.

AM at I=0.75 µM continues oscillating in both new fits, unlike the selected
physical candidate; the important upper fold is not established. The joint
fixed-I fold attempts provided here terminated with significant residuals and near-zero crossing speeds: they are NOT validated
folds and their proposed coefficients must not be used as fitted candidates.

From the repository root, with NumPy, SciPy and Matplotlib installed:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD/scripts:$PWD/src"
for case in AM AFM FM; do
  python scripts/fit_transformed_coordinates.py --case "$case" --out results/coordinate_experiment
done
```

For the included diagnostic scripts, data are expected under
`results/coordinate_experiment`; `diagnose_coordinate_fit.py` simulates the
same piecewise-constant input as the paper and uses DOP853 independently of
the NEST benchmark. It is not a performance benchmark.
