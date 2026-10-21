# Mixed coordinates and polynomial switching curves

Let `u=phi(C)`, `V=psi(h)`, and define a common shear

\[
U=\phi(C)-g(V),\qquad g(V)=a(V-v_0)+b(V-v_0)^2.
\]

The inverse is explicit:

\[
h=\psi^{-1}(V),\qquad C=\phi^{-1}(U+g(V)).
\]

The Jacobian determinant is `phi'(C)*psi'(h)`; the shear adds no singularity.
The original calcium/gate inverse domains still apply to `U+g(V)` and `V`.
Neither the inverse nor the simulator takes an absolute value or clips calcium.
With identity calcium coordinates, positivity requires a domain check; using
logarithmic calcium coordinates would instead guarantee positive readout.

All switching surfaces have the form `U=theta_j`. In physical coordinates they
are `phi(C)=theta_j+g(psi(h))`. When phi and psi are affine, these are parallel
lines for `b=0` and vertical translates of a quadratic graph for `b!=0`.
Ordered theta values cannot cross. The number of regions is unchanged.
The JSON field `thresholds_C` is only a cross-section at `V=v0` for these models,
not a list of calcium-only switching thresholds. A null entry means that the
surface has no physical intersection with that cross-section.

## What remains analytic

A **new** continuous PWA surrogate is fitted in `(U,V,I)`:

\[
\begin{pmatrix}\dot U\\\dot V\end{pmatrix}
=B_0\begin{pmatrix}1\\U\\V\\I\end{pmatrix}
+\sum_j d_j(U-\theta_j)_+,
\qquad \dot I=-(I-I_\infty)/\tau.
\]

Each regional augmented system has a constant matrix and an affine constant
term, so its motion is obtained by a matrix exponential, including exponential
input. State switching is detected by the scalar condition `U(t)=theta_j`.
In general that condition still requires numerical root finding; no elementary
closed form for all first-exit times or speed advantage is claimed here.

The source Li–Rinzel field is transformed by the full chain rule for fitting:

\[
\dot U=\phi'(C)f_C-g'(V)\psi'(h)f_h,
\qquad \dot V=\psi'(h)f_h.
\]

Physical surrogate derivatives are recovered with the inverse chain rule.
Physical peak times/extrema must be measured after the inverse map: extrema of
U are generally not extrema of C. The old autonomous statistics accelerator,
which treats its first coordinate as calcium, must therefore not be used for
mixed models. The new experiment integrates physical readout trajectories for
static calibration in **all** families, including the unmixed control.

## Coordinate change versus refitting

Conjugating an already fitted PWA field by a nonlinear coordinate map usually
makes its regional field nonlinear. Merely re-expressing the same field in new
coordinates also cannot improve its physical predictions. Here the field is
refitted as PWA in the new coordinates, which defines a different approximation
family with curved physical switching surfaces. Its physical field is generally
non-PWA, while its internal regional matrices are constant.

The exact source Jacobian at an equilibrium transforms by similarity, even for
a nonlinear coordinate map. The source node/focus/saddle transitions therefore
remain the spectral heuristic. The fit retains the previous target types and
soft input anchors, while ordered U thresholds and shear parameters are fitted.

## More general polynomial thresholds

The current pilot fits a shared polynomial and separate level values theta_j.
Fitting arbitrary independent polynomials first does not by itself provide one
global, invertible straightening map. Intersecting boundaries, for example,
cannot become distinct parallel coordinate levels under an invertible map.
More general nonintersecting curves may admit a suitable common coordinate;
one must construct it, check inverse uniqueness and regularity, and then refit
the PWA surrogate in that coordinate. The shared shear is a simple explicit
construction that avoids this additional inverse problem.

## Reproduce the FM continuation

Build the fitting helper with the usual environment setup, then run:

```bash
g++ -O3 -shared -fPIC native/strips_rk4.cpp -o .build/strips_rk4.so
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
  .venv/bin/python scripts/fit_mixed_coordinates.py \
  --output results/fm_mixed_coordinates_replay \
  --seed-model results/fm_one_input_threshold/selected/control.json \
  --global-iterations 24 --budget 800
PYTHONPATH=src:scripts .venv/bin/python scripts/report_mixed_coordinates.py \
  results/fm_mixed_coordinates_replay
```

The source server candidate remains frozen. The continuation uses the same
previously trained control seed for all families. Selection and failure checks
use training only; the six fresh test trajectories never supply field samples
or optimization residuals. The earlier local attempt is retained separately in
`results/fm_mixed_coordinates/first_pass/` and is not reproduced by this command.
