# Modal Riccati correction: reproducible FM pilot

Base: four-region PWA with one hidden state, files from `slow_threshold_test`.
No changes to its existing 3 calcium thresholds, hinge coefficients, or 5 hidden-feedback coefficients.

## Model

Let `y=(u,v,r)`, where `u=(C-Cref)/S`, `v=(h-HR)/HS`. The base four-region vector field is `F0(y,I)`. The low-C Jacobian has a real slow eigenvalue `lambda=-0.1047560356 s^-1`, left eigenvector `l=[-0.01060303  1.0055273  -0.02952995]`, right eigenvector `vslow=[0.00580851 0.99687715 0.07875405]` with `l @ vslow=1`.

We add **one** quadratic hinge:

`F(y,I)=F0(y,I)+k*vslow*max(0,l@y-theta)**2`

`k=-0.5956250890, theta=0.4631857945`.

Two fitted parameters; no extra state, no extra linear PWA region (but one nonlinear switching surface). Continuous and C1 across the new surface, provided the baseline is continuous.

When `u` is strictly in the first original C-region and `s=l@y>theta`, `s'=a+b*s+c*s²` with `a=l@b(I)+k*theta²`, `b=lambda-2*k*theta`, `c=k`. Orthogonal modes evolve linearly: `p'=A0*p+(Id-vslow*l) b(I)`. Thus this **part of the regional flow** has an exact formula via two small matrix exponentials. When `s<=theta`, it returns to the affine flow. **The entire switched system is not one globally elementary flow**: original C-thresholds and the new s-threshold require event detection. The modal Riccati formula applies only up to the earliest crossing.

## Constant I, late periods, seconds

| I | LR target | Base 4-region PWA | Modal quadratic, independently DOP853 | Ca minimum | Ca maximum |
|---|---:|---:|---:|---:|---:|
| 0.53 | 46.00 | 40.00 | 45.80 | 0.0278 | 1.3000 |
| 0.55 | 35.75 | 35.25 | 36.00 | 0.0277 | 1.3000 |
| 0.70 | 22.75 | 23.25 | 23.30 | 0.0285 | 1.3000 |
| 1.00 | 17.50 | 17.25 | 17.20 | 0.0300 | 1.3260 |
| 1.12 | 17.75 | 17.00 | 17.00 | 0.0313 | 1.3447 |

At I=1.13 the tested trajectory converges to rest rather than sustaining a late large cycle. 620s runs, DOP853 rtol=2e-9, atol=2e-11, max_step=.12, from (C,h,r)=(1.3,.8,u0).

## Activation fraction of late observations

| I | Active % | Active outside first C-region % | In first C-region % |
|---|---:|---:|---:|
| 0.53 | 31.3 | 4.3 | 83.7 |
| 0.55 | 19.9 | 4.9 | 79.0 |
| 0.70 | 0.0 | 0.0 | 67.0 |
| 1.00 | 0.0 | 0.0 | 50.3 |
| 1.12 | 0.0 | 0.0 | 35.1 |
| 1.13 | 0.0 | 0.0 | 0.0 |

## Formula check

`check_exact.py` compares exact Riccati + transverse affine propagation with DOP853 over short intervals. Errors stay around 1e-15 **if both the original C-region and the active/inactive quadratic regime remain unchanged**. When the C-threshold is crossed, applying the original region formula beyond the event gives an error; this is expected. This is not a complete validated event-driven implementation or performance benchmark.

## Limitations

- The calcium maximum reaches about 1.345 µM in high-IP3 trajectories, above the prior 1.3-µM guard. This is an inherited failure, not resolved here.
- No independently optimized peak shape, dynamic-IP3 impulses, SIC/NEST speed benchmark, bifurcation continuation, multistart basin study, or global proof of the limit cycle.
- k and theta fitted using the five listed constant-IP3 periods and a rest requirement at I=1.13; these are *training*, not held-out measurements.
- The quadratic term is globally applied when s>theta, including short excursions outside the first C-region (~4-5% of late time for I=.53,.55); **there the simple first-region Riccati decomposition does not apply**. One can integrate as nonlinear ODE or construct a new compatible regional exact solver; no global cheap exact flow is claimed.

Files: `experiment.py` (fit, RK4 and independent DOP853), `check_exact.py` (local formula check), `results.json`, `periods.png`, `run.log`, and a copy of `validated.json`, `augmented_results.json` and `search_slow.py`/`run.py` from the baseline.
