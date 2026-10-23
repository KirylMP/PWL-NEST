# Four-region FM surrogate with one latent state

Reference: `PWL-NEST/work/models/continued_shape_candidates/FM_control.json`, original eight-region PWA, NOT independently validated against Li–Rinzel here. Source four-region compression `validated.json`.

Equations (in affine transformed coordinates `u=(C-Cref)/S`, `v=(h-href)/HS`):

    du = original_compressed_du(u,v,I) + a*(r-u)
    dv = original_compressed_dv(u,v,I) + b*(r-u)
    dr = alpha*(u-r) + d*v + e*(I-1.12)

Coefficients: a=-0.2753418481, b=-0.0678852557, alpha=2.3174597906, d=0.1613021108, e=0.0974947124. Hidden initial condition `r(0)=u(0)`. The augmented system is affine inside each of FOUR original C regions; with dynamic exponentially decaying I it is a 4D state system with affine regional flow. All parameters are **numerical search results**, not certified.

Model does NOT automatically preserve original equilibria (`d*v + e*(I-1.12)` moves r equilibrium). Cross-validated against 8-region PWA autonomous behavior at fixed I; not against original LR. Differential evolution uses I=0.53,0.55,0.7,1.0,1.12 and high-I=1.13 in objective, so 1.13 is not held out. Other I points and other initial conditions diagnostic; independent future tests required.

Optimization: scipy differential_evolution, seed 42, 32 generations, population 45, 5 optimized parameters, RK4 0.05 s with post-fit 0.025 s. Validation trajectories up to 700 seconds and physical bounds. Period and oscillation status from finite windows; no bifurcation certification. Computational speed relative to original LR/NEST not measured.

`augment_search.py` optimizer and numerical model, `check_augmented.py` validation, `augmented_results.json` optimization and period table, `augmented_validation.json` physical bounds and initial-condition checks, `augmented_periods.png` chart, `augment_log.txt` detailed output.
