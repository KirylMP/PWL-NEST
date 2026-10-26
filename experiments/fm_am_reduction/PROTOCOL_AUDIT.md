# FM protocol implementation mapping (v3)

The historical v1/v2 experiment implemented only section 7 step 5. Its minima
were not minima of the full protocol. `run_fm.py` now enables the remaining cycle
objective and independent structural check. Historical results remain separate.

## Implemented

- Sections 2–3: LR equations/Jacobian, signed real cube root nullclines,
  normalized monotone Phi, continuous hinge graphs, positive rates, 26 joint
  parameters with seven fixed physical thresholds. A small-k Taylor expansion
  preserves the nonzero derivative at k=0; the old constant-limit branch
  incorrectly froze that derivative for tiny numerical perturbations.
- Section 4: finite-grid spectral dynamic programming with branch |dI| trapezoid
  weights. All available inter-sample gaps are used up to the configured grid
  cap. Moving a boundary within an empty gap leaves this empirical objective
  unchanged; no claim of a continuum/global optimum is made.
- Sections 5.1–5.3: physical C/I quadrature, realized regional trace/determinant,
  actual isolated equilibrium roots, Hungarian matching with missing/extra
  penalties. Positive-C roots outside the physical box are retained. Coincident
  affine nullclines raise a degeneracy flag. LR bracketing splits at critical
  points and checks near-tangent roots, including folds within numerical tolerance.
- Section 5.4: oriented first return (opposite crossing followed by the initial
  orientation), G=(P-s)/(s-h0), Gaussian-CDF variation, smooth positive part,
  regular edges only. Unstable roots have P'>1; stable roots have 0<P'<1.
  The multiplier is evaluated through integrated divergence and the ratio of
  normal velocities; this avoids cancellation for strongly contracting orbits.
- Section 5.5: all five losses, weights 1/10/10/30/30, averaged cycle error,
  unaveraged extra-cycle error, unrounded reference counters, common final epsilon.
- Section 6: DOP853 with section/domain events, trajectory sensitivities, moving
  section/time correction. Continuous-field switching has no saltation jump.
  Derivatives of fixed-coordinate Phi geometry are centered differences; the
  sensitivity ODE propagates their effect, including motion of z_j and z_sec.
  Geometry/equilibrium residual Jacobians use centered differences because the
  discrete matching itself is nonsmooth. Return sensitivities are checked against
  parameter perturbations, and representative PWA returns against RK45.
- Section 7: independent affine/log/power/mixed starts, geometry stage followed
  by full 26-parameter continuation .15 -> .05 -> .02, per-stage checkpoints,
  worker-failure records, common final score and raw return tables.
- Section 8: shifted-grid equilibrium/cycle spectra, sections through both models'
  equilibria and root searches for stable AND unstable cycles, neighbor-I seeds,
  deduplication by stability/period/extrema/common section intersections, structural signature error, matched extrema
  and periods with missing/extra counts, and interval-valued transitions.
- Section 10: reproducible snapshots, grids, tolerances, seeds, constraints,
  histories, losses, raw maps, failures, plots and logs within each experiment.

## Numerical feasibility and remaining limits

The source protocol leaves exact training grids, section locations, integration
budgets and parameter bounds to the run configuration. Defaults and all overrides
are saved. Training uses source equilibrium sections with common physical C,
h0 and u for both LR and candidate. Reference nodes with fewer than two regular
edges abort preparation; they are never silently converted to zero cycles.

A candidate must have at least two regular edges and at least half the reference
edge count per section. Otherwise its domain is labeled inadmissible, count values
are null, and a numerical feasibility residual is used to steer/reject the run.
On admissible domains this residual is zero and the objective is exactly the five
specified terms. This is a configurable numerical coverage heuristic, not a proof
that a section covers every orbit. Full missing-return reasons are saved.

Independent validation uses physical nullcline-based launch rules on several
sections, a denser grid, tighter tolerances and an extra neighbor-I correction
pass. No valid return intervals means unresolved cycle coverage, unless the PWA
has a uniformly signed divergence that excludes closed orbits. Tangencies,
near-degenerate roots and domain gaps are unresolved, not ordinary cycles.
Only regular hyperbolic comparisons enter the subset structural metric; the
full-grid metric remains null when any points are unresolved.

### Cycle audit correction (v4)

The v3 scalar root tolerance (1e-8 in u) was insufficient for strongly expanding
forward returns. For LR at I=87/81, it could accept an alleged unstable orbit
with |G| about .01. Different sections then produced different approximate
periods/extrema, and deduplication could count these as distinct cycles. The
validator now shoots repelling cycles backward and requires a small closing
residual, independently of root-bracket convergence. Common oriented section
crossings are recorded and used in deduplication; opposite stability types are
never merged. Expanding-time roots are diagnostics, not accepted orbits.
An upward sign crossing is sent directly to inverse-time shooting instead of
refining an ill-conditioned expanding root first. Value-only returns propagate
physical state and divergence; they no longer integrate an unused sensitivity
column. Parameter-sensitive fitting still propagates all required derivatives.
Value-only and sensitivity-enabled returns are compared in regression checks.

The original six-I training grid leaves (.65,.8) unsampled. Extra cycles in that
interval do not contribute to the sampled penalty. The smooth-count formula
and unaveraged extra-cycle formula are unchanged; training-section coverage and
I/u resolution require separate checks. `audit_cycle_counts.py` compares original
and denser grids, saves raw returns and checks observed orbits against training
section coverage. It does not silently change the objective or select a new fit.

At I=.8, selected (2), the original 17-u grid gives approximately (.5163,.5163)
instead of (1,1). Refinement on the same u interval resolves the missing positive
maximum of G. New `run_fm.py` runs use protocol id v4, 65 u nodes and add I=.75
and .78. All formulas/weights remain unchanged, including the unaveraged extra
term. The original v3 grid remains available for historical audit reconstruction.

Completeness of all equilibria/cycles, resolution of arbitrarily close roots,
and global optimization are not proved. Counts and deduplication depend on the
saved finite grids/tolerances. Transition values from scans are intervals.
Numerical sensitivity methods are local and can be ill conditioned at tangencies,
changes of event sequence or matching, and near bifurcations. A low training loss
is not acceptance; inspect the independent structural/period/amplitude reports.

AM, threshold reduction and K sweeps remain on hold for FM debugging. This patch
does not claim a completed minimum-complexity study, NEST integration or speedup.
