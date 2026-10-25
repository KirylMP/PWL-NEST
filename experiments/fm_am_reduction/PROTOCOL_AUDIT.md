# Implementation audit for the FM debugging run

The previous fitter implements the initial objective in section 7 step 5, not
the complete objective in section 5.5. A low training score does not establish
correct oscillations. More restarts alone cannot supply the missing constraints.

Corrected in the FM debugging version:

- Section 2.1 uses the real cube root for negative arguments too. Previously
  those valid nullcline samples were incorrectly dropped. At the canonical FM
  grid C >= .03 the radicand is nonnegative; this error chiefly affects other
  K values and smaller C. It does not explain the poor FM result alone.
- Sections 4 and 5.2 now use the same trapezoid weights in I per sampled branch.
  The old rank matching incorrectly connected branches around folds and the
  second stage used a different weighting. Sampling by I approximates |dI|
  along every branch; finite-grid enumeration can miss very narrow branches.
- Section 5.1 now uses trapezoid quadrature in physical C and I instead of equal
  endpoint weights. H uses the same finite compact grid as N.
- Section 7 now starts near affine, near logarithmic, power and mixed coordinates,
  fitting graph coefficients after choosing the coordinate. The old random
  starts perturbed eta after computing its graph coefficients.
- Removed deprecated NumPy array-to-scalar conversions. C limits now use C_0.
- Every completed trial and current winner is saved. Individual worker failures
  are logged without discarding other restarts. Fresh FM runs preserve old runs.

Remaining deviations and limitations:

- Sections 5.4, cycle loss and extra-cycle penalty, and epsilon continuation are
  NOT implemented. Config weights 30/30 have no effect on this pre-cycle fit.
- Section 6 sensitivities are not implemented: optimization uses numerical
  differences, which can be poorly conditioned at changes in equilibrium matches.
- Section 4 currently uses grid dynamic programming without continuous boundary
  refinement. Sections 8–9 are diagnostic, not acceptance or minimality proofs.
- Equilibria of LR are enumerated using sign-change brackets, which can miss
  tangent roots. Coincident affine nullclines need separate degeneracy analysis.
- Search bounds (coefficients, speeds, eta), optimization budget, sampled grids,
  and solver tolerances restrict the search. These are computational choices;
  the source protocol does not supply their values.
- Figures include stable/unstable/saddle equilibria, one-sided matrices at
  thresholds, and periodic oscillations observed from three initial states. Attraction is
  not certified by multipliers.
  No repelling cycles are enumerated. Unresolved trajectories and domain exits
  remain explicit. Transition positions are only finite-grid intervals.
- Validation figures use shifted I nodes and DOP853 in latent coordinates, not
  the training residual. They do not enter candidate selection. All equilibria
  and raw orbit results are saved, including failures and nonphysical roots.

Default new run: FM only, seven fixed thresholds, 64 restarts, 16 processes,
1500 least-squares function evaluations per restart. The budget counts reported
optimizer evaluations; numerical Jacobian evaluations add work. A restart that
hits the budget may still be saved; inspect success/status/optimality fields.
Old training losses are not comparable after these corrections.
