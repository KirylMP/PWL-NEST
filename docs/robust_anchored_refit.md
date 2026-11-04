# Robust fold shooting for the three anchored models

This is a follow-up to `results/anchored_three_main` at commit `a041899`.
It refines the original three-zone AM, AFM and FM models. Fixed conditions:
the rest equilibrium at I=0, both original equilibrium threshold crossings,
the signs of regional traces/determinants and crossing orientations, and the
cycle fold input. A candidate is rejected if the numerical fold solve, route,
transversality, sampled training cycle or positivity checks fail. No fourth
zone is added to the candidate.

The earlier fold solver always used the original four flight times. It failed
on some small FM perturbations even though a valid nearby solution existed.
The new solver first tries the known fitted fold: AM seed 7, AFM seed 22, FM
seed 8 from `a041899`. If that fails, it tries the nearest successful fold
cached within the worker, then the original orbit. A numerical failure is not
a proof of nonexistence. Cache fallback can depend on evaluation order, and
multiple fold roots can exist; independently continue and check the selected
branch. The stationary-cycle loss and seven training/four holdout samples
remain the same. The numerical Jacobian uses central differences when possible,
one-sided differences otherwise, and logs columns with no feasible probe.

Normalized coefficient bounds expand from [-2,2] to [-4,4]; 80 is the default
maximum `least_squares` function-evaluation budget (not the number of actual
ODE simulations). Seed 0 starts at the previous best; seeds 1–7 draw nearby
and seeds 8–31 draw more broadly within [-3,3] before filtering for numerical
feasibility. Failed starts are retried up to 20 times, then reported as failed.
These are local/limited searches, not proofs of a global optimum. The pivot
coefficient is solved from fold equations and isn't bounded by that same box.

Each report stores the best valid evaluation (which may be a derivative probe)
and the final optimizer point/status separately, plus fold solver statistics,
anchor residuals, and held-out errors. `baseline_squared_residual` is the
previous best seed re-evaluated on the current machine, whereas the holdout
`baseline` trajectory is the original frozen model; don't conflate them.
Inherited old optimizer/fold metadata is removed from candidate files. The
fold's independent integration and finite-difference nondegeneracy are
diagnostics, not a global proof, and temporary negative calcium can still be
missed by finite sampling. Recheck branch continuity, convergence, and
time-dependent inputs before finalizing candidates.

Defaults are 3 cases × 32 seeds = 96 tasks, 16 concurrent workers, with
BLAS/OpenMP threads set to one per worker. This can take hours or longer.
Logs and outputs are separate per attempt. A rerun with unchanged settings
skips completed reports and starts a new attempt for incomplete jobs; it does
not resume an optimizer checkpoint. Ctrl+C cancels queued work and terminates
active workers; completed results remain. Failed random-start attempts are
listed in the log and `start_failure.json`. Do not run two controllers on one output.
The existing `summarize_lower_refit.py` reads these reports recursively.
