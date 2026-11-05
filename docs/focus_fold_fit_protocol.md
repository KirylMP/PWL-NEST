# Fixed-fold fit: stable lower zone, two foci

## Anchors and spectrum

Use the three-zone continuous PWA system and the original `fixed_fold` source and reference targets. For each AM, AFM, FM fix the equilibrium at I=0, the two equilibrium threshold-crossing points `(C_L,I_1)` and `(C_R,I_2)`, and the fold input `I_*`. Four coefficients are eliminated by linear equality constraints; `a,b` and threshold positions remain fixed. One additional pivot is solved with the fold's segment times by the four matching equations and `log(M)=0`. The reduction by one free coefficient is local and conditional on the solve and its Jacobian; numerical failure does not prove nonexistence. Check transversality and nondegeneracy separately.

Regional conditions are `T_L<0, D_L>0` (stable node or focus), `T_M>0, T_M²-4D_M<0` (unstable focus), and `T_R<0, T_R²-4D_R<0` (stable focus). Thus a lower stable node is explicitly allowed. The middle and upper focus tests use a small numerical margin; screening has its own configurable margin. The old three-focus constraint and its outputs are historical and have not been modified.

## Loss and acceptance

The fitted loss has only four terms per training input: `((Cmin−CminLR)/0.03)^2 + ((Cmax−CmaxLR)/0.08)^2 + ((T−TLR)/5)^2 + mean((shape−shapeLR)^2)/0.1²`. Calcium is in μM and time in seconds; shape is dimensionless, with maxima aligned, time normalized to period and amplitude normalized to [0,1]; 64 phase samples are used. A value of 0.1 is a configurable numerical scale, not an uncertainty estimate. There is no model-coefficient regularization in this fitting loss. A large residual for an invalid trial is a numerical rejection marker and is never retained as the loss of an accepted candidate. Feasibility seed-finding uses a separate numerical merit function.

At training points, accept only numerically found stable cycles with sampled positive calcium, matching and region/route checks. The return solver supports lower–middle, middle–upper, and lower–middle–upper routes; finite sampling and finite starts are not proofs of first crossing or completeness. Reject a nonphysical candidate; do not add a negative-calcium term to the loss. Sampled positivity is not a proof between samples. The saved fixed-fold orbit must pass its own sampled checks; convergence and nondegeneracy are numerical, not automatic proofs.

A theory of PWA oscillators can establish a cycle under its complete hypotheses; it does not determine its threshold-crossing itinerary, guarantee positive calcium, or guarantee a numerical root finder converges. Distinguish no admissible root found, a rejected numerical cycle, and mathematical nonexistence.

## Running

```bash
export PYTHONPATH="$PWD/scripts"
python -m unittest discover -s tests -p 'test_lower_stable.py' -v
python -u scripts/run_focus_fit_batch.py --seeds 1 --workers 3 --max-nfev 3 --joint-nfev 100 --output results/lower_stable_smoke
python scripts/summarize_focus_fit.py results/lower_stable_smoke
python -u scripts/run_focus_fit_batch.py --seeds 16 --workers 16 --max-nfev 80 --joint-nfev 250 --output results/lower_stable_main
python scripts/summarize_focus_fit.py results/lower_stable_main
```

Each start writes its own status, log, and fitted report when available. AM starts near one known feasible candidate; AFM/FM attempt to find individual feasible seeds. Failure at any stage is reported without a claim of nonexistence. Existing per-start folders are skipped, not resumed; use a new output directory for an independent batch. Holdout has four inputs and is excluded from loss; `4/4` counts found cycles, not accuracy. Optimizer `xtol` is not proof of global convergence, and near-zero discriminants require caution. Recheck fitted candidates and compare holdout errors, not just the loss.

On the local test with one start, three objective evaluations and 100 initialization evaluations, all AM/AFM/FM completed with 4/4 holdout cycles, but all hit their short optimization budget. This is a smoke test, not a fitted result. The AFM and FM middle discriminants were approximately −0.0105 and −0.0119, so proximity to the focus–node boundary remains a concern.

## Continue until a candidate count is reached

`run_until_candidates.py` continues an existing compatible batch directory. It scans previous completed reports, and launches new unique seed numbers only for cases below the requested target. A counted candidate has a completed report and candidate, seven positive-calcium stable training cycles, a numerically matched neutral fold with positive sampled calcium, and the required spectral signs and focus conditions. The four holdout points do not enter the count or optimization; report their results separately. Candidates can be near-duplicates, and optimizer termination is not a proof of convergence or of global optimality.

The cap `--max-new-attempts` is per case and per invocation. The runner returns a nonzero status if the cap is reached before the target. A new invocation can continue with fresh seed numbers; existing files and logs are never overwritten. Stop with Ctrl+C; queued jobs are cancelled, active process groups terminated, and existing files remain. An interrupted/incomplete directory is not counted and is not resumed as an optimizer checkpoint. Never run two controllers on the same output directory simultaneously.

```bash
python -m unittest discover -s tests -p 'test_until_candidates.py' -v
python -u scripts/run_until_candidates.py --output results/lower_stable_main \
  --target 16 --workers 16 --max-new-attempts 128
python scripts/summarize_focus_fit.py results/lower_stable_main
```

Settings for the new tasks come from the existing `manifest.json`; mismatched batch scope is rejected. AFM and FM start generation still uses the existing random perturbation and local fold solver, not uniform exploration of all possible fields. Numerically unresolved or rejected starts count toward the cap, not toward the candidate target. Positivity, return routes and fold are checked numerically with finite sampling.
