# Three-zone fit with fixed equilibria and cycle fold

This experiment fits AM, AFM and FM independently starting from frozen
physical-coordinate, two-threshold models. It uses no extra calcium threshold
in the exported model. A zero-strength dummy hinge is used internally to reuse
the affine anchor calculations and is removed from the exported candidate.

Four coefficients are eliminated by linear equalities: the original state of
rest is an equilibrium at I=0, and equilibria lie at both original thresholds
at the original specified input values. Region determinant and trace signs
and equilibrium-crossing directions are checked; the two outer regions must
be stable and the middle one unstable. Four travel times and a fifth coefficient
are solved numerically from the fold orbit closure and log(M)=0 at its original
I. The fixed-input fold must remain on the old both-threshold itinerary and
the sampled first-crossing and transversality checks must pass.

Seed 0 starts from the original coefficients; seeds 1 and above use independent
uniform random starts in a local parameter neighborhood. If a random draw has
no admissible fold or cycles, a seed tries up to 20 draws. The fit uses seven
training cycles, then evaluates four holdout cycles. The initial and best loss
include a small parameter regularizer. The best valid evaluated point may be a
finite-difference probe and not the last optimizer iterate. The adjusted fold
pivot coefficient is solved by the shooting equations and is not subject to
the same box as the explicitly optimized coefficients.

The equations and inequalities are enforced numerically, with residuals in
`report.json`. This is not a global proof or a guarantee of other branches,
the true fold amplitude, timing of response to time-dependent inputs, or
NEST performance. Samples can miss an earlier crossing or a negative C;
independent integration and finite-difference nondegeneracy are diagnostics.
Inspect failure logs and heldout cycles, and verify tolerances and continuation
before accepting a candidate. All exported candidate metadata for the old fit
are removed; the source hash and new fit evidence are saved in `anchored_fit`.

The default batch uses 32 seeds per case and up to 12 processes. It may take
hours. Thread counts for BLAS/OpenMP are one per worker. Completed reports are
skipped on relaunch with identical batch settings; interrupted/failed attempts
are preserved and restarted from scratch, not resumed from checkpoints. Never
run two batch controllers in the same output folder simultaneously. If some
seeds cannot find feasible starts after the allowed attempts, the batch ends
nonzero after other jobs finish; inspect their logs.

```bash
python -m unittest discover -s tests -p 'test_anchored_lower_fold.py' -v
python scripts/run_anchored_three_multistart.py --output results/anchored_three_main --dry-run
python -u scripts/run_anchored_three_multistart.py --output results/anchored_three_main
python scripts/summarize_lower_refit.py results/anchored_three_main
```
