# Fixed-fold optimization in log and square coordinates

This exploratory experiment starts from the six neutral-orbit candidates in
`FIXED_FOLD_COORDINATES.md`. It fits seven input values using the earlier loss
scales: calcium minimum / 0.03 µM, calcium maximum / 0.08 µM, period / 5 s,
and RMS peak-aligned amplitude-normalized shape / 0.1. Here the LR target
waveform comes from 256 samples, reduced to 64; PWA extrema and the position
of its peak are initially estimated on 41 samples per trajectory segment.
Therefore the loss values are approximate and should not be directly equated
with a run using exact extrema and phase alignment.

The zero-input rest and both equilibrium-switching inputs are imposed by the
field parameterization. Each objective evaluation numerically solves the four
matching equations and log(M)=0 at the prescribed fold input for one pivot
parameter and four flight times. Failed solves, changed itineraries, violations
of the three regional stability constraints, and changes exceeding a local
branch trust region are rejected. The latter is an explicit search restriction,
not a proof that other branches do not exist. Some rejected trials give a
constant search barrier; `xtol` termination is not evidence of a global
optimum or even high-accuracy local convergence. Stated positive calcium for
the square coordinate is a readout property, not an upper bound or a guarantee
of biological validity.

Starting candidates and optimization settings are stored in each report. Two
starts were attempted for AM log, AM square, AFM log and FM log; one start for
AFM square and FM square. `selection.json` selects the smallest training loss,
not the best held-out result, and records the exact candidate hash. These are
exploratory candidates; they do not replace models used for the manuscript.

Independent DOP853 integrations validate the sampled cycles and test closure of
the prescribed neutral orbit at maximal steps 0.03, 0.01 and 0.003 s. Four
held-out points and 256-point cycle sampling are reported separately in each
`candidate_verification.json`; this is finite-time numerical evidence, not a
guarantee of stability or completeness. `fold_audit.json` has 501 samples per
segment, finite-difference nondegeneracy diagnostics in the matching system,
and independent closure. Discontinuous regional Jacobians can make integrated
log(M) converge slowly; the maximum discrepancy across these finite tests is
approximately 1.2e-7 for the selected candidates. Square-coordinate training
cycles must stay on the positive u branch during the inspected orbit, but u can
cross zero outside it.

`dynamic.npz` and `dynamic.json` give an independent 400 s piecewise-constant
input replay, sampled every 0.05 s with exact upward threshold events. The old
physical candidate is included, not an independently integrated LR trace.
In particular the AM log fit produces 1 event in this protocol versus 8 for
the physical candidate, while the AM square fit reaches around 3.09 µM in a
transient. The FM square candidate has a fold maximum around 2.62 µM, above the
2/1.185 µM upper bound of the LR closed-pool parameterization. These candidates
are not validated network models, and no NEST performance claim is made.

From the repository root, with NumPy, SciPy and Matplotlib installed:

```bash
export PYTHONPATH="$PWD/scripts:$PWD/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m unittest discover -s tests -p test_transformed_fold_optimization.py -v
python scripts/optimize_fixed_fold_cycles.py --case AM --kind square \
  --candidate results/fold_anchors/AM/square/seed_3.json \
  --output results/my_new_AM_square --max-nfev 80
python scripts/verify_transformed_fold.py results/my_new_AM_square/candidate.json
python scripts/audit_cycle_optimization.py
python scripts/diagnose_cycle_optimization_v3.py
python scripts/plot_cycle_optimization_v3.py
```

The last three commands overwrite the stored v3 audit, dynamic and figure
files; save any previous results first. `report.json` explains optimization
termination, attempted routes and failures; `summary.csv` reports hold-out
errors separately from training losses.
