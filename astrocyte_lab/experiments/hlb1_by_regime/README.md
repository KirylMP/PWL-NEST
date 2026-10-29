# Two-threshold foci, lower HLB1 sign by regime

Checks six existing no-fold candidates. All must have stable focus / unstable focus / stable focus from lowest to highest region. At the lower crossing FM requires alpha>0; AM and AFM require alpha<0. A model satisfying the conditions is kept exactly as is; only failing models are refitted. Both the source JSON hash and regional diagnostics are preserved. Frozen models remain untouched.

With e=exp(x2), T0=-2 sqrt(D0) e, omega0=sqrt(D0)*sqrt(1-e²) and A=T1/(2 omega1)>0, alpha=A-e/sqrt(1-e²). Let cap=A/sqrt(1+A²). For FM use e=cap*sigmoid(v); for AM/AFM use e=cap+(1-cap)*sigmoid(v). Both enforce 0<e<1 with the intended alpha sign, subject to numerical precision; result checks independently reject an invalid sign. The parameter x2=log(e) is stored in the existing representation.

Nonzero sign and exact focus alone do not enforce a finite margin from alpha=0 or focus/node degeneracy; inspect alpha and discriminants. Positive/negative alpha predicts unstable/stable small cycles respectively for the generic continuous focus-focus HLB1; numerical finite boundary matching and DOP853 check actual local cycles, and lack of detection does not prove absence. No fold, count or Poincare loss. q unrestricted. Train and holdout settings are as in the previous no-fold fits. Existing FM uses peak weight 1, AM/AFM 0. Six restarts from each failing existing model and finite optimization do not prove optimality or infeasibility.

Outputs: `summary.json` includes `initial_checks`, `trials`, `winners`, and validations. Selected model folders include copied parameters, DOP853 comparison and lower-cycle audit. Refit results are saved separately; no automatic replacement of any source model.
