# Reapplied FM hidden-state experiments

This sidecar preserves the uploaded `fm_four_region_hidden_state.zip` and the
checks made from it without replacing the newer published FM candidates or the
analytic NEST event solver. The archive's own reference is an eight-region PWA;
the validation here compares its four-region model and hidden-state extension
directly with Li–Rinzel using the repository's saved FM protocols.

The uploaded model has three calcium knots, at approximately 0.04307, 0.16902,
and 0.38278 µM, defining four calcium regions. Its hidden state is initialized
as `r(0)=u(0)` and uses the five archived coefficients. The independent replay
is in `validation.json`; reproduce it with:

```sh
PYTHONPATH=src .venv/bin/python scripts/validate_uploaded_hidden_fm.py
```

The previous focused refit changed the five coefficients slightly and reduced
its selected training objective by about 1.1%; it did not resolve the 0.53 µM
period mismatch (40 s versus 46 s for Li–Rinzel). `followup_findings.json`
records that candidate and the separate one-hinge C, IP3, and r screens. Those
screens used central, fixed knot locations and are exploratory; none passed all
the constant-input cycle/rest checks. In particular, the I hinge disrupted
oscillations at 1.12 µM, while the r hinge created false oscillations at 1.13
µM. The C hinge worsened the tested trajectories.

The current published draft does not contain the older `nest_adapter` SIC
readout, so this replay reports calcium and h trajectories only. It makes no
SIC or speed claim. No tracked published model or solver was edited in porting
these experiments.
