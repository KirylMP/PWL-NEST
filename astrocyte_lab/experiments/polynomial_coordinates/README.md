# Polynomial calcium coordinate: Taylor versus Chebyshev

For AM, AFM and FM independently, approximate the archived mixed coordinate
`z=Phi(C)` by a polynomial of degree 2–6. A Taylor polynomial uses derivatives
at `C=0.5 µM`. A Chebyshev least-squares polynomial uses 2048 Chebyshev nodes
on `[0.001, 2/1.185] µM`. Both are saved as coefficients in powers of
`u=(C-0.5)/0.5`; they are the same polynomial function class, but the
coefficients are obtained by different approximation rules. Degree is the
highest power, not the number of coefficients.

Each polynomial is **fixed after approximation**. The two-threshold PWA
parameters are then refitted: no fold or Poincare condition, unrestricted
hidden q, fixed physical calcium thresholds, rest at I=0 and two fixed LR
input values at threshold crossings. Training uses seven reference cycles,
period and normalized phase-aligned waveform errors, plus a maximum-calcium
term for FM. The coefficients of the polynomial are not jointly optimized.
Taylor and Chebyshev are therefore compared as *ways to approximate the
existing mixed coordinate*, not as different polynomial expressiveness.

Derivative positivity is checked on the **entire declared interval** by
finding all roots of the derivative's derivative and checking them and the
endpoints. Nonmonotone maps are rejected and reported; the software does not
quietly alter the requested polynomial. No claim is made below 0.001 µM or
outside the domain. Predictions outside the interval are invalid. Error of
the map against the original mixed transformation is saved as `max_abs_z_error`
and `rms_z_error`, in transformed units, not calcium units. For accepted maps,
the inverse used in fitting and validation is 52-step bisection. It does not
have a general closed form and has not been optimized for NEST. This run does
not benchmark speed or generate a NEST module.

For each viable map, two candidate sources (saved mixed and local physical)
provide initial dynamic parameters; each uses two starts by default. These
are starting guesses only, not equivalent models under the changed coordinate.
Each group's best fit is chosen by training loss, then validated by DOP853 at
seven training and four held-out inputs. `comparison.png` in each selected
folder plots extrema, period and a waveform. The validation is a finite scan;
it does not determine the full cycle count, unstable branches or the fold.

By default the mixed FM input is the older archived candidate, because the
final snapshot was server-only. Set `FM_MIXED_MODEL` to the final server
`parameters.json` to use it; the exact source SHA-256 is recorded. If it is
changed, the polynomial is fitted to that file, but historical validation
statistics cannot be assumed unchanged.

Run from repository root with dependencies NumPy, SciPy, Matplotlib and g++:

```bash
WORKERS=16 POLY_STARTS=2 bash astrocyte_lab/experiments/polynomial_coordinates/run_server.sh
```

Defaults are `POLY_DEGREES="2 3 4 5 6"`, `POLY_MODES="am afm fm"`,
`POLY_POWELL_MAXFEV=6000`, `POLY_LS_MAX_NFEV=400`, `POLY_DURATION=350`,
`POLY_VALIDATION_DURATION=450`, `POLY_DT=0.01`. `POLY_OUT` sets the output
folder. `POLY_STARTS` is per seed; there are two seed models per accepted map.
Degree 2–8 can be specified. Using fewer starts or degrees gives a cheaper
exploration, not evidence that no better candidate exists.

The runner prints `RUN_DIR` at start. It saves `summary.json`, `results.csv`,
`run.log`, maps, per-start logs and candidates, and for every selected fit
`selected/parameters.json`, `selected/validation.json` and
`selected/comparison.png`. `results.csv` includes training/holdout validity
counts and errors; compare average errors only when validity counts agree.
Monotone maps with no valid candidate or failed validation make the runner
exit nonzero after writing the summary. Rejected nonmonotone maps are recorded
as such rather than being treated as optimization failures.
