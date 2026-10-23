# Independent check: slow-mode Riccati pilot

## What was checked

The uploaded model and parameters were reproduced without the archive's Numba code. The augmented four-region FM PWA (three original C hinges, one hidden state) was compared with Li–Rinzel on all 10 saved training and 6 saved held-out dynamic-IP3 protocols. Initial states for impulse protocols were taken from the stored Li–Rinzel reference traces. Integrations used DOP853 (`rtol=2e-9`, `atol=2e-11`, `max_step=0.12 s`, output every 0.2 s), with identical jump times and amplitudes.

The two fitted correction parameters remained fixed at `k=-0.5956250890`, `theta=0.4631857945`. No fitting was done on dynamic protocols.

## Dynamic-protocol results

| Set | Model | Mean C RMSE (µM) | Peak counts matching LR |
|---|---|---:|---:|
| Training, 10 protocols | Hidden-state PWA | 0.1578 | 4/10 |
| Training, 10 protocols | Riccati correction | 0.1357 | 10/10 |
| Held out, 6 protocols | Hidden-state PWA | 0.1038 | 2/6 |
| Held out, 6 protocols | Riccati correction | 0.0860 | 6/6 |

Held-out mean C RMSE improved by 17.1%. Five of six held-out RMSEs improved and one (`tau=170 s`) worsened from 0.2188 to 0.2259 µM. Mean peak-time error among the count-matching held-out traces is 0.75 s for Riccati; this is not better than the baseline's 0.53 s computed over only its two count-matching traces. Thus peak-count recovery is encouraging, while precise phase alignment is not uniformly improved.

Across dynamic tests, the quadratic term was active in about 43% of training samples and 57% of held-out samples. Its active state was outside the first C region in about 0.8–0.9% of samples on average, with a maximum of 1.6% for any one protocol. The peak C reached 1.323 µM in these dynamic tests. The separately reported constant-IP3 sweep still reaches about 1.345 µM at high input.

## Exponential-IP3 analytical segment

Inside the first calcium region, the baseline field has `y'=A y + b0 + b1 exp(-t/tau)`. With `s=lᵀy`, `p=(Id-vl)y`, `lᵀv=1`, and while the quadratic hinge stays active:

```text
s' = a0 + a1 exp(-t/tau) + B s + k s²
p' = A p + (Id-vl)(b0 + b1 exp(-t/tau))
```

The transverse equation has an exact augmented matrix-exponential solution. Substituting `s=-(1/k) w'/w` gives `w''-B w'+k(a0+a1 exp(-t/tau))w=0`; removing the first derivative and changing variable `x=2 tau sqrt(-k a1) exp(-t/(2 tau))` gives a modified-Bessel equation for the tested sign `k a1<0`. The resulting local closed-form flow was compared with high-accuracy DOP853 at 11 points that remained in the same C region and active-hinge state: maximum absolute latent-state error was `3.4e-16`. A 12th test point crossed the C-region boundary; extending the first-region formula through that crossing produced `6.4e-4` error, as expected.

## Assessment

The dynamic input does **not** destroy the analytic structure of the slow region: its Riccati part has a closed-form Bessel solution, and transverse coordinates remain a matrix exponential. This is an exact segment formula, while the full hybrid trajectory still needs event location at C thresholds and the modal threshold. When the quadratic hinge is active outside the first C region, the current modal decomposition no longer closes; the prototype therefore uses a numerical fallback there. The prototype and its limits are described next.

## Event-solver prototype benchmark

A hybrid prototype was then implemented: exact augmented matrix exponentials in inactive affine regions, exact Bessel/modal propagation in the active first C region, DOP853 fallback while the correction is active outside that region, and Brent refinement after a switching plane is bracketed. On all six held-out trajectories, scans at 0.2 s and 0.025 s found the same numbers of switching events and produced C trajectories with RMSE at most 1.9e-12 µM between scan resolutions, for both the original and refitted parameter pair. This supports the 0.2-s scan on these trajectories, but does not certify that a very brief multiple crossing cannot occur elsewhere.

| Held-out protocol | Speedup vs DOP853 | C RMSE vs DOP853 (µM) |
|---|---:|---:|
| τ=9 s | 1.79× | 2.1e-9 |
| τ=29 s | 1.65× | 1.1e-8 |
| τ=58 s | 1.64× | 5.4e-9 |
| τ=105 s | 1.42× | 4.5e-7 |
| τ=170 s | 1.25× | 1.5e-7 |
| irregular | 0.97× | 3.6e-8 |

For the original parameters, mean speedup was 1.35× (median 1.37×); for the refit it was 1.41× (median 1.31×). Maximum C RMSE against DOP853 was 4.5e-7 µM for the original pair and 1.5e-7 µM for the refit. Timings are medians of warmed Python runs on this environment. Gains are modest and protocol-dependent; they exclude NEST integration, SIC, and compiled-code optimization.

## Limited dynamic refit

A 22-evaluation Powell search varied only `k` and `theta`, scored on all 10 dynamic training protocols, and did not evaluate held-out protocols until the candidate was selected. It moved `k` from -0.59563 to -0.55381 while leaving `theta=0.46319`. At high-accuracy DOP853, mean C RMSE moved from 0.13571 to 0.13522 µM on training and 0.08601 to 0.08525 µM on held-out tests; peak counts remained 10/10 and 6/6. The mean held-out peak-time error changed only 0.854 to 0.848 s. The fixed-I=0.53 period shifted from 45.8 to 44.8 s (Li–Rinzel: 46 s), worsening that low-input period fit. These sub-1% dynamic gains do not meet the previously discussed 15% meaningful-improvement bar, so the refit is retained only as a comparison and the original parameters remain the incumbent.

Reproduction scripts and raw metrics are in `dynamic_validation.py`, `check_exponential_flow.py`, `hybrid_solver_benchmark.py`, `fit_dynamic.py`, `refit_all_training.py`, `verify_refit.py`, `dynamic_protocol_validation.json`, `exponential_flow_check.json`, `hybrid_solver_benchmark.json`, `hybrid_solver_refit_benchmark.json`, `all_training_refit_results.json`, and `high_accuracy_refit_validation.json`. The unmodified upload is preserved under `input_archive/`.
