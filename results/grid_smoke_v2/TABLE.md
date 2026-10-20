# Coordinate grid: results available so far

Cells select models by independently verified TRAIN loss only. Test errors are diagnostic; failed/unrun cells stay empty. Different numbers of completed restarts must be reported with any table.

## AM: held-out C RMSE (uM)

| C / h | logit | cube |
|---|---:|---:|
| sqrt | 0.018547 | 0.0183 |
| sqrt_tails | 0.018547 | 0.0183 |

## FM: held-out C RMSE (uM)

| C / h | logit | cube |
|---|---:|---:|
| sqrt | 0.11673 | 0.11167 |
| sqrt_tails | 0.11652 | 0.11279 |

Own stable rest, 20 s input delay; C/h never clipped. Raw causal and phase-aligned errors are stored separately. No full acceptance, speed benchmark or NEST claim.

See best/ for fitted models and metrics; trials/ for every restart including failures; status.json for process state.
