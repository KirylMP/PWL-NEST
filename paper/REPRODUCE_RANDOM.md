# Reproducing the selected physical-coordinate analysis

Working candidates AM 465, AFM 800 and FM 474 and their SHA-256 hashes are in `paper/data/random_selected/manifest.json`. The NEST benchmark data are in `results/two_inputs_*` and `results/ca_events_*` on branches `experiment/lr-two-inputs-light` and `experiment/lr-ca-spikes-light`. They are separate from this manuscript working tree; obtain the recorded files before regenerating the timing figures. Existing earlier experiments and figures are not overwritten.

From the repository root, with NumPy, SciPy and Matplotlib installed:

```bash
export PYTHONPATH="$PWD/scripts:$PWD/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/audit_random_local_hlb.py
python scripts/audit_random_selected.py
python scripts/continue_random_selected.py --steps 420
python scripts/continue_random_upper.py --steps 140 --fm-steps 420
python scripts/plot_random_continuation.py
python scripts/plot_random_bifurcation_comparison.py
python scripts/plot_random_selected.py
python scripts/random_selected_variable_input.py
python scripts/guard_variable_input.py
python scripts/plot_guard_variable_input.py
python scripts/check_random_am_ramps.py
python scripts/validate_random_nest_ivp.py
python scripts/validate_random_nest_lr_ivp.py
python scripts/plot_random_selected_nest.py
python scripts/plot_ca_spike_benchmark.py
python scripts/summarize_lower_guard.py results/guard_main
python scripts/plot_guard_paper.py
make -C paper english
make -C paper ru
```

The continuation uses finite guesses, step budgets, sampled residence checks and numerical integration; an unresolved endpoint is not proof of a branch ending. The AM variable-input negative-calcium response is a result, not clamped or excluded. The plotted spike-time differences pair ordinal events and should not be confused with a unique event-matching rule.

NEST 3.10.0 was reported by the server; the benchmark manifests preserve process settings and module hashes but did not capture the CPU model. The compiled wheel's exact source revision is not independently confirmed. Three repeats use the same deterministic scenario and measure timing variation, not biological variability.

## Exact input values and numerical sensitivity

Use the full-precision candidate files, not rounded printed coefficients. Their SHA-256 hashes and exact paths are in `paper/data/random_selected/manifest.json`; each selected report contains the fold flight times and all input points. Values below are quoted with enough digits to identify the runs; the JSON files are authoritative. Units are µM.

| Model | Fit inputs | Holdout inputs |
|---|---|---|
| AM | 0.377121658, 0.424917528, 0.448815462, 0.496611331, 0.520509266, 0.568305135, 0.592203069 | 0.401019593, 0.472713397, 0.544407200, 0.616101004 |
| AFM | 0.375914857, 0.449272975, 0.485952034, 0.559310153, 0.595989212, 0.669347330, 0.706026389 | 0.412593916, 0.522631093, 0.632668271, 0.742705448 |
| FM | 0.555546080, 0.611569384, 0.639581035, 0.695604339, 0.723615991, 0.779639294, 0.807650946 | 0.583557732, 0.667592687, 0.751627642, 0.835662598 |

`audit_random_selected.py` uses full-precision candidate coefficients and the fold solution in each report. It evaluates symmetric finite differences using `mapped` in `check_selected_return_map.py`, including 201 sampled residence points per segment and crossing direction checks. Failed solves or checks are recorded in `paper/data/random_selected/audit.json`; sampled checks are not a proof that no earlier crossing occurs. In AM, central input steps 1e-5 and 1e-6 fail to resolve the return map; decreasing the step does not remove floating-point sensitivity. The selected model artifacts are in the working tree and need a commit and stable repository revision for publication.
