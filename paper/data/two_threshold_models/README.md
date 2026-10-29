# Two-threshold models: accuracy and NEST timings

## Files

- `cycle_errors.csv` — 28 rows: training and held-out results for four main model variants (physical/mixed calcium coordinates × fold fixed/not fixed) at K_ER 0.05, 0.075, 0.10 µM, plus the best fractional and rational FM variants. Each row contains the number of inputs (`total`) and the number of detected valid cycles (`valid`), and means over valid cycles only. `source_file` identifies the JSON in `sources/`.
- `mixed_fold_locations.csv` — previously searched numerical fold candidates for the frozen mixed models versus Li–Rinzel, with residual diagnostics and number of successful starts. These models were fitted without fixing the fold. Nearby stable and unstable branches and numerical convergence remain to be checked; finite search is not exhaustive.
- `network_timings.csv` — 12 rows for the 10-neuron, 90-astrocyte all-to-all NEST experiment, AM/AFM/FM, physical/mixed, recording intervals 10/1000 ms. Contains medians, all three recorded runtime samples, model parameter SHA-256 and settings. This is full `nest.Run` wall time with SIC and recording, not isolated solver time.
- `earlier_identity_timings.csv` — 20 completed rows in the earlier identity-coordinate AM/FM NEST summary. This is a **different set of candidates and protocols**, included for historical comparison, not the 12 models in `cycle_errors.csv`. Empty fields mean not reported or not applicable. The source aggregate includes identity rows only; no power-coordinate timings are inferred from it.
- `sources/` — unmodified source JSON files or comparison files extracted byte-for-byte from the user-provided mixed-fold archive. `SHA256.json` records hashes of the source JSON files. Separate full candidate files and network traces are not included here.

For the 100-neuron scaling experiment, see `../astrocyte_neuron_ratio/`.

## Cycle error definitions

`mean_abs_Cmin` and `mean_abs_Cmax` are mean absolute extrema errors in µM; `mean_abs_period` is mean absolute period error in seconds; `mean_abs_shape_rmse` is the mean over cycles of calcium waveform RMSE in µM. Waveforms use normalized phase and peak alignment. `train` has seven target inputs and `holdout` four. Some data come from separately saved independent DOP853 runs with different historical fitting/validation settings; the same nominal inputs do not make the calculations bit-identical.

The fixed-fold mixed data come from `mixed_models_for_review.zip`; the three non-fixed physical candidates, their frozen mixed comparisons and fixed-fold physical baseline come from `physical_no_fold_20261029T183743Z_V23NCi`. Fixed-fold versions were constrained to an orbit that crosses both thresholds; in AM, LR's fold orbit does not cross the lower threshold. The non-fixed variants did not impose a fold or a Poincaré loss. The older fixed-fold implementation used stricter hidden-coordinate domain checks, whereas the later validation allows unrestricted q. All twelve main held-out rows report four of four cycles found.

The FM `fractional_rank_01` and `rational_rank_01` rows are experimental rational-coordinate models from `fm2_rational_20261029T180210Z_GS3f0n`, not the frozen mixed or physical candidates. In particular, rational holdout has only **3/4** valid cycles. Its error averages are not directly comparable to the 4/4 rows. No detected cycle in a finite run does not establish absence.

## Timing interpretation

`network_timings.csv` is the all-to-all directed 10-neuron/90-astrocyte network, `dt=0.1 ms`, 60 s simulation, 16 threads and three same-seed repeats. All astrocytes were recorded, and each directed neuron connection had an astrocyte. `lr_over_pwa` is the ratio of the separate median LR and PWA wall times. LR and PWA are independently fitted and may produce different activity; the difference combines dynamics, transformation, SIC, recording and network effects. Changing the recording interval changes the amount of recorded data and does not isolate coordinate inversion cost. See the unmodified network source JSON for graph hashes and timing scope.

The historical `earlier_identity_timings.csv` contains standalone cases with and without SIC and a network scenario; consult each row's `source`, node and SIC fields. Its C and SIC RMSE values describe the specific driven protocol, not phase-aligned cycle errors and not necessarily the final paper candidates. Never compare these error columns directly with `cycle_errors.csv`.
