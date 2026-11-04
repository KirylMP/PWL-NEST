# Many-start comparison of lower-zone refits

Run the same fitting script and fixed-fold constraints as the preceding
three-versus-four-zone experiment. The default adds seeds 2 through 33
inclusive: 32 new starts × three cases × two arms = 192 tasks. Previous seeds
0 and 1 remain in the earlier results and are not recomputed. Starts perturb
the same local parameter neighborhood; this is not a global parameter search.
Keep the same seven training points, four holdouts, 400 s simulation duration,
and 20 optimizer evaluations as before to make comparisons meaningful.

Use at most 12 worker processes by default. All common BLAS/OpenMP thread
counts are set to one per process. No NEST module is needed. A failed or
unresolved fit is not evidence that the optimizer found an optimum. Check
holdout errors, actual equilibrium position and stability, numerical fold
checks, transients, nonnegative calcium, and other branches before selection.
The objective includes soft rest and stability penalties. Its value includes
regularization, so also compare the separately recorded metrics.

Every attempt writes its log alongside an attempt directory. Finished runs
have a `report.json`; on relaunch with the same batch settings, these are
skipped and incomplete or failed tasks receive a new attempt directory. A new
attempt starts from scratch, not from its checkpoint. The number of workers
may be changed on restart. Never run two batch controllers against the same
output directory simultaneously. Using different case/seed ranges, optimizer
settings or duration requires a different output directory.

The batch controller exits nonzero if any tasks fail. Results still include
all completed tasks. Stop with Ctrl+C if needed; no report is written for a
task interrupted mid-fit. Review logs and restart the same batch after all old
child processes have exited. `summarize_lower_refit.py` finds completed report
files recursively and therefore supports the attempt directories.
