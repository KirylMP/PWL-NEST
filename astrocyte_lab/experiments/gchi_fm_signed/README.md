# FM fitting on a server

Run `python run_server.py` after installing numpy, scipy and a C++ compiler.
The model has 3 thresholds in C, asinh glutamate input with optimizable scale,
free hidden anchor coordinates, fixed physical anchors, and hard spectral-type
constraints. Each worker has a separate random seed, checkpoint and log.

Set `FIT_WORKERS`, `FIT_RANDOM_STARTS`, `FIT_MAXFEV`, `FIT_SEED_BASE`,
`FIT_DT`, `FIT_STEPS`, `CXX`, `FIT_RUN_DIR` through the environment.
By default the fast evaluator uses dt=0.02 s, 60000 steps (1200 s), checks
positivity and approximate full-state closure; its output is rechecked using
DOP853 for 1600 s. This is a finite search with one initial point per input;
failure to find a cycle is not proof of absence. Parameters remain preliminary.

Each worker writes its own checkpoint.json. To resume manually in that worker:
`FIT_RESUME=checkpoint.json python fit_free_q.py`. Keep its files together.
`summary.json` and `review.zip` are written only after workers complete.
