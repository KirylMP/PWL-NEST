# Constant-IP3 accuracy and speed audit

`benchmark_accuracy_speed.py` measures each NEST implementation against its own independent DOP853 trajectory, then alternates unrecorded LR/PWA timings. The LR–PWA modelling discrepancy is a separate question. The settings and module hash are saved with the results. `reference_boundary_free=false` means the continuous LR reference is not sufficient to emulate NEST's calcium clipping after each step. Fine/coarse reference disagreement is an empirical convergence check, not a rigorous error bound. Sampled maxima can miss peaks between samples.

The default cases are AM/AFM/FM, IP3 = 0.2/0.6/1.0 µM, simulation steps 0.05/0.1/0.2 ms, 300 s, 32 cells, 1 thread, one warmup and three paired repetitions. Recorded error uses a separate single cell and 50 ms recording; timings use no recorder. Each system starts in the numerical state from the selected model's rest target; LR h and PWA q are distinct variables even though their initial numbers coincide. Recordings are not counted in timing. Repeat on an otherwise idle server, and report elapsed time and reference disagreement along with error. Do not pool across models, inputs or resolutions.

In the upstream [NEST 3.10 source](https://github.com/nest/nest-simulator/blob/v3.10/models/astrocyte_lr_1994.cpp), the LR integrator is GSL RKF45 with hardcoded absolute control 1e-3 and relative control zero; this is a local criterion, not a trajectory error guarantee. In the stock model there is no status field to tune that value. The resolution sweep tests the simulation-step dependence and is not an independent tolerance sweep. The selected PWA module analytically propagates inside each region and searches detected exits using bisection; its endpoint-based check may miss an exit and re-entry within one simulation step. Check the server's NEST version and actual module hash before applying these statements.

Run in the repository root after the ongoing benchmark finishes. Use a fresh output directory:

```bash
export PYTHONPATH=src:scripts
python -m unittest discover -s tests -p 'test_accuracy_speed.py' -v
python scripts/benchmark_accuracy_speed.py --cases AM --inputs 0.6 \
  --steps-ms 0.1 --duration-s 2 --interval-ms 50 --nodes 2 \
  --repeats 1 --warmup 0 --output results/accuracy_speed_smoke
python scripts/summarize_accuracy_speed.py results/accuracy_speed_smoke
python -u scripts/benchmark_accuracy_speed.py --output results/accuracy_speed_main
python scripts/summarize_accuracy_speed.py results/accuracy_speed_main
```

The smoke test checks plumbing only. Consider convergence on long and near-boundary oscillations separately; report missing pairs, failures and reference errors rather than treating them as zero. For a real GSL tolerance comparison an independently verified configurable LR implementation or a dedicated NEST build is required, and overhead and numerical equivalence must be assessed.
