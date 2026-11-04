# Network scaling experiment

Run in the same environment as `docs/constant_ip3_sweep.md`. The 100-neuron
network has all 9900 directed synapses without autapses, and each selected edge
gets one astrocyte. A fixed SIC weight per astrocyte means total feedback grows
with the number of astrocytes. This is not a fixed-total-gain experiment.
Seed controls both the nested edge subset and NEST's stochastic input; LR and
PWA use the same seed and selected-edge hash, although their recurrent spike
histories may diverge. Build/Prepare/Run/Cleanup and failure messages are
saved per run; diagnostics are separate, record a subset and report at least
three detected peaks rather than prove a converged cycle.

First run the small smoke test; NEST is not available in the development
environment, so this script has not yet been verified against the server:

```bash
export PYTHONPATH=src:scripts
python -m unittest discover -s tests -p 'test_lr_scaling_graph.py' -v
python scripts/benchmark_lr_scaling.py --cases AM --neurons 4 --counts 4 \
  --seeds 101 --repeats 1 --warmup 0 --duration 2 --sample 4 \
  --output results/lr_scaling_smoke
python scripts/summarize_lr_scaling.py results/lr_scaling_smoke
cat results/lr_scaling_smoke/summary.csv
```

Only after the smoke summary reports paired successful runs and diagnostics:

```bash
python scripts/benchmark_lr_scaling.py --output results/lr_scaling_main
python scripts/summarize_lr_scaling.py results/lr_scaling_main
```

The full default can take many hours: three modes, four network sizes, five
seeds, one warmup plus three timing repetitions and diagnostics for both
models. Files are flushed after each run so partial progress survives, but the
script does not resume an existing output directory. Do not treat the first
output as publication-ready without checking rates, IP3 ranges, peak counts,
and that the actual workload is appropriate. Threshold crossings include
transients. Startup requires `.build/pwl_astro_module.so`; rebuild if absent.
