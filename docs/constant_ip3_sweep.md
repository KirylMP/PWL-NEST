# Constant IP3 sweep for physical AM / AFM / FM

This experiment uses the runtime-configurable `astrocyte_lr_pwa` NEST module
and the three frozen physical no-fold candidates included here. It is a
different PWA implementation from the compiled per-candidate module used by
some earlier benchmarks; do not combine their timings as identical builds.
IP3 is initialized to the same constant as its baseline, with no input spikes.
Each matched pair starts with the same numeric C and hidden state (PWA q is not
the LR receptor h), taken from the candidate's rest target. Finite-time
convergence and multistability are not fully resolved by one initial state.

Install NEST 3.10 and project dependencies on the server; see `docs/nest_module.md`.
From the repository root:

```bash
export PYTHONPATH=src:scripts
python scripts/build_nest_module.py
python -m unittest discover -s tests -p 'test_constant_ip3_adapter.py' -v

# Small first smoke test before the full sweep.
python scripts/benchmark_constant_ip3.py --cases AM AFM FM --inputs 0.4 \
  --nodes 2 --duration-s 2 --transient-s 1 --repeats 1 --warmup 0 \
  --output results/constant_ip3_smoke
python scripts/summarize_constant_ip3.py results/constant_ip3_smoke

# Main run: defaults are 18 IP3 inputs, 32 cells, 300 s, 3 repeats, 1 warmup.
python scripts/benchmark_constant_ip3.py --output results/constant_ip3_main
python scripts/summarize_constant_ip3.py results/constant_ip3_main
```

Each output directory must be new. `runs.jsonl` contains individual timings,
errors, warmups and single-cell diagnostic traces; `summary.csv` retains
incomplete repeats and diagnostic failures. `manifest.json` includes the
module SHA256; each row records the candidate SHA256. Crossings per cell and
biological second include the transient. The short smoke cannot establish
oscillation or convergence. Run without profilers for timing; counters are
always enabled and their overhead is included. Constant IP3 is numerically
checked in the separate diagnostic runs. Verify all diagnostics and a smaller
time step before interpreting results. The script has not been run in NEST in
the development environment: the server smoke is required first.
