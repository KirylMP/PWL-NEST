# Lightweight LR three-focus fitting experiment

Standalone snapshot of the three-focus, fixed-fold return-map experiment from PWL-NEST. Full project, article, historical results, and provenance remain on the original branches. Two tracked reference candidates live under results/anchored_three_main and the three original model parameter files under paper/data/physical_models/fixed_fold. New results/ are ignored.

Requires Python with NumPy and SciPy; no NEST is used. With your existing environment activated:

```bash
export PYTHONPATH="$PWD/scripts"
python -m unittest discover -s tests -p 'test_focus*.py' -v
python -u scripts/run_focus_fit_batch.py --seeds 1 --workers 3 --max-nfev 1 --joint-nfev 5 --output results/focus_fold_smoke
python scripts/summarize_focus_fit.py results/focus_fold_smoke
python -u scripts/run_focus_fit_batch.py --seeds 16 --workers 16 --max-nfev 80 --joint-nfev 250 --output results/focus_fold_main
python scripts/summarize_focus_fit.py results/focus_fold_main
```

See docs/focus_fold_fit_protocol.md for details and numerical limitations. Missing AFM/FM starts may be rejected or unresolved; this is not a proof of nonexistence.
