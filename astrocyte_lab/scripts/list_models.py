#!/usr/bin/env python3
"""List catalog models and their fit state."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'models'
models = sorted(root.glob('*/metadata.json'))
if not models:
    print('No models yet. Run fit.sh or import_run.py.')
for path in models:
    try:
        data = json.loads(path.read_text())
        print(f'{path.parent.name}  stage={data.get("stage")}  loss={data.get("loss")}  lower={data.get("fm_lower_type")}  fold={data.get("fold_constrained")}  plot={data.get("has_bifurcation")}')
    except (ValueError, OSError) as exc:
        print(f'{path.parent.name}: cannot read metadata: {exc}')
