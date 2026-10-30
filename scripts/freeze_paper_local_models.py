#!/usr/bin/env python3
"""Collect locally selected models and validations without changing the fits."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'paper/local_results'
FIT = RESULTS / 'physical_no_fold_multistart'
DEST = RESULTS / 'models'
manifest = {}
for mode in ('am', 'afm', 'fm'):
    for coordinate in ('physical', 'mixed'):
        comparison = FIT / mode / 'comparison' / ('new_rank_01' if coordinate == 'physical' else 'frozen_mixed')
        source = comparison / 'parameters.json'
        validation = comparison / 'validation.json'
        if not source.is_file() or not validation.is_file():
            raise FileNotFoundError(f'Missing completed fit or validation: {comparison}')
        dst = DEST / f'{mode}_{coordinate}'
        dst.mkdir(parents=True, exist_ok=True)
        for p in (source, validation, comparison / 'comparison.png'):
            if p.exists():
                shutil.copy2(p, dst / p.name)
        manifest[f'{mode}_{coordinate}'] = {
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'source': str(source.relative_to(ROOT)),
            'validation': str(validation.relative_to(ROOT)),
            'note': ('Earlier archived mixed FM; final frozen server model unavailable.'
                     if (mode, coordinate) == ('fm', 'mixed') else
                     'Selected local fit, minimum training loss among six starts.'
                     if coordinate == 'physical' else 'Previously archived mixed model.')}
DEST.mkdir(parents=True, exist_ok=True)
(DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Saved', DEST)
