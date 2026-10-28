#!/usr/bin/env python3
"""Copy completed model artifacts from a fit run into the model catalog."""
import argparse
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path, help='Run directory containing am/fm, or a base_nest directory containing fit/am and fit/fm')
    ap.add_argument('--allow-incomplete', action='store_true', help='Also import candidates from earlier stages, explicitly marking them incomplete')
    args = ap.parse_args()
    run = args.run.resolve()
    if not run.is_dir():
        ap.error(f'Run directory does not exist: {run}')
    fit = run / 'fit' if (run / 'fit').is_dir() else run
    run_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', run.name)
    catalog = LAB / 'models'
    catalog.mkdir(parents=True, exist_ok=True)
    imported = 0
    for case in ('am', 'fm'):
        source = fit / case
        candidate = source / 'candidate.json'
        if not candidate.is_file():
            print(f'{case}: no candidate in {source}; skipped')
            continue
        raw = candidate.read_bytes()
        data = json.loads(raw)
        if data.get('case') != case:
            raise ValueError(f'{candidate}: expected case {case}')
        if data.get('stage') != 'closed' and not args.allow_incomplete:
            print(f'{case}: stage={data.get("stage")!r}; skipped (use --allow-incomplete to import explicitly)')
            continue
        sha = hashlib.sha256(raw).hexdigest()
        name = f'{case}_{len(data["physical_thresholds_C"])}thr_{run_id}'
        dest = catalog / name
        if dest.exists():
            existing = dest / 'parameters.json'
            if existing.is_file() and hashlib.sha256(existing.read_bytes()).hexdigest() == sha:
                print(f'{case}: already imported: {dest}')
                imported += 1
                continue
            raise FileExistsError(f'{dest} already exists with different data; choose a differently named run')
        with tempfile.TemporaryDirectory(prefix='.import_', dir=catalog) as tmp:
            target = Path(tmp)
            shutil.copy2(candidate, target / 'parameters.json')
            for old, new in [('candidate_bifurcation.png', 'bifurcation.png'),
                             ('candidate_branches.json', 'branches.json'),
                             ('lower_bifurcation.png', 'lower_bifurcation.png'),
                             ('lower_bifurcation.json', 'lower_bifurcation.json')]:
                if (source / old).is_file():
                    shutil.copy2(source / old, target / new)
            analysis_log = fit / 'logs' / f'{case}_lower_analysis.log'
            if analysis_log.is_file():
                shutil.copy2(analysis_log, target / 'lower_analysis.log')
            optimizers = {}
            for path in sorted(source.glob('*_optimizer.json')):
                optimizers[path.name] = json.loads(path.read_text())
            metadata = dict(case=case, threshold_count=len(data['physical_thresholds_C']),
                stage=data.get('stage'), complete=data.get('stage') == 'closed',
                accepted=data.get('accepted'), loss=data.get('loss'),
                fold_constrained=data.get('fold_constrained'),
                fm_lower_type=data.get('fm_lower_type', 'node' if case=='fm' else None),
                has_lower_analysis=(target / 'lower_bifurcation.json').is_file(),
                has_bifurcation=(target / 'bifurcation.png').is_file(),
                parameter_sha256=sha, source_candidate=str(candidate), source_run=str(run),
                imported_utc=datetime.now(timezone.utc).isoformat(), optimizers=optimizers)
            (target / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
            target.rename(dest)
        imported += 1
        print(f'{case}: imported {dest}')
        if not (dest / 'bifurcation.png').exists():
            print(f'{case}: bifurcation plot is missing; source validation may not have completed')
    if not imported:
        raise SystemExit('No models imported. Check the run path and stage.')


if __name__ == '__main__':
    main()
