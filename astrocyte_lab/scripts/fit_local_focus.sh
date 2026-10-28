#!/usr/bin/env bash
# Compare soft constraints from the same completed FM model.
set -euo pipefail
lab="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$lab/.." && pwd)"
fit="$repo/experiments/fm_am_minimal_zones/rest_refinement/run.py"
parent="${1:-${FM_PARENT_CANDIDATE:-}}"
if [[ -z "$parent" ]]; then
  parent="$(python3 - "$lab/models" <<'PY'
import json,sys
from pathlib import Path
paths=[]
for p in Path(sys.argv[1]).glob('*/parameters.json'):
    try:
        d=json.loads(p.read_text())
        if d.get('case')=='fm' and d.get('construction')=='full_fixed' and d.get('stage')=='closed' and d.get('fold_constrained') and 'local_focus' not in p.parent.name:
            paths.append(p)
    except (OSError,ValueError): pass
if not paths: raise SystemExit('No closed full_fixed FM model with constrained fold; provide parameters.json explicitly.')
print(max(paths,key=lambda p:p.stat().st_mtime_ns).resolve())
PY
)"
fi
parent="$(realpath "$parent")"
python3 - "$parent" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('case')!='fm' or d.get('construction')!='full_fixed' or d.get('stage')!='closed' or not d.get('fold_constrained'):
    raise SystemExit('Parent must be a closed full_fixed FM model with constrained fold.')
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export FM_RESUME_CLOSED=1 FM_LOCAL_FOCUS_WEIGHT="${FM_LOCAL_FOCUS_WEIGHT:-100}"
export FM_LOCAL_FOCUS_FLOOR="${FM_LOCAL_FOCUS_FLOOR:-0.1}"
export POINCARE_WEIGHT_SCALE="${POINCARE_WEIGHT_SCALE:-100}"
export FM_CYCLE_COUNT_POINTS="${FM_CYCLE_COUNT_POINTS:-17}" FM_CYCLE_COUNT_SEEDS="${FM_CYCLE_COUNT_SEEDS:-32}"
export FM_CYCLE_COUNT_WEIGHT="${FM_CYCLE_COUNT_WEIGHT:-600}" WORKERS="${WORKERS:-16}"
mkdir -p "$lab/runs"
printf 'PARENT=%s\n' "$parent"
failures=0
run_label="${FM_EXPERIMENT_LABEL:-local_focus}"
if [[ ! "$run_label" =~ ^[a-zA-Z0-9_]+$ ]]; then printf 'Invalid FM_EXPERIMENT_LABEL\n' >&2; exit 1; fi
for target in ${FM_LOCAL_FOCUS_TARGETS:-2 5 8}; do
  label="${target//./p}"
  run="$(mktemp -d "$lab/runs/fit_${run_label}_${label}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
  mkdir -p "$run/fm" "$run/logs"
  cp "$parent" "$run/fm/parent_candidate.json"
  cp "$parent" "$run/fm/candidate.json"
  export FM_LOCAL_FOCUS_TARGET="$target"
  printf 'RUN_DIR=%s target=%s weight=%s floor=%s\n' "$run" "$target" "$FM_LOCAL_FOCUS_WEIGHT" "$FM_LOCAL_FOCUS_FLOOR"
  args=(fm --coordinate mixed --construction full_fixed --fm-lower-type node --out "$run/fm")
  if ! python3 -u "$fit" "${args[0]}" closed "${args[@]:1}" 2>&1 | tee "$run/logs/fm_closed.log"; then
    printf 'Closed fit failed; retained files at %s\n' "$run" >&2
    failures=$((failures+1)); continue
  fi
  cp "$run/fm/candidate.json" "$run/fm/closed_candidate.json"
  if ! python3 -u "$fit" "${args[0]}" validate "${args[@]:1}" 2>&1 | tee "$run/logs/fm_validate.log"; then
    printf 'Validation failed for %s\n' "$run" >&2; failures=$((failures+1))
  fi
  if ! python3 -u "$lab/scripts/analyze_lower_fm.py" "$run/fm" 2>&1 | tee "$run/logs/fm_lower_analysis.log"; then
    printf 'Lower analysis failed for %s\n' "$run" >&2; failures=$((failures+1))
  fi
  python3 "$lab/scripts/import_run.py" "$run" || failures=$((failures+1))
done
if (( failures )); then printf 'Completed with %s failed stages; inspect runs and logs.\n' "$failures" >&2; exit 1; fi
printf 'All local focus fits finished.\n'
