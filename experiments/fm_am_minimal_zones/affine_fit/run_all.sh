#!/usr/bin/env bash
# Fixed-threshold affine AM/FM fitting, then NEST benchmark. No original candidate is overwritten.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
mkdir -p "$here/results"
run="$(mktemp -d "$here/results/affine_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
mkdir -p "$run/am" "$run/fm" "$run/nest"
printf 'RUN_DIR=%s\n' "$run"
trap 'printf "Stopped; all saved results are in %s\n" "$run" >&2' ERR
export AM_OUTPUT_DIR="$run/am" FM_OUTPUT_DIR="$run/fm"
export MAP_KIND=affine FM_COORDINATE=affine
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export POINCARE_WEIGHT_SCALE="${POINCARE_WEIGHT_SCALE:-100}"
export FM_TARGET_GRID_SIZE="${FM_TARGET_GRID_SIZE:-10}"
export FM_CYCLE_COUNT_POINTS="${FM_CYCLE_COUNT_POINTS:-17}"
export FM_CYCLE_COUNT_SEEDS="${FM_CYCLE_COUNT_SEEDS:-32}"
export FM_CYCLE_COUNT_WEIGHT="${FM_CYCLE_COUNT_WEIGHT:-600}"
export FM_FOLD_SOLVER_MAX_NFEV="${FM_FOLD_SOLVER_MAX_NFEV:-200}"
export FM_FIX_FOLD=1
export WORKERS="${WORKERS:-16}" VALIDATE_DENSE="${VALIDATE_DENSE:-1}"

bash "$root/run_fit.sh" am geometry orbits poincare phase closed validate
python3 -u "$root/am_two_thresholds/report_fixed.py" 2>&1 | tee "$run/am/report.log"
bash "$root/run_fit.sh" fm geometry poincare orbit powell closed validate
python3 -u "$root/fm_four_zone_fit/report.py" 2>&1 | tee "$run/fm/report.log"

# Refuse to benchmark anything but the freshly saved affine candidates.
python3 - "$run/am/candidate.json" "$run/fm/candidate.json" <<'PY'
import json,sys
for path,count in zip(sys.argv[1:],(2,3)):
    c=json.load(open(path))
    if c.get('transform') is not None or len(c['physical_thresholds_C'])!=count:
        raise SystemExit(f'Incorrect coordinate family or threshold count: {path}')
    print(f'Benchmark candidate: {path}, coordinate=affine, thresholds={count}',flush=True)
if not json.load(open(sys.argv[2])).get('fold_constrained'):
    raise SystemExit('FM did not pass the fixed-fold projection; benchmark is not started')
PY
export AM_CANDIDATE="$run/am/candidate.json" FM_CANDIDATE="$run/fm/candidate.json"
export NEST_RESULTS_DIR="$run/nest"
export NEST_PROTOCOL="${NEST_PROTOCOL:-work-pulse}"
export NEST_DURATION_S="${NEST_DURATION_S:-60}" NEST_VALIDATION_S="${NEST_VALIDATION_S:-30}"
export NEST_SINGLE_NODES="${NEST_SINGLE_NODES:-64}" NEST_MULTI_NODES="${NEST_MULTI_NODES:-64}"
export NEST_THREADS="${NEST_THREADS:-16}"
bash "$root/nest_benchmark/run_benchmarks.sh"
printf 'Completed: %s\n' "$run"
