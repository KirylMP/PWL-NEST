#!/usr/bin/env bash
# Benchmark explicit catalog models; never silently select an older candidate.
set -euo pipefail
lab="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$lab/.." && pwd)"
if (( $# != 2 )); then
  printf 'Usage: bash astrocyte_lab/scripts/nest.sh AM_MODEL_DIR FM_MODEL_DIR\n' >&2; exit 2
fi
am="$(cd "$1" && pwd)/parameters.json"
fm="$(cd "$2" && pwd)/parameters.json"
python3 - "$am" "$fm" <<'PY'
import json, sys
from pathlib import Path
for case, path, expected in [('am', sys.argv[1], 2), ('fm', sys.argv[2], 3)]:
    p=Path(path)
    data=json.loads(p.read_text())
    if data.get('case') != case or len(data['physical_thresholds_C']) != expected:
        raise SystemExit(f'{p}: NEST currently requires {case.upper()} with {expected} thresholds')
    if data.get('stage') != 'closed':
        raise SystemExit(f'{p}: stage is not closed; choose a completed candidate')
PY
mkdir -p "$lab/runs"
run="$(mktemp -d "$lab/runs/nest_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
printf 'RUN_DIR=%s\nAM_CANDIDATE=%s\nFM_CANDIDATE=%s\n' "$run" "$am" "$fm"
trap 'printf "NEST stopped; retained files: %s\n" "$run" >&2' ERR
export AM_CANDIDATE="$am" FM_CANDIDATE="$fm"
export NEST_RESULTS_DIR="$run/benchmark" NETWORK_RESULTS_DIR="$run/network"
export NEST_DURATION_S="${NEST_DURATION_S:-60}" NEST_VALIDATION_S="${NEST_VALIDATION_S:-60}"
export NEST_PROTOCOL="${NEST_PROTOCOL:-work-pulse}" NEST_THREADS="${NEST_THREADS:-16}" NETWORK_THREADS="${NETWORK_THREADS:-16}"
bench="$repo/experiments/fm_am_minimal_zones/nest_benchmark"
bash "$bench/run_benchmarks.sh"
bash "$bench/run_all_to_all.sh"
printf 'NEST complete: %s\n' "$run"
