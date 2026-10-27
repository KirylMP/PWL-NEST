#!/usr/bin/env bash
# Run both FM fits and the selected AM fit in the existing research branch module.
set -euo pipefail
MYDIR="$(cd "$(dirname "$0")" && pwd)"
PWL_NEST_ROOT="${PWL_NEST_ROOT:-$PWD}"
PWL_NEST_ROOT="$(cd "$PWL_NEST_ROOT" && pwd)"
export PWL_NEST_ROOT
BENCH="$PWL_NEST_ROOT/experiments/fm_am_minimal_zones/nest_benchmark"
[[ -f "$BENCH/module/generate.py" && -f "$BENCH/run_benchmarks.sh" ]] || {
  echo 'Requires PWL-NEST research/fixed-threshold-minimal-zones branch.' >&2;exit 1;
}
python3 "$MYDIR/verify_decoder.py"
OUTROOT="${OUTROOT:-$MYDIR/results}"
mkdir -p "$OUTROOT"
AM_CANDIDATE="$MYDIR/models/AM_3zones.json"
export AM_CANDIDATE
for variant in 6 7; do
    export FM_CANDIDATE="$MYDIR/models/FM_4zones_${variant}.json"
    export NEST_RESULTS_DIR="$OUTROOT/FM_${variant}"
    mkdir -p "$NEST_RESULTS_DIR"
    echo "--- NEST case: AM_3zones vs FM_4zones_${variant} ---"
    (
      cd "$BENCH"
      # Default protocol comparable to old work benchmark; override env as needed.
      export NEST_PROTOCOL="${NEST_PROTOCOL:-work-pulse}"
      export NEST_DURATION_S="${NEST_DURATION_S:-60}"
      export NEST_VALIDATION_S="${NEST_VALIDATION_S:-30}"
      export NEST_RESOLUTION_MS="${NEST_RESOLUTION_MS:-0.1}"
      export NEST_SINGLE_NODES="${NEST_SINGLE_NODES:-64}"
      export NEST_MULTI_NODES="${NEST_MULTI_NODES:-64}"
      export NEST_THREADS="${NEST_THREADS:-16}"
      export NEST_REPEATS="${NEST_REPEATS:-3}"
      bash run_benchmarks.sh
    )
done
printf 'Saved results under %s\n' "$OUTROOT"
