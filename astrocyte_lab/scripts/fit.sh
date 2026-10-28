#!/usr/bin/env bash
# Fit either original or refined constructions, then catalog the models.
set -euo pipefail
lab="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$lab/.." && pwd)"
construction="${1:-base}"
if [[ "$construction" != base && "$construction" != refined && "$construction" != focus ]]; then
  printf 'Usage: bash astrocyte_lab/scripts/fit.sh [base|refined|focus]\n' >&2; exit 2
fi
mkdir -p "$lab/runs"
run="$(mktemp -d "$lab/runs/fit_${construction}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
printf 'RUN_DIR=%s\n' "$run"
export REST_RUN_DIR="$run" WORKERS="${WORKERS:-16}"
if [[ "$construction" == focus ]]; then
  export CONSTRUCTION=base FM_LOWER_TYPE=focus MODELS=fm
else
  export CONSTRUCTION="$construction" FM_LOWER_TYPE=node MODELS="${MODELS:-am fm}"
fi
status=0
bash "$repo/experiments/fm_am_minimal_zones/rest_refinement/run_server.sh" || status=$?
analysis_status=0
if [[ "$construction" == focus && -f "$run/fm/candidate.json" ]]; then
  printf 'Analyzing lower FM transition; finite scan can take time.\n'
  python3 -u "$lab/scripts/analyze_lower_fm.py" "$run/fm" 2>&1 | tee "$run/logs/fm_lower_analysis.log" || analysis_status=$?
fi
# Preserve any completed models even if a later fit or validation failed.
if [[ "$construction" == focus && "$status" != 0 ]]; then
  python3 "$lab/scripts/import_run.py" "$run" --allow-incomplete || true
else
  python3 "$lab/scripts/import_run.py" "$run" || true
fi
if (( status != 0 )); then
  printf 'Fit failed; intermediate results remain at %s\n' "$run" >&2
  exit "$status"
fi
if (( analysis_status != 0 )); then printf 'Lower analysis failed; see %s/logs/fm_lower_analysis.log\n' "$run" >&2; exit "$analysis_status"; fi
printf 'Fit complete: %s\n' "$run"
