#!/usr/bin/env bash
# Fit either original or refined constructions, then catalog the models.
set -euo pipefail
lab="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$lab/.." && pwd)"
construction="${1:-base}"
if [[ "$construction" != base && "$construction" != refined ]]; then
  printf 'Usage: bash astrocyte_lab/scripts/fit.sh [base|refined]\n' >&2; exit 2
fi
mkdir -p "$lab/runs"
run="$(mktemp -d "$lab/runs/fit_${construction}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
printf 'RUN_DIR=%s\n' "$run"
export REST_RUN_DIR="$run" CONSTRUCTION="$construction" MODELS="${MODELS:-am fm}" WORKERS="${WORKERS:-16}"
status=0
bash "$repo/experiments/fm_am_minimal_zones/rest_refinement/run_server.sh" || status=$?
# Preserve any completed models even if a later fit or validation failed.
python3 "$lab/scripts/import_run.py" "$run" || true
if (( status != 0 )); then
  printf 'Fit failed; intermediate results remain at %s\n' "$run" >&2
  exit "$status"
fi
printf 'Fit complete: %s\n' "$run"
