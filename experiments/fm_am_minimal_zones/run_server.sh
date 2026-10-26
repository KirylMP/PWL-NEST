#!/usr/bin/env bash
# Full fixed-zone protocol for both endpoint regimes, followed by the LR K sweep.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export POINCARE_WEIGHT_SCALE="${POINCARE_WEIGHT_SCALE:-1}"
export WORKERS="${WORKERS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HERE/.cache}"
mkdir -p "$HERE/logs" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export VALIDATE_DENSE=1
echo "Started $(date -u +%FT%TZ), workers per audit=$WORKERS, Poincare scale=$POINCARE_WEIGHT_SCALE"
git rev-parse HEAD
python3 -m pip freeze > logs/server_environment.txt

run_am() {
  bash "$HERE/run_fit.sh" am geometry orbits poincare phase closed validate
  python3 -u "$HERE/am_two_thresholds/report_fixed.py" 2>&1 | tee "$HERE/logs/am_report.log"
}
run_fm() {
  bash "$HERE/run_fit.sh" fm geometry poincare orbit powell closed validate
  python3 -u "$HERE/fm_four_zone_fit/report.py" 2>&1 | tee "$HERE/logs/fm_report.log"
}
run_am & AM_PID=$!
run_fm & FM_PID=$!
STATUS=0
wait "$AM_PID" || STATUS=1
wait "$FM_PID" || STATUS=1
if ((STATUS)); then
  echo 'A model pipeline failed; inspect logs before resuming.' >&2
  exit "$STATUS"
fi
python3 -u "$HERE/sweep_K_zeros.py" 2>&1 | tee "$HERE/logs/K_sweep.log"
echo "Completed $(date -u +%FT%TZ)"
