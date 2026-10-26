#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (($# < 2)); then
  echo "Usage: $0 am|fm STAGE [STAGE ...]" >&2
  echo "AM: geometry orbits poincare phase closed validate" >&2
  echo "FM: geometry poincare orbit powell closed validate" >&2
  exit 2
fi

MODEL="$1"
shift
case "$MODEL" in
  am) MODEL_DIR="$HERE/am_two_thresholds"; SCRIPT=fit_am.py ;;
  fm) MODEL_DIR="$HERE/fm_four_zone_fit"; SCRIPT=fit_four.py ;;
  *) echo "Unknown model '$MODEL' (expected am or fm)" >&2; exit 2 ;;
esac

mkdir -p "$HERE/logs"
cd "$MODEL_DIR"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

for STAGE in "$@"; do
  case "$MODEL:$STAGE" in
    am:geometry|am:orbits|am:poincare|am:phase|am:closed|am:validate|fm:geometry|fm:poincare|fm:orbit|fm:powell|fm:closed|fm:validate) ;;
    *) echo "Unsupported stage '$STAGE' for '$MODEL'" >&2; exit 2 ;;
  esac
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  LOG="$HERE/logs/${MODEL}_${STAGE}_${STAMP}.log"
  echo "[$(date -u +%FT%TZ)] start model=$MODEL stage=$STAGE POINCARE_WEIGHT_SCALE=${POINCARE_WEIGHT_SCALE:-1} log=$LOG"
  if [[ "$MODEL:$STAGE" == "am:"* ]]; then
    MAP_KIND="${MAP_KIND:-mixed_power}" python3 -u "$SCRIPT" "$STAGE" 2>&1 | tee "$LOG"
  else
    python3 -u "$SCRIPT" "$STAGE" 2>&1 | tee "$LOG"
  fi
  echo "[$(date -u +%FT%TZ)] done model=$MODEL stage=$STAGE"
done
