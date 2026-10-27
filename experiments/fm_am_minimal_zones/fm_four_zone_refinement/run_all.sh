#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
if [[ ! -e results/base/candidate.json ]]; then
  python3 -u run.py base prepare
fi
log="logs/base_closed_$(date -u +%Y%m%dT%H%M%SZ).log"
python3 -u run.py base closed 2>&1 | tee "$log"
for variant in region1 region2 region3 all; do
  for stage in geometry poincare closed; do
    log="logs/${variant}_${stage}_$(date -u +%Y%m%dT%H%M%SZ).log"
    printf 'Starting %s %s; log=%s\n' "$variant" "$stage" "$log"
    python3 -u run.py "$variant" "$stage" 2>&1 | tee "$log"
  done
done
python3 compare.py
