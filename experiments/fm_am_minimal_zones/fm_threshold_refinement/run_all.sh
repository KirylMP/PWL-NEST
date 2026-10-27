#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
for variant in base lower middle all; do
  for stage in geometry poincare closed; do
    log="logs/${variant}_${stage}_$(date -u +%Y%m%dT%H%M%SZ).log"
    printf 'Starting %s %s; log=%s\n' "$variant" "$stage" "$log"
    python3 -u run.py "$variant" "$stage" 2>&1 | tee "$log"
  done
done
python3 compare.py
