#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
cd "$root"
mkdir -p astrocyte_lab/runs
run="${POLY_OUT:-$(mktemp -d "$root/astrocyte_lab/runs/polynomial_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")}"
mkdir -p "$run";run="$(cd "$run" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
extra=()
if [[ -n "${FM_MIXED_MODEL:-}" ]]; then
  [[ -f "$FM_MIXED_MODEL" ]] || { echo "FM_MIXED_MODEL not found: $FM_MIXED_MODEL" >&2; exit 1; }
  extra+=(--fm-mixed "$FM_MIXED_MODEL")
else
  echo 'FM source: earlier archived candidate in paper/local_results/models/fm_mixed/parameters.json'
fi
printf 'RUN_DIR=%s\n' "$run"
printf 'Degrees=%s; modes=%s; workers=%s; starts per seed=%s\n' "${POLY_DEGREES:-2 3 4 5 6}" "${POLY_MODES:-am afm fm}" "${WORKERS:-16}" "${POLY_STARTS:-2}"
python3 -u "$here/run_server.py" --out "$run" --workers "${WORKERS:-16}" --degrees ${POLY_DEGREES:-2 3 4 5 6} --modes ${POLY_MODES:-am afm fm} --starts "${POLY_STARTS:-2}" --powell-maxfev "${POLY_POWELL_MAXFEV:-6000}" --ls-max-nfev "${POLY_LS_MAX_NFEV:-400}" --duration "${POLY_DURATION:-350}" --validation-duration "${POLY_VALIDATION_DURATION:-450}" --dt "${POLY_DT:-0.01}" "${extra[@]}" 2>&1 | tee "$run/run.log"
printf '\nRESULTS=%s\nCSV=%s\n' "$run" "$run/results.csv"
