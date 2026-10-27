#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pwa_nest_mpl}"
mkdir -p results
run_dir="$(mktemp -d "$PWD/results/all_to_all_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
printf 'NETWORK_RESULTS=%s\n' "$run_dir"
trap 'printf "Network stopped; files: %s\n" "$run_dir" >&2' ERR
PWA_PROFILE=0 bash build_module.sh 2>&1 | tee "$run_dir/build.log"
export LD_LIBRARY_PATH="$PWD/module/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 -u network_all_to_all.py --output "$run_dir" --case "${NETWORK_CASE:-both}" \
  --neurons "${NETWORK_NEURONS:-10}" --duration "${NETWORK_DURATION_S:-60}" --dt "${NETWORK_DT_MS:-0.1}" \
  --threads "${NETWORK_THREADS:-16}" --repeats "${NETWORK_REPEATS:-2}" --seed "${NETWORK_SEED:-100}" \
  --delta-ip3 "${NETWORK_DELTA_IP3:-0.008}" --poisson-rate "${NETWORK_POISSON_RATE:-2000}" \
  --weight "${NETWORK_EX_WEIGHT:-1}" 2>&1 | tee "$run_dir/network.log"
printf 'Completed: %s\n' "$run_dir"
