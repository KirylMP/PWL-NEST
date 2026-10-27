#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pwa_nest_mpl}"
mkdir -p results
run_dir="$(mktemp -d "$PWD/results/brunel_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
printf 'NETWORK_RESULTS=%s\n' "$run_dir"
trap 'printf "Network run stopped; files: %s\n" "$run_dir" >&2' ERR
PWA_PROFILE=0 bash build_module.sh 2>&1 | tee "$run_dir/build.log"
export LD_LIBRARY_PATH="$PWD/module/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 -u network_brunel.py --output "$run_dir" --case "${NETWORK_CASE:-both}" \
  --duration "${NETWORK_DURATION_S:-10}" --dt "${NETWORK_DT_MS:-0.1}" --threads "${NETWORK_THREADS:-16}" \
  --ne "${NETWORK_NE:-800}" --ni "${NETWORK_NI:-200}" --astro "${NETWORK_ASTRO:-1000}" \
  --ce "${NETWORK_CE:-80}" --ci "${NETWORK_CI:-20}" --pool "${NETWORK_POOL:-10}" \
  --repeats "${NETWORK_REPEATS:-2}" --delta-ip3 "${NETWORK_DELTA_IP3:-0.0002}" \
  --seed "${NETWORK_SEED:-100}" 2>&1 | tee "$run_dir/network.log"
printf 'Completed: %s\n' "$run_dir"
