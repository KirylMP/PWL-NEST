#!/usr/bin/env bash
set -euo pipefail
lab="$(cd "$(dirname "$0")/../.." && pwd)"
repo="$(cd "$lab/.." && pwd)"
bench="$repo/experiments/fm_am_minimal_zones/nest_benchmark"
snapshot="${PAPER_SNAPSHOT:-$lab/paper_frozen_v1}"
physical_run="${PAPER_PHYSICAL_RUN:-$lab/runs/physical_no_fold_20261029T183743Z_V23NCi}"
mkdir -p "$lab/runs"
run="${PAPER_NETWORK_OUT:-$(mktemp -d "$lab/runs/paper_network_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")}"
mkdir -p "$run";run="$(cd "$run" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PWA_UNRESTRICTED_Q=1
export MPLCONFIGDIR="$run/mpl" XDG_CACHE_HOME="$run/cache"
export LD_LIBRARY_PATH="$bench/module/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf 'RUN_DIR=%s\n' "$run"
for mode in am afm fm; do
 case "$mode" in am) export PAPER_K_ER=0.1 PAPER_BASELINE_IP3="${PAPER_IP3_AM:-0.2}";; afm) export PAPER_K_ER=0.075 PAPER_BASELINE_IP3="${PAPER_IP3_AFM:-0.3}";; fm) export PAPER_K_ER=0.05 PAPER_BASELINE_IP3="${PAPER_IP3_FM:-0.45}";; esac
 for coord in physical mixed; do
  if [[ "$coord" == physical ]]; then src="$physical_run/$mode/comparison/new_rank_01/parameters.json"; else src="$snapshot/${mode}_mixed/parameters.json"; fi
  [[ -f "$src" ]] || { printf 'Missing model: %s\n' "$src" >&2;exit 1; }
  out="$run/$mode/$coord";mkdir -p "$out";cp "$src" "$out/parameters.json"
  export AM_CANDIDATE="$out/parameters.json" FM_CANDIDATE="$out/parameters.json"
  printf 'BUILD mode=%s coordinate=%s K=%s model=%s\n' "$mode" "$coord" "$PAPER_K_ER" "$src"
  PWA_PROFILE=0 bash "$bench/build_module.sh" > "$out/build.log" 2>&1 || { cat "$out/build.log";exit 1; }
  python3 -u "$lab/experiments/paper_network/verify.py" > "$out/verify.log" 2>&1 || { cat "$out/verify.log";exit 1; }
  for interval in ${PAPER_RECORD_INTERVALS_MS:-10 1000}; do
   label="${interval//./p}";dest="$out/record_${label}ms";mkdir -p "$dest"
   printf 'RUN mode=%s coordinate=%s record_interval=%s ms\n' "$mode" "$coord" "$interval"
   python3 -u "$lab/experiments/paper_network/network.py" --output "$dest" --case am \
    --neurons "${PAPER_NEURONS:-10}" --duration "${PAPER_DURATION_S:-60}" --dt "${PAPER_DT_MS:-0.1}" \
    --threads "${PAPER_THREADS:-16}" --repeats "${PAPER_REPEATS:-3}" --seed "${PAPER_SEED:-100}" \
    --record-interval "$interval" --delta-ip3 "${PAPER_DELTA_IP3:-0.008}" \
    --poisson-rate "${PAPER_POISSON_RATE:-2000}" --weight "${PAPER_EX_WEIGHT:-1}" \
    > "$dest/network.log" 2>&1 || { cat "$dest/network.log";exit 1; }
  done
 done
done
python3 "$lab/experiments/paper_network/summarize.py" "$run"
printf 'RESULTS=%s\n' "$run"
