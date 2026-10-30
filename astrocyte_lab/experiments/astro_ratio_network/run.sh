#!/usr/bin/env bash
set -euo pipefail
lab="$(cd "$(dirname "$0")/../.." && pwd)";repo="$(cd "$lab/.." && pwd)";bench="$repo/experiments/fm_am_minimal_zones/nest_benchmark"
physical_run="${PAPER_PHYSICAL_RUN:-$lab/runs/physical_no_fold_20261029T183743Z_V23NCi}";snapshot="${PAPER_SNAPSHOT:-$lab/paper_frozen_v1}"
mkdir -p "$lab/runs";run="${ASTRO_RATIO_OUT:-$(mktemp -d "$lab/runs/astro_ratio_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")}";mkdir -p "$run";run="$(cd "$run" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PWA_UNRESTRICTED_Q=1 PAPER_K_ER=.05 PAPER_BASELINE_IP3=.45
export MPLCONFIGDIR="$run/mpl" XDG_CACHE_HOME="$run/cache" LD_LIBRARY_PATH="$bench/module/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf 'RUN_DIR=%s\n' "$run"
for coord in physical mixed; do
 if [[ "$coord" == physical ]];then src="$physical_run/fm/comparison/new_rank_01/parameters.json";else src="$snapshot/fm_mixed/parameters.json";fi
 [[ -f "$src" ]] || { printf 'Missing model: %s\n' "$src" >&2;exit 1; }
 out="$run/$coord";mkdir -p "$out";cp "$src" "$out/parameters.json";export AM_CANDIDATE="$out/parameters.json" FM_CANDIDATE="$out/parameters.json"
 printf 'BUILD %s model=%s\n' "$coord" "$src"
 PWA_PROFILE=0 bash "$bench/build_module.sh" > "$out/build.log" 2>&1 || { cat "$out/build.log";exit 1; }
 python3 -u "$lab/experiments/paper_network/verify.py" > "$out/verify.log" 2>&1 || { cat "$out/verify.log";exit 1; }
 printf 'START %s\n' "$coord"
 reuse=(); if [[ "$coord" == mixed ]]; then reuse=(--reuse-lr "$run/physical"); fi
 python3 -u "$lab/experiments/astro_ratio_network/network.py" --out "$out" --ratios ${ASTRO_RATIOS:-1 2 4 8 16} \
  "${reuse[@]}" \
  --neurons "${ASTRO_NEURONS:-100}" --indegree "${ASTRO_INDEGREE:-10}" --duration "${ASTRO_DURATION_S:-60}" \
  --dt "${ASTRO_DT_MS:-0.1}" --threads "${ASTRO_THREADS:-16}" --repeats "${ASTRO_REPEATS:-3}" \
  --record-interval "${ASTRO_RECORD_MS:-10}" --ip3 "${ASTRO_IP3:-0.6}" --delta-ip3 "${ASTRO_DELTA_IP3:-0.0002}" \
  > "$out/network.log" 2>&1 || { cat "$out/network.log";exit 1; }
done
python3 "$lab/experiments/astro_ratio_network/summarize.py" "$run"
printf 'RESULTS=%s\n' "$run"
