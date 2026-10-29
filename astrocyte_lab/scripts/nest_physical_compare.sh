#!/usr/bin/env bash
# Compare two physical-calcium coordinate choices with LR in NEST 3.10.
set -euo pipefail
lab="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$lab/.." && pwd)"
bench="$repo/experiments/fm_am_minimal_zones/nest_benchmark"
mkdir -p "$lab/runs"
run="${NEST_PHYSICAL_RESULTS_DIR:-}"
if [[ -z "$run" ]]; then run="$(mktemp -d "$lab/runs/nest_physical_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"; else mkdir -p "$run"; run="$(cd "$run" && pwd)"; fi
printf 'RUN_DIR=%s\n' "$run"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NEST_DURATION_S="${NEST_DURATION_S:-60}" NEST_VALIDATION_S="${NEST_VALIDATION_S:-60}"
export NEST_SINGLE_NODES="${NEST_SINGLE_NODES:-64}" NEST_MULTI_NODES="${NEST_MULTI_NODES:-64}"
export NEST_THREADS="${NEST_THREADS:-16}" NEST_REPEATS="${NEST_REPEATS:-3}" NEST_PROTOCOL="${NEST_PROTOCOL:-work-pulse}"
export NETWORK_NEURONS="${NETWORK_NEURONS:-10}" NETWORK_DURATION_S="${NETWORK_DURATION_S:-60}" NETWORK_THREADS="${NETWORK_THREADS:-16}" NETWORK_REPEATS="${NETWORK_REPEATS:-2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$run/matplotlib_cache}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$run/cache}"
failures=0
for mode in identity power; do
  if [[ "$mode" == identity ]]; then
    am="$lab/models/am_2thr_traj_identity_20261029/parameters.json"
    fm="$lab/models/fm_2thr_traj_identity_20261029/parameters.json"
  else
    am="$lab/models/am_2thr_traj_power_20261029/parameters.json"
    fm="$lab/models/fm_2thr_traj_power_20261029/parameters.json"
  fi
  export AM_CANDIDATE="$am" FM_CANDIDATE="$fm"
  for dt in ${NEST_RESOLUTIONS_MS:-0.1 1.0}; do
    label="${dt//./p}";out="$run/$mode/dt_${label}ms";mkdir -p "$out"
    export NEST_RESOLUTION_MS="$dt" NETWORK_DT_MS="$dt"
    export NEST_RESULTS_DIR="$out/benchmark" NETWORK_RESULTS_DIR="$out/network"
    printf '[%s] mode=%s dt=%s AM=%s FM=%s\n' "$(date -u +%FT%TZ)" "$mode" "$dt" "$AM_CANDIDATE" "$FM_CANDIDATE"
    if bash "$bench/run_benchmarks.sh" 2>&1 | tee "$out/benchmark_driver.log"; then
      printf 'BENCHMARK_OK mode=%s dt=%s\n' "$mode" "$dt"
      if [[ "${RUN_NETWORK:-1}" == 1 ]]; then
        if bash "$bench/run_all_to_all.sh" 2>&1 | tee "$out/network_driver.log"; then
          printf 'NETWORK_OK mode=%s dt=%s\n' "$mode" "$dt"
        else
          failures=$((failures+1));printf 'NETWORK_FAILED mode=%s dt=%s\n' "$mode" "$dt" >&2
        fi
      fi
    else
      failures=$((failures+1));printf 'BENCHMARK_FAILED mode=%s dt=%s; validation or build may have failed\n' "$mode" "$dt" >&2
    fi
  done
done
python3 "$lab/scripts/summarize_nest_physical.py" "$run" || true
printf 'RESULTS=%s failures=%s\n' "$run" "$failures"
if ((failures)); then exit 1; fi
