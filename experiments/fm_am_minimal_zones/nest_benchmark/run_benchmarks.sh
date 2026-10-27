#!/usr/bin/env bash
# Build from the current AM/FM candidates and time PWA vs LR in NEST.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash build_module.sh
export LD_LIBRARY_PATH="$PWD/module/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
result_root="${NEST_RESULTS_DIR:-results}"
run_dir="$result_root/nest_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$run_dir"
duration="${NEST_DURATION_S:-20}"
validation="${NEST_VALIDATION_S:-20}"
resolution="${NEST_RESOLUTION_MS:-0.1}"
repeats="${NEST_REPEATS:-3}"
protocol="${NEST_PROTOCOL:-dynamic}"
python3 -u verify_nest_state.py 2>&1 | tee "$run_dir/state_checks.log"
python3 -u benchmark_nest.py --protocol "$protocol" --duration-s "$duration" --validation-s "$validation" --resolution-ms "$resolution" --nodes "${NEST_SINGLE_NODES:-16}" --threads 1 --repeats "$repeats" --output "$run_dir/one_thread.json" 2>&1 | tee "$run_dir/one_thread.log"
python3 -u benchmark_nest.py --protocol "$protocol" --duration-s "$duration" --validation-s "$validation" --resolution-ms "$resolution" --nodes "${NEST_MULTI_NODES:-64}" --threads "${NEST_THREADS:-16}" --repeats "$repeats" --output "$run_dir/multi_thread.json" 2>&1 | tee "$run_dir/multi_thread.log"
python3 plot_traces.py "$run_dir/one_thread_traces.npz"
printf 'Results: %s\n' "$(cd "$run_dir" && pwd)"
