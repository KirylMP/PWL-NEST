#!/usr/bin/env bash
# Run the same NEST workload without instrumentation and then with it.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
run_dir="$(mktemp -d "$PWD/results/profile_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
printf 'PROFILE_RESULTS=%s\n' "$run_dir"
trap 'printf "Profile stopped; outputs: %s\n" "$run_dir" >&2' ERR
export NEST_PROTOCOL="${NEST_PROTOCOL:-work-pulse}"
export NEST_DURATION_S="${NEST_DURATION_S:-30}" NEST_VALIDATION_S="${NEST_VALIDATION_S:-25}"
export NEST_RESOLUTION_MS="${NEST_RESOLUTION_MS:-0.1}" NEST_REPEATS="${NEST_REPEATS:-3}"
export NEST_SINGLE_NODES="${NEST_SINGLE_NODES:-64}" NEST_MULTI_NODES="${NEST_MULTI_NODES:-64}"
export NEST_THREADS="${NEST_THREADS:-16}" PWA_INIT_MODE="${PWA_INIT_MODE:-equilibrium}"
PWA_PROFILE=0 NEST_RESULTS_DIR="$run_dir/normal" bash run_benchmarks.sh
PWA_PROFILE=1 NEST_RESULTS_DIR="$run_dir/instrumented" bash run_benchmarks.sh
python3 profile_summary.py "$run_dir" | tee "$run_dir/profile_summary.log"
printf 'Completed: %s\n' "$run_dir"
