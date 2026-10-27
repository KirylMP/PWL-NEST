#!/usr/bin/env bash
# Fit the original AM/FM constructions with the rest constraint, then benchmark.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
bench="$here/../nest_benchmark"
mkdir -p "$here/results"
run="$(mktemp -d "$here/results/base_nest_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
printf 'RUN_DIR=%s\n' "$run"
trap 'printf "Stopped; retained files: %s\n" "$run" >&2' ERR
export REST_RUN_DIR="$run/fit" CONSTRUCTION=base MODELS='am fm'
bash "$here/run_server.sh"
export AM_CANDIDATE="$run/fit/am/candidate.json" FM_CANDIDATE="$run/fit/fm/candidate.json"
export NEST_RESULTS_DIR="$run/nest" NETWORK_RESULTS_DIR="$run/network"
export NEST_DURATION_S="${NEST_DURATION_S:-60}" NEST_VALIDATION_S="${NEST_VALIDATION_S:-60}"
export NEST_PROTOCOL="${NEST_PROTOCOL:-work-pulse}"
export NEST_THREADS="${NEST_THREADS:-16}" NETWORK_THREADS="${NETWORK_THREADS:-16}"
printf 'Starting NEST benchmark with AM=%s FM=%s\n' "$AM_CANDIDATE" "$FM_CANDIDATE"
bash "$bench/run_benchmarks.sh"
printf 'Starting all-to-all network comparison\n'
bash "$bench/run_all_to_all.sh"
printf 'Completed: %s\n' "$run"
