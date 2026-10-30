#!/usr/bin/env bash
# Run the paper's all-to-all and scaling benchmarks on the selected candidates.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PATH="$root/.venv/bin:$PATH" PYTHONPATH="$root/src" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PAPER_SNAPSHOT="$root/astrocyte_lab/paper_models"
export PAPER_PHYSICAL_RUN="$root/paper/local_results/physical_no_fold_multistart"
export PAPER_NETWORK_OUT="$root/paper/local_results/nest_network"
export ASTRO_RATIO_OUT="$root/paper/local_results/nest_scaling"
export PAPER_THREADS="${PAPER_THREADS:-2}" ASTRO_THREADS="${ASTRO_THREADS:-2}"
export PAPER_RECORD_INTERVALS_MS="${PAPER_RECORD_INTERVALS_MS:-10 1000}"
export PAPER_DURATION_S="${PAPER_DURATION_S:-60}" ASTRO_DURATION_S="${ASTRO_DURATION_S:-60}"
export PAPER_REPEATS="${PAPER_REPEATS:-3}" ASTRO_REPEATS="${ASTRO_REPEATS:-3}"
export BUILD_JOBS="${BUILD_JOBS:-2}"
[[ -f "$PAPER_PHYSICAL_RUN/fm/comparison/new_rank_01/parameters.json" ]] || { echo 'Complete physical_no_fold_multistart fitting and validation first.' >&2; exit 1; }
bash astrocyte_lab/experiments/paper_network/run.sh
bash astrocyte_lab/experiments/astro_ratio_network/run.sh
python3 scripts/paper_nest_speed_figure.py --local
python3 scripts/paper_local_lr_targets.py --workers 2
