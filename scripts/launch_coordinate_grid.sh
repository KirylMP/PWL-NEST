#!/usr/bin/env bash
set -euo pipefail
grid_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
grid_start_dir="$PWD"
cd "$grid_project_dir"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$grid_project_dir/.cache/matplotlib"
export XDG_CACHE_HOME="$grid_project_dir/.cache"
export PYTHONPATH="$grid_project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p .build .cache
grid_python="${GRID_PYTHON:-}"
if [[ -z "$grid_python" ]]; then
  for grid_candidate in "$grid_project_dir/.venv/bin/python" "$grid_start_dir/.venv/bin/python" "$(dirname "$grid_project_dir")/.venv/bin/python"; do
    if [[ -x "$grid_candidate" ]]; then grid_python="$grid_candidate"; break; fi
  done
fi
grid_python="${grid_python:-python3}"
if ! "$grid_python" -c 'import numpy,scipy,matplotlib' >/dev/null 2>&1; then
  "$grid_python" -m venv "$grid_project_dir/.venv-grid"
  grid_python="$grid_project_dir/.venv-grid/bin/python"
  "$grid_python" -m pip install -e "$grid_project_dir"
fi
command -v g++ >/dev/null || { echo 'g++ is required to build the native fitting engines.' >&2; exit 1; }
for grid_engine in am_engine fm_engine strips_rk4; do
  if [[ ! -f ".build/$grid_engine.so" || "native/$grid_engine.cpp" -nt ".build/$grid_engine.so" ]]; then
    g++ -O3 -shared -fPIC "native/$grid_engine.cpp" -o ".build/$grid_engine.so"
  fi
done
"$grid_python" "$grid_project_dir/scripts/grid_preflight.py"
grid_output="${GRID_OUTPUT:-$grid_project_dir/runs/coordinate-grid-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$grid_output"
if [[ -f "$grid_output/master.pid" ]] && kill -0 "$(cat "$grid_output/master.pid")" 2>/dev/null; then
  echo "A process with the recorded master PID is already running: $grid_output" >&2; exit 1
fi
grid_worker_count="${GRID_WORKERS:-$($grid_python -c 'import os; print(min(4,max(1,(os.cpu_count() or 2)//2)))')}"
grid_command=("$grid_python" "$grid_project_dir/scripts/overnight_coordinate_grid.py" --output "$grid_output" --hours "${GRID_HOURS:-8}" --workers "$grid_worker_count" --budget "${GRID_BUDGET:-1500}" --trial-timeout "${GRID_TRIAL_TIMEOUT:-900}")
if [[ "${GRID_FOREGROUND:-0}" == 1 ]]; then
  exec "${grid_command[@]}" "$@"
fi
nohup "${grid_command[@]}" "$@" >"$grid_output/master.log" 2>&1 < /dev/null &
grid_pid=$!
echo "$grid_pid" >"$grid_output/launch.pid"
echo "Started master PID=$grid_pid"
echo "Results: $grid_output"
echo "Log: $grid_output/master.log"
echo "Stop gracefully: kill -TERM $grid_pid"
