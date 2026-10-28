#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$here/results"
if [[ -n "${REST_RUN_DIR:-}" ]]; then
  run="$REST_RUN_DIR"
  if [[ -e "$run/am/candidate.json" || -e "$run/fm/candidate.json" ]]; then printf 'Run directory already contains a candidate: %s\n' "$run" >&2; exit 1; fi
  mkdir -p "$run"
else
  run="$(mktemp -d "$here/results/rest_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
fi
mkdir -p "$run/am" "$run/fm" "$run/logs"
printf 'RUN_DIR=%s\n' "$run"
trap 'printf "Stopped; saved results: %s\n" "$run" >&2' ERR
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export POINCARE_WEIGHT_SCALE="${POINCARE_WEIGHT_SCALE:-100}"
export FM_TARGET_GRID_SIZE="${FM_TARGET_GRID_SIZE:-10}"
export FM_CYCLE_COUNT_POINTS="${FM_CYCLE_COUNT_POINTS:-17}" FM_CYCLE_COUNT_SEEDS="${FM_CYCLE_COUNT_SEEDS:-32}"
export FM_CYCLE_COUNT_WEIGHT="${FM_CYCLE_COUNT_WEIGHT:-600}" WORKERS="${WORKERS:-16}"
construction="${CONSTRUCTION:-refined}"
lower_type="${FM_LOWER_TYPE:-node}"
if [[ "$construction" != refined && "$construction" != base && "$construction" != split && "$construction" != split_fixed && "$construction" != six_fixed && "$construction" != full_fixed && "$construction" != full_h_fixed && "$construction" != upper70 && "$construction" != upper100 && "$construction" != upper_both && "$construction" != upper_three ]]; then printf 'Unknown construction: %s\n' "$construction" >&2; exit 1; fi
for model in ${MODELS:-am fm}; do
  if [[ "$model" != am && "$model" != fm ]]; then printf 'Unknown model: %s\n' "$model" >&2; exit 1; fi
  parent_var="${model^^}_PARENT_CANDIDATE"
  parent_args=()
  if [[ -n "${!parent_var:-}" ]]; then parent_args=(--parent "${!parent_var}");fi
  for stage in prepare geometry poincare closed validate; do
    printf '[%s] %s %s\n' "$(date -u +%FT%TZ)" "$model" "$stage"
    type_args=()
    if [[ "$model" == fm ]]; then type_args=(--fm-lower-type "$lower_type"); fi
    if python3 -u "$here/run.py" "$model" "$stage" --coordinate mixed --construction "$construction" "${type_args[@]}" --out "$run/$model" "${parent_args[@]}" 2>&1 | tee "$run/logs/${model}_${stage}.log"; then
      if [[ "$stage" != validate ]]; then cp "$run/$model/candidate.json" "$run/$model/${stage}_candidate.json"; fi
    else
      status=$?
      if [[ "$stage" == closed && "${VALIDATE_ON_CLOSED_FAILURE:-1}" == 1 ]]; then
        printf 'Closed stage failed; scanning retained candidate before stopping.\n' >&2
        python3 -u "$here/run.py" "$model" validate --coordinate mixed --construction "$construction" "${type_args[@]}" --out "$run/$model" 2>&1 | tee "$run/logs/${model}_validate_after_failure.log" || true
      fi
      exit "$status"
    fi
  done
done
printf 'Completed: %s\n' "$run"
