#!/usr/bin/env bash
# A separate experiment with a finite-scan penalty for early large cycles.
set -euo pipefail
lab="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# != 1 ]]; then printf 'Usage: bash astrocyte_lab/scripts/fit_lower_onset.sh PATH/parameters.json\n' >&2; exit 2; fi
export FM_EXPERIMENT_LABEL=lower_onset FM_LOCAL_FOCUS_TARGETS=5 FM_LOCAL_FOCUS_WEIGHT=0
export FM_LOWER_LARGE_WEIGHT="${FM_LOWER_LARGE_WEIGHT:-600}"
export FM_LOWER_LARGE_I="${FM_LOWER_LARGE_I:-0.524,0.526,0.5275,0.528}"
export LOWER_I_START="${LOWER_I_START:-0.518}" LOWER_I_STOP="${LOWER_I_STOP:-0.534}"
bash "$lab/scripts/fit_local_focus.sh" "$1"
