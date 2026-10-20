#!/usr/bin/env bash
set -euo pipefail
# Run from the user's existing lab directory. Preserve its files and virtualenv.
grid_lab_dir="$PWD"
grid_checkout="$grid_lab_dir/pwl-nest-nightly-$(date -u +%Y%m%dT%H%M%SZ)"
export GIT_TERMINAL_PROMPT=0
git clone --depth 1 --filter=blob:none --sparse --branch work https://github.com/KirylMP/PWL-NEST.git "$grid_checkout"
git -C "$grid_checkout" sparse-checkout set src scripts native models docs
if [[ -x "$grid_lab_dir/.venv/bin/python" ]]; then export GRID_PYTHON="$grid_lab_dir/.venv/bin/python"; fi
exec bash "$grid_checkout/scripts/launch_coordinate_grid.sh" "$@"
