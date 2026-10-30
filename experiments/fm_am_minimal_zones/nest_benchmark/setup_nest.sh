#!/usr/bin/env bash
# Separate, non-system NEST environment. Does not install a PWA NEST module.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
runtime="${FM_NEST_RUNTIME:-$here/.nest_runtime}"
mamba_root="$runtime/mamba"
env_dir="$runtime/nest"
mkdir -p "$runtime"
if [[ ! -x "$runtime/bin/micromamba" ]]; then
  command -v curl >/dev/null || { echo 'curl is required' >&2; exit 1; }
  archive="$runtime/micromamba.tar.bz2"
  curl --fail --location --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest --output "$archive"
  tar -xjf "$archive" -C "$runtime" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$mamba_root"
if [[ ! -x "$env_dir/bin/python" ]]; then
  "$runtime/bin/micromamba" create --yes --prefix "$env_dir" --channel conda-forge --strict-channel-priority \
    'python=3.12' 'nest-simulator=3.10' cmake ninja gsl gxx_linux-64
fi
"$runtime/bin/micromamba" run --prefix "$env_dir" python - <<'PY'
import nest
print('NEST version:',getattr(nest,'__version__','unknown'))
found='astrocyte_lr_1994' in nest.Models()
print('astrocyte_lr_1994 available:',found)
if not found:raise SystemExit('NEST environment lacks astrocyte_lr_1994; benchmark cannot proceed')
PY
if [[ -x "$env_dir/bin/nest-config" ]]; then
  "$runtime/bin/micromamba" run --prefix "$env_dir" nest-config --version
else
  echo 'nest-config is missing; native extension cannot be compiled in this environment' >&2
  exit 1
fi
printf 'NEST environment: %s\n' "$env_dir"
printf 'Run Python using: %s/bin/micromamba run -p %s python\n' "$runtime" "$env_dir"
