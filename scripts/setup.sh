#!/usr/bin/env bash
set -euo pipefail
cd /workspace/PWL-NEST
python -m venv --system-site-packages .venv
export PIP_CACHE_DIR=/workspace/PWL-NEST/.cache/pip
.venv/bin/python -m pip install --no-build-isolation -e .
mkdir -p .build
for case in am fm; do
  g++ -O3 -shared -fPIC "native/${case}_engine.cpp" -o ".build/${case}_engine.so"
done
g++ -O3 -shared -fPIC native/strips_rk4.cpp -o .build/strips_rk4.so
g++ -std=c++17 -O3 -shared -fPIC native/pwa_event.cpp -o .build/pwa_event.so
g++ -O3 -std=c++20 -shared -fPIC native/fit_flow.cpp -o .build/fit_flow.so
