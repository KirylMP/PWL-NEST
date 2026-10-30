#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nest_config="${NEST_CONFIG:-$(command -v nest-config || true)}"
if [[ -z "$nest_config" || ! -x "$nest_config" ]]; then echo 'nest-config not found; activate the NEST environment or set NEST_CONFIG' >&2; exit 1; fi
if [[ "$($nest_config --version)" != 3.10* ]]; then echo 'This extension requires NEST 3.10.x' >&2; exit 1; fi
if [[ -x module/deps/bin/cmake ]]; then export PATH="$PWD/module/deps/bin:$PATH"; fi
if [[ -z "${BOOST_INCLUDE_DIR:-}" ]]; then
  python_prefix="$(python3 -c 'import sys; print(sys.prefix)')"
  if [[ -f "$python_prefix/include/boost/type_index.hpp" ]]; then export BOOST_INCLUDE_DIR="$python_prefix/include"; fi
fi
if ! command -v cmake >/dev/null || { [[ -z "${BOOST_INCLUDE_DIR:-}" ]] && [[ ! -f /usr/include/boost/type_index.hpp ]] && [[ ! -f module/deps/include/boost/type_index.hpp ]]; } || { [[ ! -f /usr/include/ltdl.h ]] && [[ ! -f module/deps/include/ltdl.h ]] && [[ ! -f "${BOOST_INCLUDE_DIR:-/nonexistent}/ltdl.h" ]]; }; then
  bash setup_build_tools.sh
fi
if [[ -x module/deps/bin/cmake ]]; then export PATH="$PWD/module/deps/bin:$PATH"; fi
command -v cmake >/dev/null || { echo 'cmake is required for NEST module build' >&2; exit 1; }
parent="$(cd ../.. && pwd)"
am="${AM_CANDIDATE:-$parent/fm_am_minimal_zones/am_two_thresholds/fits/mixed_power_fixed/candidate.json}"
fm="${FM_CANDIDATE:-$parent/fm_am_minimal_zones/fm_four_zone_fit/candidate.json}"
python3 module/generate.py --am "$am" --fm "$fm"
nest_include="${NEST_INCLUDE_DIR:-$(python3 -c 'import importlib.util,pathlib; p=importlib.util.find_spec("nest"); print(pathlib.Path(p.origin).parent / "include" / "nest" if p and p.origin else "")')}"
boost_include="${BOOST_INCLUDE_DIR:-}"
if [[ -n "$nest_include" && ! -f "$nest_include/nest_extension_interface.h" ]]; then
  nest_include=""
fi
if [[ -z "$boost_include" && -f "$PWD/module/deps/include/boost/type_index.hpp" ]]; then
  boost_include="$PWD/module/deps/include"
fi
abi="${NEST_GLIBCXX_ABI:-}"
if [[ -z "$abi" ]]; then
  command -v nm >/dev/null || { echo 'nm is required to detect NEST C++ ABI; alternatively set NEST_GLIBCXX_ABI=0 or 1' >&2; exit 1; }
  nest_kernel="$(python3 -c 'import importlib.util,pathlib; p=importlib.util.find_spec("nest"); print(pathlib.Path(p.origin).parent / "nestkernel_api.so")')"
  symbols="$(nm -D "$nest_kernel" | grep 'StructuralPlasticityNode21get_synaptic_elements' || true)"
  if [[ "$symbols" == *'get_synaptic_elementsESs'* ]]; then abi=0
  elif [[ "$symbols" == *'get_synaptic_elementsB5cxx11'* || "$symbols" == *'get_synaptic_elementsENSt7__cxx11'* ]]; then abi=1
  else echo "Could not detect C++ ABI of $nest_kernel; set NEST_GLIBCXX_ABI=0 or 1" >&2; exit 1; fi
fi
if [[ "$abi" != 0 && "$abi" != 1 ]]; then echo 'NEST_GLIBCXX_ABI must be 0 or 1' >&2; exit 1; fi
printf 'NEST include=%s; Boost include=%s; C++ ABI=%s\n' "$nest_include" "${boost_include:-system}" "$abi"
cmake -S module -B module/build -Dwith-nest="$nest_config" -DNEST_INCLUDE_DIR="$nest_include" -DBOOST_INCLUDE_DIR="$boost_include" -DEXTRA_INCLUDE_DIR="$PWD/module/deps/include" -DNEST_GLIBCXX_ABI="$abi" -DPWA_PROFILE_ENABLED="${PWA_PROFILE:-0}"
cmake --build module/build -j "${BUILD_JOBS:-4}"
printf '\nBuilt %s/module/build/pwa_astrocyte_module.so\n' "$(pwd)"
