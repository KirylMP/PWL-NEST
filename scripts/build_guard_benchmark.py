"""Compile a dedicated NEST 3.10 module for the paired guard benchmark."""
import argparse
from pathlib import Path
import subprocess
import nest

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--dev-include',type=Path);a=p.parse_args()
if not nest.__version__.startswith('3.10.'):raise RuntimeError('This adapter targets NEST 3.10')
include=Path(nest.__file__).parent/'include/nest';(ROOT/'.build').mkdir(exist_ok=True)
symbols=subprocess.check_output(['nm','-D',str(Path(nest.__file__).parent/'nestkernel_api.so')],text=True)
old_abi='_ZN4nest12ModelManager20register_node_model_' in symbols and 'ERKSs' in symbols
cmd=['g++','-std=c++20','-O3','-DNDEBUG','-fPIC','-shared','-fopenmp',
     '-D_GLIBCXX_USE_CXX11_ABI='+str(0 if old_abi else 1),'-I'+str(include)]
if a.dev_include:cmd+=['-I'+str(a.dev_include.resolve())]
output=ROOT/'.build/pwa_guard_bench_module.so'
subprocess.run(cmd+[str(ROOT/'native/nest_guard_bench.cpp'),'-o',str(output)],check=True)
print('Built',output)
