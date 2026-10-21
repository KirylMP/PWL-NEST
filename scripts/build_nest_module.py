"""Build the NEST 3.10 adapter using the active Python's NEST wheel headers.

Needs g++, Boost, libltdl and GSL headers. --dev-include permits local headers
when system dev packages cannot be installed. No root or CMake is required.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import urllib.request

import nest

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dev-include', type=Path)
p.add_argument('--fetch-dev-headers', action='store_true',
               help='Download checksum-pinned Debian dev packages and extract under .cache (no root)')
a=p.parse_args()
if a.fetch_dev_headers:
    cache=ROOT/'.cache/nest-source';cache.mkdir(parents=True,exist_ok=True)
    packages=[
        ('libt/libtool/libltdl-dev_2.6.2-3_amd64.deb','f8a2323a189076b5948c8147df6ba417582bfb043b25fb621ba8e31a25513ad3'),
        ('g/gsl/libgsl-dev_2.8+dfsg-6+b1_amd64.deb','a6b903d1bf40d67b6c126c404adbe99cb3be8774c0f5d0ec797eaa999c8460b9'),
        ('b/boost1.83/libboost1.83-dev_1.83.0-4.2~bpo12+1_amd64.deb','83ac68a24f5a9959b5bfcabd558b08e699b56dd17e41e12c04237c6bef2eb399'),
    ]
    for name,digest in packages:
        target=cache/Path(name).name
        if not target.exists():
            urllib.request.urlretrieve('https://deb.debian.org/debian/pool/main/'+name,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Checksum mismatch for {target}')
        subprocess.run(['dpkg-deb','-x',str(target),str(cache/'dev')],check=True)
    a.dev_include=cache/'dev/usr/include'
if not nest.__version__.startswith('3.10.'):
    raise RuntimeError('This adapter targets NEST 3.10; verify API compatibility before using another version')
include=Path(nest.__file__).parent/'include/nest'
(ROOT/'.build').mkdir(exist_ok=True)
# PyPI manylinux NEST 3.10 is built with libstdc++'s old string ABI.
symbols=subprocess.check_output(['nm', '-D', str(Path(nest.__file__).parent/'nestkernel_api.so')], text=True)
old_abi='_ZN4nest12ModelManager20register_node_model_' in symbols and 'ERKSs' in symbols
command=['g++', '-std=c++20', '-O3', '-DNDEBUG', '-fPIC', '-shared', '-fopenmp',
         '-D_GLIBCXX_USE_CXX11_ABI='+str(0 if old_abi else 1), '-I'+str(include)]
if a.dev_include:
    command+=['-I'+str(a.dev_include.resolve())]
subprocess.run(command+[str(ROOT/'native/nest_pwa_module.cpp'), '-o', str(ROOT/'.build/pwl_astro_module.so')], check=True)
print(ROOT/'.build/pwl_astro_module.so')
wheel_libs=Path(nest.__file__).parent.parent/'nest_simulator.libs'
gsl=list(wheel_libs.glob('libgsl-*.so*'))
if gsl:
    link=[str(gsl[0]), '-Wl,-rpath,'+str(wheel_libs)]
else:
    link=['-lgsl', '-lgslcblas']
subprocess.run(command+[str(ROOT/'native/solver_timing.cpp'), *link,
                       '-o', str(ROOT/'.build/solver_timing.so')], check=True)
print(ROOT/'.build/solver_timing.so')
