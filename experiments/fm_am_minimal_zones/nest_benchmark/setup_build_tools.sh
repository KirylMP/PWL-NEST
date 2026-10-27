#!/usr/bin/env bash
# Install build-only tools and pinned Boost headers under this experiment.
set -euo pipefail
cd "$(dirname "$0")"
deps="$PWD/module/deps"
mkdir -p "$deps/bin"
if [[ ! -x "$deps/tools/bin/python" ]]; then python3 -m venv "$deps/tools"; fi
"$deps/tools/bin/python" -m pip install --disable-pip-version-check 'cmake==3.31.6' 'zstandard==0.23.0'
ln -sfn "$deps/tools/bin/cmake" "$deps/bin/cmake"
"$deps/tools/bin/python" - "$deps" <<'PYCODE'
import hashlib, pathlib, sys, tarfile, tempfile, urllib.request, zipfile
import zstandard
root=pathlib.Path(sys.argv[1])
packages=[
 ('Boost', 'https://conda.anaconda.org/conda-forge/noarch/libboost-headers-1.90.0-h707e725_3.conda',
  'ca00a55588bd34d7cb8b5ffa9ad535bf3a1e16a88fea6b5fb98a22e76e69b673',
  'include/boost/type_index.hpp', ('include/boost/',)),
 ('libltdl', 'https://conda.anaconda.org/conda-forge/linux-64/libltdl-2.4.3a-h5888daf_0.conda',
  '7620c6425d4491e17083106ca49624448fc16186c30a93cf2b58f862bba416d1',
  'include/ltdl.h', ('include/ltdl.h','include/libltdl/')),
]
for label,url,expected,sentinel,prefixes in packages:
 if (root/sentinel).is_file():continue
 with tempfile.TemporaryDirectory(prefix='header-download-',dir=root) as tmp:
    path=pathlib.Path(tmp)/'headers.conda'
    print(f'Downloading pinned {label} headers from conda-forge: {url}',flush=True)
    urllib.request.urlretrieve(url,path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=expected:raise ValueError(f'{label} package checksum mismatch: {digest}')
    with zipfile.ZipFile(path) as package:
        archives=[n for n in package.namelist() if n.startswith('pkg-') and n.endswith('.tar.zst')]
        if len(archives)!=1:raise ValueError(f'Unexpected {label} package structure')
        with package.open(archives[0]) as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as stream, tarfile.open(fileobj=stream,mode='r|') as tar:
            tar.extractall(root,members=(m for m in tar if any(m.name.startswith(prefix) for prefix in prefixes)),filter='data')
 if not (root/sentinel).is_file():raise RuntimeError(f'{label} headers are incomplete')
 print(f'{label} headers installed under',root/'include')
PYCODE
printf 'Build tools: %s\n' "$deps"
