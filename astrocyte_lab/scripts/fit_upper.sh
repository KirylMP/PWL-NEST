#!/usr/bin/env bash
set -euo pipefail
lab="$(cd "$(dirname "$0")/.." && pwd)"
parent="${1:-${FM_PARENT_CANDIDATE:-}}"
if [[ -z "$parent" ]]; then
  parent="$(python3 - "$lab/models" <<'PY'
import json,sys
from pathlib import Path
paths=[]
for p in Path(sys.argv[1]).glob('*/parameters.json'):
    try:
        d=json.loads(p.read_text())
        if d.get('case')=='fm' and d.get('construction')=='full_fixed' and d.get('stage')=='closed' and d.get('fold_constrained'):
            paths.append(p)
    except (OSError,ValueError):
        continue
if not paths:raise SystemExit('No completed full_fixed FM candidate in astrocyte_lab/models; import it or provide its parameters.json as an argument.')
print(max(paths,key=lambda p:p.stat().st_mtime_ns).resolve())
PY
)"
fi
if [[ ! -f "$parent" ]]; then printf 'Parent candidate not found: %s\n' "$parent" >&2; exit 1; fi
export FM_PARENT_CANDIDATE="$(cd "$(dirname "$parent")" && pwd)/$(basename "$parent")"
printf 'PARENT=%s\n' "$FM_PARENT_CANDIDATE"
status=0
for variant in ${UPPER_VARIANTS:-upper70 upper100 upper_both upper_three}; do
  printf '\n==== %s ====\n' "$variant"
  bash "$lab/scripts/fit.sh" "$variant" || status=1
done
python3 "$lab/scripts/list_models.py"
if (( status )); then printf 'One or more variants failed; intermediate models and logs were retained.\n' >&2; exit 1; fi
printf 'All upper-threshold fits completed.\n'
