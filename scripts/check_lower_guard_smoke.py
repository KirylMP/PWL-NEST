"""Validate inactive guard equivalence and active guard on the AM reset."""
import json
from pathlib import Path
import sys
import numpy as np

root=Path(sys.argv[1]);rows=[json.loads(x) for x in (root/'runs.jsonl').read_text().splitlines()]
def diagnostic(kind,model):
    return next(x['diagnostic'] for x in rows if x['kind']==kind and x['model']==model and x['diagnostic'] is not None)
a=diagnostic('constant','three');b=diagnostic('constant','four')
if b['guard_crossings']!=0:raise ValueError('guard unexpectedly active in constant AM test')
if np.max(np.abs(np.asarray(a['trace']['C'])-np.asarray(b['trace']['C'])))>1e-7:raise ValueError('inactive guard changed sampled trajectory')
a=diagnostic('reset_from_1.2','three');b=diagnostic('reset_from_1.2','four')
if a['min_C']>=0:raise ValueError('stress did not reproduce unguarded negative excursion')
if b['min_C']<=0 or b['guard_crossings']<1:raise ValueError('guarded reset failed or did not enter added zone')
for row in rows:
    if row['result']['events_emitted']!=row['result']['events_recorded']:raise ValueError('spike recorder count mismatch')
print('Smoke passed: unchanged trajectory above cut; guard entered and prevented negative sampled calcium in this reset.')
