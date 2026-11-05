"""Check recorder output against independently recorded calcium crossings."""
import json
import sys
from pathlib import Path
import numpy as np

path=Path(sys.argv[1]);manifest=json.loads((path/'manifest.json').read_text())
threshold=float(manifest['settings']['threshold']);interval=float(manifest['settings']['interval_ms'])/1000
dt=float(manifest['settings']['dt_ms'])/1000
rows=[json.loads(x) for x in (path/'runs.jsonl').read_text().splitlines()]
for I in sorted({x['I'] for x in rows}):
    r=next(x for x in rows if x['I']==I and x.get('diagnostics'))
    for model in ('LR','PWA'):
        v=r['diagnostics'][model];tr=v['trace'];t=np.asarray(tr['time_s']);c=np.asarray(tr['C']);ev=np.asarray(v['event_times_s'])
        rising=np.where((c[:-1]<threshold)&(c[1:]>=threshold))[0]
        for k in rising:
            if not np.any((ev>=t[k]-2*dt)&(ev<=t[k+1]+2*dt)):
                raise ValueError(f'{model} I={I}: missing spike in sampled threshold interval')
        if I==0.5 and len(ev)<2:raise ValueError(f'{model}: expected oscillatory test spikes')
        if I==0.2 and len(ev)!=0:raise ValueError(f'{model}: unexpected low-input spikes')
        if v['ca_events_emitted']!=v['ca_events_recorded']:
            raise ValueError(f'{model}: emitted and recorded event counts differ')
        print(model,'I',I,'threshold',threshold,'recorded events',len(ev),'sampled rising crossings',len(rising))
for r in rows:
    if 'error' in r['result']:raise ValueError(str(r['result']['error']))
print('Smoke passed')
