"""Show timings and event counts from calcium-event benchmark runs."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('directories',nargs='+',type=Path);a=p.parse_args()
groups=defaultdict(lambda:defaultdict(list));diagnostics={}
for directory in a.directories:
    for line in (directory/'runs.jsonl').read_text().splitlines():
        r=json.loads(line);key=(r['case'],r['I']);model=r['model']
        if r.get('diagnostics'):diagnostics[key]=r['diagnostics']
        if not r['warmup']:
            if 'error' in r['result']:print('ERROR',key,model,r['result']['error'])
            else:groups[key][model].append(r['result'])
for key,models in sorted(groups.items()):
    print('\n',key)
    med={}
    for model in ('LR','PWA'):
        runs=models.get(model,[])
        if not runs:print(model,'NO SUCCESSFUL RUNS');continue
        t=[v['run_s'] for v in runs];med[model]=statistics.median(t)
        d=diagnostics.get(key,{}).get(model,{})
        print(model,'run_s',t,'median',med[model],
              'recorded_events',[v['ca_events_recorded'] for v in runs],
              'emitted_events',[v['ca_events_emitted'] for v in runs],
              'diagnostic',d.get('dynamics',d.get('error')))
    if len(med)==2 and med['PWA']>0:print('speedup',med['LR']/med['PWA'])
