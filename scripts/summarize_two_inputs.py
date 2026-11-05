"""Summarize raw fixed-input benchmark data; no extrapolation beyond measured cases."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('directories',nargs='+',type=Path);a=p.parse_args()
groups=defaultdict(lambda:defaultdict(list)); diagnostics={}
for path in a.directories:
    for line in (path/'runs.jsonl').read_text().splitlines():
        row=json.loads(line);key=(row['case'],row['I']);model=row['model'];res=row['result']
        if not row['warmup'] and 'run_s' in res:groups[key][model].append(res['run_s'])
        if row.get('diagnostics') is not None:diagnostics[key]=row['diagnostics']
        if 'error' in res:print('ERROR',key,model,res['error'])
for (case,I),models in sorted(groups.items()):
    print('\n',case,'I=',I)
    meds={k:statistics.median(v) for k,v in models.items() if v}
    for model in ('LR','PWA'):
        diag=diagnostics.get((case,I),{}).get(model,{})
        print(model,'seconds:',models.get(model,[]),'median:',meds.get(model),'dynamics:',diag.get('dynamics',diag.get('error')))
    if 'LR' in meds and 'PWA' in meds and meds['PWA']>0:print('speedup:',meds['LR']/meds['PWA'])
