"""Summarize a completed or interrupted constant-IP3 sweep."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
meta=json.loads((a.directory/'manifest.json').read_text());settings=meta['settings'];groups=defaultdict(list)
for line in (a.directory/'runs.jsonl').read_text().splitlines():
    row=json.loads(line)
    if not row['warmup']:groups[(row['case'],row['I'])].append(row)
summary=[]
for (case,I),rows in sorted(groups.items()):
    result={'case':case,'I':I}
    diagnostics=next((r['diagnostics'] for r in rows if r.get('diagnostics')), {})
    for model in ('LR','PWA'):
        selected=[r['result'] for r in rows if r['model']==model]
        times=[r['run_s'] for r in selected if 'run_s' in r]
        result[model+'_successful_repeats']=len(times)
        result[model+'_median_run_s']=statistics.median(times) if times else None
        result[model+'_errors']=' | '.join(r.get('error','') for r in selected if r.get('error'))
        diag=diagnostics.get(model,{})
        for key,value in diag.get('dynamics',{}).items():result[model+'_'+key]=value
        result[model+'_diagnostic_error']=diag.get('error','')
        result[model+'_max_ip3_drift']=diag.get('max_ip3_drift')
        if model=='PWA':
            counts=[r['counters']['region_crossings']/(r['node_count']*settings['duration_s']) for r in selected if 'counters' in r]
            result['PWA_crossings_per_cell_s']=statistics.median(counts) if counts else None
    lr,pwa=result['LR_median_run_s'],result['PWA_median_run_s']
    result['speedup']=lr/pwa if lr is not None and pwa else None
    summary.append(result)
fields=sorted({key for r in summary for key in r},key=lambda x: (x not in ('case','I'),x))
with (a.directory/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(summary)
print('Saved',a.directory/'summary.csv','points:',len(summary),'incomplete or failed points:',sum(r['LR_successful_repeats']!=settings['repeats'] or r['PWA_successful_repeats']!=settings['repeats'] for r in summary))
