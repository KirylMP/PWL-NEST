"""Summarize diagnostic errors and paired run times of benchmark_accuracy_speed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args()
rows=[json.loads(line) for line in (a.directory/'runs.jsonl').read_text().splitlines() if line.strip()]
groups={}
for row in rows:
    key=(row['case'],row['I'],row['dt_ms'])
    g=groups.setdefault(key,{'diagnostic':{},'timing':{}})
    if 'repeat' in row:
        if not row['warmup']:g['timing'][(row['model'],row['repeat'])]=row
    else:g['diagnostic'][row['model']]=row
out=[]
for (case,I,dt),g in sorted(groups.items()):
    row={'case':case,'I':I,'dt_ms':dt}
    repeats=sorted({rep for model,rep in g['timing']})
    ratios=[]
    for rep in repeats:
        lr=g['timing'].get(('LR',rep),{}).get('timing',{}).get('run_s')
        pw=g['timing'].get(('PWA',rep),{}).get('timing',{}).get('run_s')
        if lr is not None and pw is not None and pw>0:ratios.append(lr/pw)
    row['pairs']=len(ratios)
    for label,func in [('speedup_median',np.median),('speedup_min',np.min),('speedup_max',np.max)]:
        row[label]=float(func(ratios)) if ratios else None
    for model in ('LR','PWA'):
        d=g['diagnostic'].get(model,{})
        for label,source in [('C_max_abs','error'),('C_rmse','error'),('ref_C_diff','reference_disagreement')]:
            field='C_max_abs' if label=='ref_C_diff' else label
            row[model+'_'+label]=d.get(source,{}).get(field)
        row[model+'_diagnostic_error']=d.get('error_diagnostic')
        row[model+'_reference_boundary_free']=d.get('reference_boundary_free')
    out.append(row)
if not out: raise SystemExit('No results')
with (a.directory/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print('case I dt_ms pairs speedup LR_max PWA_max LR_ref PWA_ref')
for x in out:
    print(x['case'],x['I'],x['dt_ms'],x['pairs'],x['speedup_median'],
          x['LR_C_max_abs'],x['PWA_C_max_abs'],x['LR_ref_C_diff'],x['PWA_ref_C_diff'])
print('Saved',a.directory/'summary.csv')
