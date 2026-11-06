"""Paired 3-/4-zone runtime curves and reset transients from raw NEST data."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'results/guard_main';raw=(folder/'runs.jsonl').read_bytes()
rows=[json.loads(line) for line in raw.splitlines()];settings=json.loads((folder/'manifest.json').read_text())
manifest=json.loads((ROOT/'paper/data/random_selected/manifest.json').read_text())['models'];groups=defaultdict(lambda:defaultdict(list));diag=defaultdict(dict)
for r in rows:
    if r['source_sha256']!=manifest[r['case']]['candidate_sha256']:raise ValueError('candidate hash mismatch')
    key=(r['case'],r['kind'],r['I']);groups[key][r['model']].append(r)
    if r['diagnostic'] is not None:diag[key][r['model']]=r['diagnostic']
records=[]
for key,models in sorted(groups.items()):
    if set(models)!=set(('three','four')):raise ValueError('missing model')
    if len({tuple(r['initial']) for xs in models.values() for r in xs})!=1:raise ValueError('different initial conditions')
    record={'case':key[0],'kind':key[1],'I':key[2]}
    for model in ('three','four'):
        r=[x['result'] for x in models[model] if not x['warmup']]
        if len(r)!=3 or any(x['events_recorded']!=x['events_emitted'] for x in r):raise ValueError('incomplete or invalid run')
        record[model+'_times']=[x['run_s'] for x in r];record[model+'_median_s']=float(np.median(record[model+'_times']))
        record[model+'_guard_crossings']=[x['guard_crossings'] for x in r]
        record[model+'_diagnostic_min_C']=diag[key][model]['min_C']
    record['overhead_pct']=100*(record['four_median_s']/record['three_median_s']-1)
    d=diag[key];record['diagnostic_trace_max_diff']=float(np.max(np.abs(np.asarray(d['three']['trace']['C'])-np.asarray(d['four']['trace']['C']))))
    records.append(record)

fig,axes=plt.subplots(1,3,figsize=(11,3.3),constrained_layout=True)
for case,color in [('AM','#1a6099'),('AFM','#bd4f28'),('FM','#775094')]:
    data=[x for x in records if x['case']==case and x['kind']=='constant']
    axes[0].plot([x['I'] for x in data],[x['overhead_pct'] for x in data],marker='o',lw=1.2,label=case,c=color)
axes[0].axhline(0,color='0.5',lw=.7);axes[0].set_xlabel('IP₃ I (µM)');axes[0].set_ylabel('Four vs three zones (%)');axes[0].set_title('Constant IP₃');axes[0].legend(frameon=False,fontsize=8)
for col,case in enumerate(('AM','FM'),1):
    d=diag[(case,'reset_from_1.2',0.)]
    for model,color,label in [('three','#222222','3 zones'),('four','#c3313e','4 zones')]:
        z=d[model]['trace'];t=np.asarray(z['time_s']);c=np.asarray(z['C']);m=t<=10
        axes[col].plot(t[m],c[m],c=color,lw=1.1,label=label)
    axes[col].axhline(.025,c='#698777',ls='--',lw=.9);axes[col].axhline(0,c='0.3',lw=.7)
    axes[col].set_title(f'{case}: reset I = 1.2 → 0');axes[col].set_xlabel('Time after reset (s)');axes[col].set_ylabel('C (µM)');axes[col].set_ylim((-.28,.3) if case=='AM' else (-.06,.15))
    if col==1:axes[col].legend(frameon=False,fontsize=8)
for a in axes:a.grid(alpha=.2)
path=ROOT/'paper/figures/lower_guard_comparison';fig.savefig(str(path)+'.pdf',bbox_inches='tight');fig.savefig(str(path)+'.png',dpi=200,bbox_inches='tight')
(ROOT/'paper/data/random_selected/lower_guard_benchmark.json').write_text(json.dumps({'runs_sha256':hashlib.sha256(raw).hexdigest(),'manifest':settings,'records':records,'scope':'finite sample; paired timings; no global physicality guarantee'},indent=2,allow_nan=False)+'\n')
print('Saved',path,'records',len(records))
