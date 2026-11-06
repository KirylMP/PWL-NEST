"""Summarize paired 3-/4-zone timings and plot a curve for each regime."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
manifest=json.loads((root/'manifest.json').read_text());data=defaultdict(lambda:defaultdict(list));diag=defaultdict(dict)
for line in (root/'runs.jsonl').read_text().splitlines():
    r=json.loads(line);key=(r['case'],r['kind'],r['I']);model=r['model']
    if not r['warmup']:data[key][model].append(r['result'])
    if r['diagnostic'] is not None:diag[key][model]=r['diagnostic']
result=[]
for (case,kind,I),models in sorted(data.items()):
    if any(len(models[m])!=manifest['settings']['repeats'] for m in ('three','four')):raise ValueError('incomplete timed repeats')
    d=diag[(case,kind,I)]
    if set(d)!=set(('three','four')):raise ValueError('missing diagnostics')
    for model in ('three','four'):
        if any(x['events_emitted']!=x['events_recorded'] for x in models[model]):raise ValueError('event count mismatch')
    if kind=='constant' and d['four']['guard_crossings']==0 and d['four']['min_C']>=manifest['settings']['cut']:
        x=np.asarray(d['three']['trace']['C']);y=np.asarray(d['four']['trace']['C'])
        if len(x)!=len(y) or np.max(np.abs(x-y))>1e-7:raise ValueError('unexpected trajectory discrepancy above guard')
    med={m:statistics.median(v['run_s'] for v in models[m]) for m in ('three','four')}
    if med['three']<=0:raise ValueError('invalid runtime')
    result.append({'case':case,'kind':kind,'I':I,'median_three_s':med['three'],'median_four_s':med['four'],
                   'four_over_three':med['four']/med['three'],'overhead_percent':100*(med['four']/med['three']-1),
                   'guard_crossings_total':models['four'][0]['guard_crossings'],
                   'events_three':models['three'][0]['events_recorded'],'events_four':models['four'][0]['events_recorded'],
                   'diagnostic_min_three':d['three']['min_C'],'diagnostic_min_four':d['four']['min_C']})
with (root/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
print('case kind I median_three median_four overhead% guard_crossings min_C_three min_C_four')
for v in result:print(v['case'],v['kind'],v['I'],round(v['median_three_s'],4),round(v['median_four_s'],4),round(v['overhead_percent'],2),v['guard_crossings_total'],round(v['diagnostic_min_three'],6),round(v['diagnostic_min_four'],6))
fig,axes=plt.subplots(3,1,figsize=(7,9),sharex=True,constrained_layout=True)
for case in ('AM','AFM','FM'):
    rows=[v for v in result if v['case']==case and v['kind']=='constant']
    if rows:
        line=axes[0].plot([v['I'] for v in rows],[v['overhead_percent'] for v in rows],marker='o',label=case)[0]
        color=line.get_color()
        axes[1].plot([v['I'] for v in rows],[v['median_three_s'] for v in rows],ls='--',c=color)
        axes[1].plot([v['I'] for v in rows],[v['median_four_s'] for v in rows],marker='o',c=color)
        axes[2].plot([v['I'] for v in rows],[v['guard_crossings_total']/manifest['settings']['nodes'] for v in rows],marker='o',c=color)
axes[0].axhline(0,color='gray',lw=.8);axes[0].set_ylabel('Four vs three (%)');axes[0].legend(frameon=False,ncol=3)
axes[1].set_ylabel('Run time (s)');axes[1].set_title('Dashed: three zones; solid: four zones',fontsize=9)
axes[2].set_ylabel('Guard crossings / cell');axes[2].set_xlabel('IP₃ I (µM)')
for ax in axes:ax.grid(alpha=.2)
fig.savefig(root/'overhead_vs_ip3.png',dpi=180);fig.savefig(root/'overhead_vs_ip3.pdf');print('Saved summary.csv and overhead_vs_ip3 figure')
