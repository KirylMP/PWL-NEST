"""Plot recorded isolated-cell dynamics and raw timing for the two-input run."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
selected=json.loads((ROOT/'paper/data/random_selected/manifest.json').read_text())['models']
fig,axes=plt.subplots(3,3,figsize=(11.2,8.0),constrained_layout=True)
summary={'scope':'32 independent astrocytes; three timed runs after warmup; separate one-cell diagnostics', 'cases':{}}
for row,case in enumerate(('AM','AFM','FM')):
    folder=ROOT/f'results/two_inputs_{case}'
    raw=(folder/'runs.jsonl').read_bytes(); runs=[json.loads(x) for x in raw.splitlines()]
    manifest=json.loads((folder/'manifest.json').read_text())
    expected=selected[case]['candidate_sha256']
    if any(r['candidate_sha256']!=expected for r in runs):raise ValueError(f'{case}: candidate checksum mismatch')
    inputs=manifest['settings']['inputs'];out=[]
    for col,I in enumerate(inputs):
        entries=[r for r in runs if r['I']==I];timed={};diagnostics={}
        for model in ('LR','PWA'):
            v=[r for r in entries if r['model']==model and not r['warmup']]
            if len(v)!=3 or any('error' in r['result'] for r in v):raise ValueError('missing or failed run')
            timed[model]=[float(r['result']['run_s']) for r in v]
            diags=[r['diagnostics'][model] for r in entries if r.get('diagnostics') and model in r['diagnostics']]
            if not diags or 'trace' not in diags[0]:raise ValueError('missing diagnostic trace')
            diagnostics[model]=diags[0]
            t=np.array(diags[0]['trace']['time_s']);C=np.array(diags[0]['trace']['C']);mask=t>=40
            axes[row,col].plot(t[mask],C[mask],color={'LR':'#222222','PWA':'#c5353c'}[model],lw=1.1,label=model)
        med={model:float(np.median(v)) for model,v in timed.items()}
        out.append({'I':I,'times':timed,'median':med,'speedup':med['LR']/med['PWA'],
                    'diagnostics':{k:dict(dynamics=v['dynamics'],max_ip3_drift=v['max_ip3_drift']) for k,v in diagnostics.items()}})
        axes[row,col].set_title(f'{case}   I = {I:g} µM',fontsize=10)
        axes[row,col].grid(alpha=.2)
        if row==2:axes[row,col].set_xlabel('Time (s)')
        if col==0:axes[row,col].set_ylabel('C (µM)')
        if row==0 and col==1:axes[row,col].legend(fontsize=8,frameon=False)
    ax=axes[row,2];xs=np.array([0.,1.]);width=.33
    for j,model in enumerate(('LR','PWA')):
        vals=[v['median'][model] for v in out];offset=(-.5+j)*width
        ax.bar(xs+offset,vals,width=width,color=('#555555' if model=='LR' else '#c5353c'),alpha=.75,label=model)
        for i,v in enumerate(out):
            jitter=[-.045,0,.045]
            ax.scatter(xs[i]+offset+np.array(jitter),v['times'][model],s=10,c='black',zorder=4)
    ax.set_xticks(xs,[f"{x['I']:g}" for x in out]);ax.set_title(f'{case} timing',fontsize=10);ax.grid(axis='y',alpha=.2)
    if row==2:ax.set_xlabel('IP₃ I (µM)')
    ax.set_ylabel('Run time (s)')
    if row==0:ax.legend(fontsize=8,frameon=False)
    summary['cases'][case]={'candidate_sha256':expected,'runs_sha256':hashlib.sha256(raw).hexdigest(),'manifest':manifest,'inputs':out}
out=ROOT/'paper/figures/random_selected_nest'
fig.savefig(str(out)+'.pdf',bbox_inches='tight');fig.savefig(str(out)+'.png',dpi=200,bbox_inches='tight')
(ROOT/'paper/data/random_selected/nest_two_inputs_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(out)
