#!/usr/bin/env python3
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);data={k:json.loads((root/k/'summary.json').read_text()) for k in ('physical','mixed')}
if data['physical']['neuron_edge_sha256']!=data['mixed']['neuron_edge_sha256']:raise ValueError('Neuronal connectivity differs')
rows=[];fig,ax=plt.subplots(figsize=(7,4))
for mode,d in data.items():
 xs=sorted(map(int,d['ratios']));ys=[]
 for x in xs:
  r=d['ratios'][str(x)];ys.append(r['speedup']);rows.append(dict(coordinate=mode,ratio=x,astrocytes=x*d['settings']['neurons'],lr_median_s=r['lr_median_s'],pwa_median_s=r['pwa_median_s'],speedup=r['speedup'],lr_first_trace_peaks=r['lr'][0]['first_astro_peaks'],pwa_first_trace_peaks=r['pwa'][0]['first_astro_peaks'],lr_ip3_range=r['lr'][0]['sampled_IP3_range'],pwa_ip3_range=r['pwa'][0]['sampled_IP3_range'],lr_C_range=r['lr'][0]['first_astro_C_range'],pwa_C_range=r['pwa'][0]['first_astro_C_range']))
 ax.plot(xs,ys,'o-',label=mode)
ax.axhline(1,color='gray',lw=.8);ax.set(xlabel='Independent astrocytes per neuron',ylabel='Network speedup: LR / PWA',title=f"Fixed network: {data['physical']['settings']['neurons']} neurons");ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(root/'speedup_vs_ratio.png',dpi=150)
summary=dict(scope='Whole network nest.Run, same fixed neuronal topology, normalized SIC per neuron, three repeats; trace peaks indicate finite observed oscillations. Different model dynamics may change network activity.',rows=rows,neuron_edge_sha256=data['physical']['neuron_edge_sha256'],details={k:str(root/k/'summary.json') for k in data})
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
