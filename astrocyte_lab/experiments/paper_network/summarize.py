#!/usr/bin/env python3
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);out=[]
for path in sorted(root.glob('*/*/record_*ms/network_summary.json')):
 data=json.loads(path.read_text());r=data['results']['am'];setting=data['settings'];mode,coord=path.relative_to(root).parts[:2]
 out.append(dict(mode=mode,coordinate=coord,record_interval_ms=setting['record_interval'],dt_ms=setting['dt'],threads=setting['threads'],duration_s=setting['duration'],neuron_count=setting['neurons'],astro_count=setting['neurons']*(setting['neurons']-1),lr_median_s=r['median_lr_s'],pwa_median_s=r['median_pwa_s'],lr_over_pwa=r['speedup'],lr_runs_s=[v['times_seconds']['run'] for v in r['lr']],pwa_runs_s=[v['times_seconds']['run'] for v in r['pwa']],graph=r['pwa'][0]['graph'],candidate_sha256=data['candidate_manifest']['am']['sha256'],path=str(path)))
index={(r['mode'],r['coordinate'],r['record_interval_ms']):r for r in out};comparison=[]
for mode in ('am','afm','fm'):
 for interval in sorted({r['record_interval_ms'] for r in out if r['mode']==mode}):
  a=index.get((mode,'physical',interval));b=index.get((mode,'mixed',interval))
  if a and b:comparison.append(dict(mode=mode,record_interval_ms=interval,physical_over_mixed=a['pwa_median_s']/b['pwa_median_s'],mixed_over_physical=b['pwa_median_s']/a['pwa_median_s'],same_topology=a['graph']==b['graph'],note='Different fitted dynamics; this ratio includes model dynamics, SIC, observation and network effects.'))
result=dict(scope='NEST all-to-all closed-loop network; whole nest.Run timing, all astrocytes observed. Ratio does not isolate mathematical transform cost.',rows=out,comparisons=comparison)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
