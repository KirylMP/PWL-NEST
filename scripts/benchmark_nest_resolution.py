"""Timing-only sweep including NEST's default 0.1 ms tick (no recorded traces)."""
import argparse
import json
from pathlib import Path
import statistics

from benchmark_nest import initial_state, isolated, network_run, protocols
from pwl_nest.nest_adapter import load_candidate
from pwl_nest.models import ROOT

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--resolutions-ms',nargs='+',type=float,default=[.1,1.,10.])
p.add_argument('--population',type=int,default=64)
p.add_argument('--duration',type=float,default=60.)
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--output',type=Path,default=ROOT/'results/nest_benchmark/resolution_sweep.json')
a=p.parse_args()
report=dict(population=a.population,duration=a.duration,repeats=a.repeats,threads=1,
            warmup='one discarded run per kind/mode/resolution',rows=[])
for case in ['AM','FM']:
    model,identity=load_candidate(case)
    protocol=dict(protocols(case)[0],duration=a.duration)
    initial=initial_state(case,protocol['baseline'])
    for res in a.resolutions_ms:
        for mode in ['astrocytes_only','network_with_SIC','isolated_solver']:
            raw=[]
            for repeat in range(a.repeats+1):
                if mode=='isolated_solver':
                    timing=isolated(case,model,initial,protocol,res,a.population)
                    times={k:v['solve_seconds'] for k,v in timing.items()}
                else:
                    timing={}
                    for kind in (['lr','pwa'] if repeat%2==0 else ['pwa','lr']):
                        _,timing[kind]=network_run(case,model,initial,protocol,kind,res,
                            population=a.population,record=False,with_targets=mode=='network_with_SIC',
                            delay_ms=10.)
                    times={k:v['run_seconds'] for k,v in timing.items()}
                if repeat:raw.append(dict(seconds=times,detail=timing))
            med={k:statistics.median(r['seconds'][k] for r in raw) for k in ['lr','pwa']}
            row=dict(case=case,model=identity,resolution_ms=res,mode=mode,raw=raw,
                     median_run_seconds=med,lr_over_pwa=med['lr']/med['pwa'])
            report['rows'].append(row)
            a.output.parent.mkdir(exist_ok=True,parents=True)
            a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ['raw','model']}),flush=True)
