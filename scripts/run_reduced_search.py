"""Run matched compact AM alternatives and a four-threshold FM search."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from pwl_nest.models import ROOT

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=ROOT/'results/threshold_reduction')
p.add_argument('--workers',type=int,default=4)
p.add_argument('--budget',type=int,default=1000)
p.add_argument('--starts',type=int,default=2)
a=p.parse_args()
if min(a.workers,a.budget,a.starts)<1:p.error('Positive workers/budget/starts required')
out=a.output.resolve()
tasks=[('FM_4C','FM',4,False),('AM_cube_2C','AM_cube',2,False),
       ('AM_log_2C','AM_log',2,False),('AM_cube_2C_2I','AM_cube',2,True),
       ('AM_log_2C_2I','AM_log',2,True)]
if any((out/name).exists() for name,_,_,_ in tasks):
    raise ValueError('Search results already exist; choose a new --output directory')
out.mkdir(parents=True,exist_ok=True)
plan=dict(tasks=[dict(name=n,source=s,C_thresholds=c,keep_IP3_thresholds=i) for n,s,c,i in tasks],
    budget_per_stage=a.budget,starts_per_task=a.starts,workers=a.workers,
    selection='Only independent DOP853 training loss; held-out and NEST outputs excluded',
    scientific_changes='Fewer C thresholds, both AM input geometries, unconstrained regional spectral types; exact stable LR background rest. Physical common loss omits spectral boundary anchors for all compared models.')
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
rng=np.random.default_rng(20261027)
times=(np.round((20.+np.r_[0.,np.cumsum(rng.uniform(16.,27.,7))])/.05)*.05).tolist()
protocols={}
for case,baseline,pulse in [('AM',.3,.35),('FM',.45,.7)]:
    specs=[dict(name=f'heldout_tau_{tau:g}',duration=420. if tau==150 else 260.,
                tau=tau,baseline=baseline,jumps=[[20.,pulse]]) for tau in [11.,32.,72.,150.]]
    amplitudes=rng.uniform(.12,.18,len(times)) if case=='AM' else rng.uniform(.2,.3,len(times))
    specs.append(dict(name='heldout_irregular',duration=300.,tau=21.,baseline=baseline,
                     jumps=[[float(t),float(d)] for t,d in zip(times,amplitudes)]))
    protocols[case]=specs
(out/'evaluation_plan.json').write_text(json.dumps(dict(protocols=protocols,
    selection='Never use these diagnostics for fitting or model selection',
    initialization='each candidate own stable rest; reduced model has exact stable LR rest',
    recorded_SIC='delivered I_SIC on AEIF target',resolution_ms=10.,
    timing=dict(resolution_ms=1.,population=64,duration=60.,repeats=3,
                excluded_warmups=1,threads=1)),indent=2)+'\n')
def worker(task):
    name,source,count,keep=task
    command=[sys.executable,str(ROOT/'scripts/fit_reduced_thresholds.py'),'--source',source,
             '--thresholds',str(count),'--output',str(out/name),'--budget',str(a.budget),'--starts',str(a.starts)]
    if keep:command.append('--keep-input')
    with (out/f'{name}.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    return dict(task=name,exit_code=result.returncode)
results=[]
with ThreadPoolExecutor(max_workers=a.workers) as pool:
    futures=[pool.submit(worker,t) for t in tasks]
    for future in as_completed(futures):
        result=future.result();results.append(result)
        (out/'process_results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)
if any(r['exit_code'] for r in results):sys.exit(1)
