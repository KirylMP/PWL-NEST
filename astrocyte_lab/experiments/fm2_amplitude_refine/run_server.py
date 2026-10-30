#!/usr/bin/env python3
"""FM2 multi-start continuation with a separate peak-calcium term."""
import argparse, concurrent.futures, json, os, shutil, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FREE=ROOT/'astrocyte_lab/experiments/free_hidden_refine'

def stats(path):
    data=json.loads(path.read_text())
    result={}
    for split in ('train','holdout'):
        rows=[r for r in data['cycles'] if r['split']==split]
        valid=[r for r in rows if r['PWA'] is not None and r.get('error')]
        result[split]={'total':len(rows),'valid':len(valid)}
        for metric in ('Cmin','Cmax','period','shape_rmse'):
            result[split]['mean_abs_'+metric]=sum(abs(r['error'][metric]) for r in valid)/len(valid) if valid else None
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prior-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--top',type=int,default=4);p.add_argument('--starts',type=int,default=4);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--peak-weight',type=float,default=1.0);p.add_argument('--jitter',type=float,default=.015)
    p.add_argument('--powell-maxfev',type=int,default=6000);p.add_argument('--ls-max-nfev',type=int,default=400)
    p.add_argument('--fd-relative',type=float,default=.001);p.add_argument('--duration',type=float,default=350)
    p.add_argument('--validation-duration',type=float,default=450);p.add_argument('--dt',type=float,default=.01);p.add_argument('--validate-top',type=int,default=4)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if min(a.top,a.starts,a.workers,a.validate_top,a.powell_maxfev,a.ls_max_nfev)<1 or min(a.duration,a.validation_duration)<100 or min(a.dt,a.fd_relative)<=0 or min(a.peak_weight,a.jitter)<0:raise ValueError('Invalid options')
    prior=a.prior_run.resolve();summary=json.loads((prior/'summary.json').read_text())
    options=sorted((r for r in summary['trials'] if abs(r['K']-.05)<1e-9 and r['status']=='training_cycles_found' and r.get('loss') is not None),key=lambda r:r['loss'])[:a.top]
    if not options:raise ValueError('No FM candidates in prior summary')
    engine=a.out/'shape_engine.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(FREE/'shape_engine.cpp'),'-o',str(engine)],check=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(a.out/'mpl'),XDG_CACHE_HOME=str(a.out/'cache'))
    tasks=[]
    for rank,row in enumerate(options):
        source=row['source_trial'];seed=prior/'K_0.050'/f'trial_{source:03d}'/'parameters.json'
        if not seed.is_file():raise FileNotFoundError(seed)
        for j in range(a.starts):
            # One unchanged start per source; reproducible independent perturbations.
            tasks.append((rank,source,j,seed,0.0 if j==0 else a.jitter))
    print(f'FM2 {len(tasks)} starts; workers={a.workers}; peak_weight={a.peak_weight}; output={a.out}',flush=True)
    def job(task):
        rank,source,j,seed,jitter=task;folder=a.out/f'seed_{source:03d}_start_{j:02d}';folder.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-u',str(HERE/'refine.py'),'--seed-model',str(seed),'--engine',str(engine),'--out',str(folder),'--powell-maxfev',str(a.powell_maxfev),'--ls-max-nfev',str(a.ls_max_nfev),'--fd-relative',str(a.fd_relative),'--duration',str(a.duration),'--dt',str(a.dt),'--peak-weight',str(a.peak_weight),'--jitter',str(jitter),'--random-seed',str(10000+rank*100+j)]
        with (folder/'fit.log').open('w') as f:proc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        path=folder/'parameters.json';result=json.loads(path.read_text()) if path.exists() else {}
        row={'source_trial':source,'start':j,'jitter':jitter,'exit_code':proc.returncode,'status':result.get('status','process_error'),'loss':result.get('loss'),'initial_loss':result.get('initial_loss'),'path':str(path),'log':str(folder/'fit.log'),'optimizers':result.get('optimizers')}
        print('TRIAL',json.dumps(row),flush=True);return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(job,tasks))
    candidates=sorted([r for r in rows if r['exit_code']==0 and r['status']=='training_cycles_found' and r['loss'] is not None],key=lambda r:r['loss'])
    output={'scope':'FM2; weighted training loss selection; finite independent validation; no fold or global cycle count. Holdout is reported for comparison, not used in training.', 'source':str(prior),'settings':vars(a)|{'prior_run':str(prior),'out':str(a.out)},'trials':rows,'validation':{}}
    if not candidates:
        (a.out/'summary.json').write_text(json.dumps(output,indent=2)+'\n');raise SystemExit('No valid training candidates; inspect trial logs')
    # Validate the best few and the exact previous winner using identical solver settings.
    validations=[('previous',prior/'K_0.050/model/parameters.json')]+[(f'rank_{i+1:02d}',Path(r['path'])) for i,r in enumerate(candidates[:a.validate_top])]
    for label,src in validations:
        dest=a.out/'comparison'/label;dest.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest/'parameters.json')
        with (dest/'validate.log').open('w') as f:proc=subprocess.run([sys.executable,'-u',str(FREE/'validate.py'),str(dest/'parameters.json'),'--duration',str(a.validation_duration)],env=env,stdout=f,stderr=subprocess.STDOUT)
        record={'exit_code':proc.returncode,'source':str(src),'path':str(dest/'validation.json'),'plot':str(dest/'comparison.png'),'log':str(dest/'validate.log')}
        if proc.returncode==0 and (dest/'validation.json').exists():record['metrics']=stats(dest/'validation.json')
        output['validation'][label]=record;print('VALIDATION',label,json.dumps(record),flush=True)
    output['winner']=candidates[0]
    (a.out/'summary.json').write_text(json.dumps(output,indent=2)+'\n');print('SAVED',a.out/'summary.json',flush=True)
    if any(v['exit_code'] for v in output['validation'].values()):raise SystemExit('Validation failed; inspect logs')
if __name__=='__main__':main()
