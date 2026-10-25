#!/usr/bin/env python3
"""Fresh FM-only run with persistent logs, provenance, and automatic figures."""
import os
# Must precede NumPy/SciPy imports, including in multiprocessing children.
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import argparse
import contextlib
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback
from plan import ROOT, PACKAGE, read_json
from prepare_reference_data import prepare_run
from fit_stage import fit_run
from analyze_candidate import analyze

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True,help='new experiment directory name')
    parser.add_argument('--trials',type=int,default=64)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--max-nfev',type=int,default=1500)
    parser.add_argument('--seed',type=int,default=20261026)
    parser.add_argument('--analysis-points',type=int,default=81)
    parser.add_argument('--duration',type=float,default=1200.)
    args=parser.parse_args()
    if Path(args.name).name!=args.name or args.name in ['.','..']:
        parser.error('--name must be a single directory name')
    if min(args.trials,args.workers,args.max_nfev,args.analysis_points)<1 or args.duration<=0:
        parser.error('budgets must be positive')
    out=ROOT/'runs'/'fm_am_reduction'/args.name
    if out.exists():parser.error(f'{out} already exists; choose a fresh --name to preserve results')
    (out/'logs').mkdir(parents=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logfile=out/'logs'/f'run_{stamp}.log'
    with logfile.open('w',buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
        try:
            cases=read_json(PACKAGE/'config'/'cases.json'); config=read_json(PACKAGE/'config'/'protocol.json')
            config['protocol_id']='fm_precycle_audited_v2'
            snapshot=out/'source'; snapshot.mkdir()
            hashes={}
            for path in sorted((PACKAGE/'scripts').glob('*.py')):
                shutil.copy2(path,snapshot/path.name)
                hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            shutil.copy2(PACKAGE/'protocol'/'FM_REDUCTION_PROTOCOL.md',snapshot/'FM_REDUCTION_PROTOCOL.md')
            run=dict(run_id=args.name,case='FM',m=7,K_ER=cases['cases']['FM']['K_ER'],
                parameters={**cases['shared_parameters'],'K_ER':cases['cases']['FM']['K_ER']},
                seed=args.seed,restarts=args.trials,workers=args.workers,max_nfev=args.max_nfev,
                analysis_points=args.analysis_points,analysis_duration_s=args.duration,
                protocol_id=config['protocol_id'],code_sha256=hashes,
                fit_scope='L_null + 10 L_spec + 10 L_eq; cycle loss not implemented')
            (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
            (out/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
            print('Experiment:',out);print('Log:',logfile);print(json.dumps(run,indent=2))
            prepare_run(out,config)
            fit_run(out,config,args.trials,args.max_nfev,args.seed,args.workers)
            analyze(out,args.workers,args.analysis_points,args.duration)
            print('FINISHED; inspect analysis/summary.json and analysis/bifurcation.png')
        except BaseException:
            traceback.print_exc()
            raise

if __name__=='__main__':main()
