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
from full_cycle_fit import DEFAULTS
from validate_structure import validate
import numpy as np

# Keep full_cycle_fit.DEFAULTS as the historical v3 reconstruction for audits.
# New fits resolve the observed extra-cycle band and the return-map maximum.
REFINED_I_NODES=sorted(DEFAULTS['I_nodes']+[.75,.78])

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
    parser.add_argument('--start-from',type=Path,help='optional previous candidate.json or selected.json')
    parser.add_argument('--analysis-points',type=int,default=81)
    parser.add_argument('--duration',type=float,default=1200.)
    parser.add_argument('--cycle-max-nfev',type=int,default=60)
    parser.add_argument('--return-nodes',type=int,default=65)
    parser.add_argument('--cycle-I',type=float,nargs='+',default=REFINED_I_NODES)
    parser.add_argument('--return-max-time',type=float,default=400.)
    parser.add_argument('--validation-points',type=int,default=81)
    parser.add_argument('--validation-nodes',type=int,default=129)
    parser.add_argument('--validation-max-time',type=float,default=600.)
    args=parser.parse_args()
    if Path(args.name).name!=args.name or args.name in ['.','..']:
        parser.error('--name must be a single directory name')
    if min(args.trials,args.workers,args.max_nfev,args.analysis_points,args.cycle_max_nfev,args.validation_points)<1 or min(args.return_nodes,args.validation_nodes)<5 or min(args.duration,args.return_max_time,args.validation_max_time)<=0:
        parser.error('budgets must be positive')
    if any(not 0<i<=2 for i in args.cycle_I) or any(a>=b for a,b in zip(args.cycle_I,args.cycle_I[1:])):
        parser.error('--cycle-I must be strictly increasing within (0, 2]')
    out=ROOT/'runs'/'fm_am_reduction'/args.name
    if out.exists():parser.error(f'{out} already exists; choose a fresh --name to preserve results')
    (out/'logs').mkdir(parents=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logfile=out/'logs'/f'run_{stamp}.log'
    with logfile.open('w',buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
        try:
            cases=read_json(PACKAGE/'config'/'cases.json'); config=read_json(PACKAGE/'config'/'protocol.json')
            config['protocol_id']='fm_full_cycle_protocol_v4'
            split=max(3,args.return_nodes//3)
            nodes=np.unique(np.r_[np.geomspace(.001,.1,split),np.linspace(.1,.98,args.return_nodes-split+1)])
            config['cycle_fit']={**DEFAULTS,'I_nodes':args.cycle_I,'u_nodes':nodes.tolist(),
                'max_nfev_per_epsilon':args.cycle_max_nfev,'max_time':args.return_max_time}
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
                initial_candidate=str(args.start_from) if args.start_from else None,
                initial_candidate_sha256=hashlib.sha256(args.start_from.read_bytes()).hexdigest() if args.start_from else None,
                fit_scope='L_null + 10 L_spec + 10 L_eq + 30 L_cyc + 30 L_extra; epsilon continuation',
                cycle_fit=config['cycle_fit'],validation_points=args.validation_points,
                validation_nodes=args.validation_nodes,validation_max_time=args.validation_max_time)
            (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
            (out/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
            print('Experiment:',out);print('Log:',logfile);print(json.dumps(run,indent=2))
            prepare_run(out,config)
            warm=None
            if args.start_from:
                loaded=read_json(args.start_from);warm=loaded.get('candidate',loaded)
                if warm.get('case')!='FM':raise ValueError('--start-from must contain an FM model')
                shutil.copy2(args.start_from,snapshot/'initial_candidate.json')
            fit_run(out,config,args.trials,args.max_nfev,args.seed,args.workers,warm)
            analyze(out,args.workers,args.analysis_points,args.duration)
            structural=validate(out,args.workers,args.validation_points,args.validation_nodes,args.validation_max_time)
            chosen=read_json(out/'optimizer'/'selected.json')
            chosen['independent_validation']=structural
            chosen['candidate']['independent_validation_completed']=True
            chosen['candidate']['protocol_status']='fitted_and_checked_research_candidate' if chosen.get('return_domain_admissible') else 'inadmissible_return_domain_diagnostic_only'
            chosen['candidate']['accepted']=False
            (out/'optimizer'/'selected.json').write_text(json.dumps(chosen,indent=2,allow_nan=False)+'\n')
            print('FINISHED; inspect analysis/structure/summary.json and analysis/structure/bifurcation.png')
        except BaseException:
            traceback.print_exc()
            raise

if __name__=='__main__':main()
