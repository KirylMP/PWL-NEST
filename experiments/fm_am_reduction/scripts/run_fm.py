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
import time
import traceback
from plan import ROOT, PACKAGE, read_json
from prepare_reference_data import prepare_run
from fit_stage import fit_run
from analyze_candidate import analyze
from full_cycle_fit import DEFAULTS
from validate_structure import validate
from cycle_training_audit import collocation_levels,refine_training_levels,calibrate_source_epsilon,nested_nodes
import numpy as np

# Keep full_cycle_fit.DEFAULTS as the historical v3 reconstruction for audits.
# New fits resolve the observed extra-cycle band and the return-map maximum.
REFINED_I_NODES=sorted(DEFAULTS['I_nodes']+[.75,.78])


def candidate_snapshot(path,attempts=5):
    """Pin one complete input, even if an older run is updating its JSON."""
    for attempt in range(attempts):
        payload=path.read_bytes()
        try:
            loaded=json.loads(payload);model=loaded.get('candidate',loaded)
            if model.get('case')!='FM':raise ValueError('--start-from must contain an FM model')
            return payload,model
        except json.JSONDecodeError:
            if attempt+1==attempts:raise
            time.sleep(.1)

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
    parser.add_argument('--return-workers',type=int,default=0,
        help='processes for independent return-map nodes; 0 divides available CPUs across concurrent trials')
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
    parser.add_argument('--validation-nodes',type=int,default=257)
    parser.add_argument('--validation-max-time',type=float,default=600.)
    parser.add_argument('--cycle-epsilon',type=float,nargs='+',default=[.5,.1,.01,.001])
    parser.add_argument('--max-I-gap',type=float,default=.05,help='coarse training spacing; final grid also contains all midpoints')
    parser.add_argument('--max-stage-nodes',type=int,default=129)
    parser.add_argument('--selection-nodes',type=int,default=129)
    parser.add_argument('--profile-nodes',type=int,default=65)
    parser.add_argument('--count-tolerance',type=float,default=.025)
    parser.add_argument('--stationarity-gtol',type=float,default=1e-6)
    parser.add_argument('--training-I-refinements',type=int,default=1)
    parser.add_argument('--common-polish-nfev',type=int,help='joint final-grid budget; defaults to cycle-max-nfev')
    args=parser.parse_args()
    if Path(args.name).name!=args.name or args.name in ['.','..']:
        parser.error('--name must be a single directory name')
    if min(args.trials,args.workers,args.max_nfev,args.analysis_points,args.cycle_max_nfev,args.validation_points)<1 or args.return_workers<0 or min(args.return_nodes,args.validation_nodes)<5 or min(args.duration,args.return_max_time,args.validation_max_time)<=0:
        parser.error('budgets must be positive')
    if any(not 0<i<=2 for i in args.cycle_I) or any(a>=b for a,b in zip(args.cycle_I,args.cycle_I[1:])):
        parser.error('--cycle-I must be strictly increasing within (0, 2]')
    if not args.cycle_epsilon or any(not np.isfinite(e) or e<=0 for e in args.cycle_epsilon) or any(a<=b for a,b in zip(args.cycle_epsilon,args.cycle_epsilon[1:])):
        parser.error('--cycle-epsilon must be positive and strictly decreasing')
    if min(args.max_I_gap,args.count_tolerance,args.stationarity_gtol)<=0 or args.profile_nodes<5:
        parser.error('grid scales/tolerances must be positive; profile nodes at least five')
    if any(not np.isfinite(x) for x in [args.max_I_gap,args.count_tolerance,args.stationarity_gtol,args.duration,args.return_max_time,args.validation_max_time]):
        parser.error('grid scales, tolerances and time budgets must be finite')
    if args.count_tolerance>=.5:parser.error('--count-tolerance must be below .5')
    if args.max_stage_nodes<args.return_nodes or args.selection_nodes<args.max_stage_nodes or args.validation_nodes<=args.selection_nodes:
        parser.error('require return-nodes <= max-stage-nodes <= selection-nodes < validation-nodes')
    if args.start_from and not args.start_from.is_file():parser.error('--start-from file does not exist')
    if args.training_I_refinements<0:parser.error('--training-I-refinements must be nonnegative')
    if args.common_polish_nfev is not None and args.common_polish_nfev<1:parser.error('--common-polish-nfev must be positive')
    split=max(3,args.return_nodes//3)
    initial_nodes=np.unique(np.r_[np.geomspace(.00001,.1,split),np.linspace(.1,.98,args.return_nodes-split+1)])
    actual_selection_nodes=len(nested_nodes(initial_nodes,args.selection_nodes))
    if args.validation_nodes<=actual_selection_nodes:parser.error('--validation-nodes must exceed the actual nested selection grid')
    out=ROOT/'runs'/'fm_am_reduction'/args.name
    if out.exists():parser.error(f'{out} already exists; choose a fresh --name to preserve results')
    (out/'logs').mkdir(parents=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logfile=out/'logs'/f'run_{stamp}.log'
    with logfile.open('w',buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
        try:
            cases=read_json(PACKAGE/'config'/'cases.json'); config=read_json(PACKAGE/'config'/'protocol.json')
            warm=None;initial_payload=None
            if args.start_from:
                initial_payload,warm=candidate_snapshot(args.start_from)
            try:available_cpus=len(os.sched_getaffinity(0))
            except AttributeError:available_cpus=os.cpu_count() or 1
            active_slots=min(args.workers,args.trials+int(warm is not None))
            return_workers=args.return_workers or max(1,available_cpus//active_slots)
            config['protocol_id']='fm_full_cycle_protocol_v5'
            nodes=initial_nodes
            coarse,fine=collocation_levels(args.cycle_I,args.max_I_gap,warm)
            config['cycle_smoothing']['epsilon_schedule']=args.cycle_epsilon
            config['cycle_fit']={**DEFAULTS,'I_nodes':fine,'coarse_I_nodes':coarse,'requested_I_nodes':args.cycle_I,
                'u_nodes':nodes.tolist(),'section_policy':'moving_equilibrium',
                'max_nfev_per_epsilon':args.cycle_max_nfev,'max_time':args.return_max_time,
                'max_stage_nodes':args.max_stage_nodes,'selection_nodes':actual_selection_nodes,
                'requested_selection_nodes':args.selection_nodes,
                'profile_nodes':len(nested_nodes(nodes,max(args.profile_nodes,len(nodes)))),'reference_workers':args.workers,
                'return_workers':return_workers,
                'requested_profile_nodes':args.profile_nodes,
                'count_tolerance':args.count_tolerance,'stationarity_gtol':args.stationarity_gtol,
                'rescue_evaluations':6,'max_I_gap':args.max_I_gap,'I_refinement_rounds':args.training_I_refinements,
                'common_polish_nfev':args.common_polish_nfev or args.cycle_max_nfev}
            config['cycle_fit']['min_regular_edge_fraction']=None
            config['cycle_fit']['regular_domain_rule']='two regular edges plus separate coverage/resolution audit'
            snapshot=out/'source'; snapshot.mkdir()
            hashes={}
            for path in sorted((PACKAGE/'scripts').glob('*.py')):
                shutil.copy2(path,snapshot/path.name)
                hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            shutil.copy2(PACKAGE/'protocol'/'FM_REDUCTION_PROTOCOL.md',snapshot/'FM_REDUCTION_PROTOCOL.md')
            if initial_payload is not None:(snapshot/'initial_candidate.json').write_bytes(initial_payload)
            run=dict(run_id=args.name,case='FM',m=7,K_ER=cases['cases']['FM']['K_ER'],
                parameters={**cases['shared_parameters'],'K_ER':cases['cases']['FM']['K_ER']},
                seed=args.seed,restarts=args.trials,workers=args.workers,max_nfev=args.max_nfev,
                analysis_points=args.analysis_points,analysis_duration_s=args.duration,
                protocol_id=config['protocol_id'],code_sha256=hashes,
                initial_candidate=str(args.start_from) if args.start_from else None,
                initial_candidate_sha256=hashlib.sha256(initial_payload).hexdigest() if initial_payload is not None else None,
                fit_scope='L_null + 10 L_spec + 10 L_eq + 30 L_cyc + 30 L_extra; epsilon continuation',
                cycle_fit=config['cycle_fit'],validation_points=args.validation_points,
                validation_nodes=args.validation_nodes,validation_max_time=args.validation_max_time)
            (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
            (out/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
            print('Experiment:',out);print('Log:',logfile);print(json.dumps(run,indent=2))
            print(f'Return-map node workers: {return_workers} across at most {active_slots} concurrent trial/incumbent processes (CPU budget {available_cpus})',flush=True)
            prepare_run(out,config)
            if args.training_I_refinements:
                refine_training_levels(out,config,warm,args.training_I_refinements)
            calibrate_source_epsilon(out,config)
            # No optimizer has run yet. The common grids and calibrated final
            # smoothing are now frozen for all starts and their ranking.
            run['cycle_fit']=config['cycle_fit'];run['epsilon_schedule']=config['cycle_smoothing']['epsilon_schedule']
            (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
            (out/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
            fit_run(out,config,args.trials,args.max_nfev,args.seed,args.workers,warm)
            analyze(out,args.workers,args.analysis_points,args.duration)
            structural=validate(out,args.workers,args.validation_points,args.validation_nodes,args.validation_max_time)
            chosen=read_json(out/'optimizer'/'selected.json')
            chosen['independent_validation']=structural
            chosen['candidate']['independent_validation_completed']=True
            chosen['candidate']['protocol_status']=('stationary_checked_research_candidate'
                if chosen.get('numerical_audit_passed') and chosen['candidate'].get('converged_stationary')
                else 'nonstationary_checked_research_candidate' if chosen.get('numerical_audit_passed')
                else 'numerical_cycle_audit_failed_diagnostic_only')
            chosen['candidate']['accepted']=False
            (out/'optimizer'/'selected.json').write_text(json.dumps(chosen,indent=2,allow_nan=False)+'\n')
            print('FINISHED; inspect analysis/structure/summary.json and analysis/structure/bifurcation.png')
        except BaseException:
            traceback.print_exc()
            raise

if __name__=='__main__':main()
