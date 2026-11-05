"""One focus/fold start and fit, invoked by run_focus_fit_batch.py."""
import argparse,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=['AM','AFM','FM']);p.add_argument('--seed',type=int,required=True);p.add_argument('--screen',required=True,type=Path);p.add_argument('--am-seed-dir',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--max-nfev',type=int,default=80);p.add_argument('--joint-nfev',type=int,default=250);p.add_argument('--start-sigma',type=float,default=.1);p.add_argument('--shape-scale',type=float,default=.1);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);status={'case':a.case,'seed':a.seed,'stage':'start','status':'running'}
    def save(): (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    def run(cmd):
        print('COMMAND',cmd,flush=True);return subprocess.run(cmd,cwd=ROOT,env=os.environ.copy()).returncode
    def script(name,*args):return [sys.executable,'-u',str(ROOT/'scripts'/name),*[str(x) for x in args]]
    save()
    if a.case=='AM':
        seed_dir=a.am_seed_dir
    else:
        joint=a.output/'joint.json';seed_dir=a.output/'seed_model';status['stage']='joint_fold_focus';save()
        cmd=script('find_focus_fold_seed.py','--case',a.case,'--screen',a.screen,'--start',('selected' if a.seed%2==0 else 'spectrum'),'--random-seed',a.seed,'--jitter',2.,'--bounded-times','--focus-weight',20,'--max-nfev',a.joint_nfev,'--output',joint)
        if run(cmd)!=0 or not joint.exists():status.update(status='joint_failed',detail='numerically unresolved; not proof of nonexistence');save();return
        status['stage']='solve_dependent_fold';save()
        if run(script('materialize_focus_fold_seed.py','--case',a.case,'--joint',joint,'--output',seed_dir))!=0:
            status.update(status='fold_or_focus_rejected',detail='numerically unresolved or failed checks');save();return
        status['stage']='find_training_cycles';save()
        if run(script('evaluate_focus_seed.py','--case',a.case,'--candidate',seed_dir/'candidate.json','--report',seed_dir/'report.json','--output',seed_dir/'training.json'))!=0:
            status.update(status='training_solver_failed');save();return
        training=json.loads((seed_dir/'training.json').read_text())['results'];missing=[x['I'] for x in training if not x['cycle']['candidates']]
        if missing:
            status.update(status='no_eligible_training_seed',unresolved_or_nonphysical_inputs=missing,detail='no admissible cycle found by finite search; not proof of nonexistence');save();return
    status['stage']='optimize';save()
    cmd=script('fit_focus_fold_return.py','--case',a.case,'--seed-dir',seed_dir,'--output',a.output/'fit','--seed',a.seed,'--max-nfev',a.max_nfev,'--start-sigma',a.start_sigma,'--shape-scale',a.shape_scale)
    if run(cmd)!=0:status.update(status='optimization_failed',detail='see task log');save();return
    status.update(status='completed',stage='done');save()

if __name__=='__main__':main()
