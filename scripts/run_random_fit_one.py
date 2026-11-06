"""One random independent field draw, fixed-fold solve and return-map fit."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True,choices=['AM','AFM','FM']);p.add_argument('--seed',required=True,type=int);p.add_argument('--output',required=True,type=Path);p.add_argument('--max-nfev',type=int,default=80);p.add_argument('--joint-nfev',type=int,default=300);p.add_argument('--max-field-draws',type=int,default=1000);p.add_argument('--bound',type=float,default=40.);p.add_argument('--shape-scale',type=float,default=.1);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);status={'case':a.case,'seed':a.seed,'stage':'random_field','status':'running'}
    def save():(a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    def script(name,*args):return [sys.executable,'-u',str(ROOT/'scripts'/name),*[str(x) for x in args]]
    def run(cmd):print('COMMAND',cmd,flush=True);return subprocess.run(cmd,cwd=ROOT,env=os.environ.copy()).returncode
    save();joint=a.output/'joint.json';seed_dir=a.output/'seed_model'
    if run(script('find_random_fold_seed.py','--case',a.case,'--seed',a.seed,'--max-nfev',a.joint_nfev,'--max-field-draws',a.max_field_draws,'--bound',a.bound,'--output',joint))!=0 or not joint.exists():
        status.update(status='joint_failed',detail='numerical failure, not proof of nonexistence');save();return
    j=json.loads(joint.read_text())
    if not isinstance(j.get('best'),dict) or 'x' not in j['best']:
        status.update(status='random_screen_failed',detail='no feasible random screened starting field');save();return
    status['stage']='solve_dependent_fold';save()
    if run(script('materialize_focus_fold_seed.py','--case',a.case,'--joint',joint,'--output',seed_dir))!=0:
        status.update(status='fold_or_focus_rejected',detail='unresolved or failed numerical checks');save();return
    status['stage']='find_training_cycles';save()
    if run(script('evaluate_focus_seed.py','--case',a.case,'--candidate',seed_dir/'candidate.json','--report',seed_dir/'report.json','--output',seed_dir/'training.json'))!=0:
        status.update(status='training_solver_failed');save();return
    training=json.loads((seed_dir/'training.json').read_text())['results'];missing=[x['I'] for x in training if not x['cycle']['candidates']]
    if missing:
        status.update(status='no_eligible_training_seed',unresolved_or_nonphysical_inputs=missing,detail='no admissible cycle found by finite search');save();return
    status['stage']='optimize';save()
    cmd=script('fit_focus_fold_return.py','--case',a.case,'--seed-dir',seed_dir,'--output',a.output/'fit','--seed',a.seed,'--initial-exact','--max-init-tries',1,'--max-nfev',a.max_nfev,'--bound',a.bound,'--shape-scale',a.shape_scale)
    if run(cmd)!=0:status.update(status='optimization_failed',detail='see task log');save();return
    status.update(status='completed',stage='done');save()

if __name__=='__main__':main()
