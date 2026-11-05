"""Re-solve one joint focus/fold seed with a dependent pivot and validate it."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from anchored_lower_fold import FREE,anchored_fold
from continue_physical_fixed import Orbit,nondegeneracy

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=['AM','AFM','FM']);p.add_argument('--joint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    base=json.loads((ROOT/'paper/data/physical_models/fixed_fold'/a.case/'parameters.json').read_text());audit=json.loads((ROOT/'paper/data/physical_models/fixed_fold/fold_audit.json').read_text())['results'][a.case.lower()];j=json.loads(a.joint.read_text());x=j['best']['x'];times=j['best']['times'];changes={k:float(v*.05*max(abs(base[k[0]][k[1]]),.3)) for k,v in zip(FREE,x)};pivot=('alpha',1);pivot_delta=changes.pop(pivot);cut=.25*min(base['rest_target'][0],base['physical_thresholds_C'][0],audit['neutral']['Cmin'])
    model,_,checks,fold=anchored_fold(base,changes,pivot,cut,0.,0.,initial_fold={'times':times,'pivot_delta':pivot_delta});orb=Orbit(model);T=np.array(model['trace']);D=np.array(model['determinant']);disc=T*T-4*D
    if not (T[0]<0<T[1] and T[2]<0 and D[0]>0 and np.all(disc[1:]<0)):raise ValueError('lower stable / middle and upper focus constraint failed')
    t=np.array(fold['times']);point=orb.point(t,fold['I'],1e-9,require_physical=False);nd=nondegeneracy(orb,np.r_[t,fold['I']]);a.output.mkdir(parents=True,exist_ok=True)
    candidate=copy.deepcopy(model)
    for key in ('status','scope','optimizer','loss','selected','parameters','event_q','fold_constraint','training_cycles','holdout_cycles'):candidate.pop(key,None)
    candidate['status']='exploratory_seed_not_validated';candidate['numerical_fold']=fold
    (a.output/'candidate.json').write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n');report={'case':a.case,'scope':'initial candidate only, not fitted to LR cycle data; sampled route checks','trace':T.tolist(),'discriminant':disc.tolist(),'fold':fold,'orbit':point,'nondegeneracy':nd,'checks':checks,'pivot':pivot,'source_joint':str(a.joint)};(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(a.case,'fold',fold['I'],'logM',fold['logM'],'trace',T.tolist(),'discriminant',disc.tolist(),'closure',point['closure'],flush=True)

if __name__=='__main__':main()
