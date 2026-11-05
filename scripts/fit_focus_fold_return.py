"""Three-focus, fixed-fold fitting by regional return maps.

Candidates violating numerical constraints or sampled calcium positivity are
rejected. The loss contains Cmin, Cmax, period and peak-aligned cycle shape.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from anchored_lower_fold import FREE,anchored_fold
from return_cycle_metrics import solve_cycle

ROOT=Path(__file__).resolve().parents[1]


def normalized_shape(values):
    values=np.asarray(values,float)
    if values.shape!=(64,) or not np.all(np.isfinite(values)) or np.ptp(values)<1e-7:
        raise ValueError('invalid cycle shape')
    return (values-np.min(values))/np.ptp(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['AM','AFM','FM'],required=True);p.add_argument('--seed-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-nfev',type=int,default=80)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--max-init-tries',type=int,default=20)
    p.add_argument('--start-sigma',type=float,default=.1);p.add_argument('--bound',type=float,default=40.)
    p.add_argument('--shape-scale',type=float,default=.1,help='dimensionless RMS shape error scale')
    a=p.parse_args()
    if min(a.max_nfev,a.max_init_tries)<1 or min(a.start_sigma,a.bound,a.shape_scale)<=0:p.error('invalid settings')
    base_path=ROOT/'paper/data/physical_models/fixed_fold'/a.case/'parameters.json';base=json.loads(base_path.read_text())
    audit=json.loads((ROOT/'paper/data/physical_models/fixed_fold/fold_audit.json').read_text())['results'][a.case.lower()]
    seed=json.loads((a.seed_dir/'candidate.json').read_text());ref=json.loads((a.seed_dir/'report.json').read_text())
    training=json.loads((a.seed_dir/'training.json').read_text())['results'];rows=base['training_cycles']
    if len(training)!=len(rows) or any(abs(x['I']-y['I'])>1e-10 for x,y in zip(training,rows)):raise ValueError('training seed inputs mismatch')
    data=[]
    for entry in training:
        cand=entry['cycle']['candidates']
        if not cand or cand[0]['Cmin']<=0:raise ValueError('seed requires physical stable cycle at each training input')
        data.append(cand[0])
    pivot=('alpha',1);free=[k for k in FREE if k!=pivot]
    scales=np.array([.05*max(abs(base[k][j]),.3) for k,j in free])
    reference=np.array([(seed[k][j]-base[k][j])/v for (k,j),v in zip(free,scales)])
    if max(abs(reference))>=a.bound:raise ValueError('seed outside free-parameter bounds')
    cut=.25*min(base['rest_target'][0],base['physical_thresholds_C'][0],audit['neutral']['Cmin']);hint=ref['fold']
    targets=[normalized_shape(row['target_shape']) for row in rows]
    nres=len(rows)*(3+64);best={'loss':float('inf')};calls=0;fail={}
    a.output.mkdir(parents=True,exist_ok=False)
    def residual(x,record=True):
        nonlocal calls,best
        calls+=1;changes={k:float(v*s) for k,v,s in zip(free,x,scales)}
        try:
            d,_,checks,fold=anchored_fold(base,changes,pivot,cut,0.,0.,initial_fold=hint)
            T=np.array(d['trace']);D=np.array(d['determinant']);disc=T*T-4*D
            if not (T[0]<0<T[1] and T[2]<0 and np.all(disc<-1e-6)):raise ValueError('focus conditions')
            arr=[];results=[]
            for row,seed_cycle,target in zip(rows,data,targets):
                name=seed_cycle['route'];sol=solve_cycle(d,row['I'],fold['times'],extra_seeds={name:[seed_cycle['times']]},route_names=(name,))
                if not sol['candidates']:
                    sol=solve_cycle(d,row['I'],fold['times'],route_names=tuple(v for v in ('both','upper','lower') if v!=name))
                if not sol['candidates']:raise ValueError('unresolved or nonphysical training cycle')
                value=sol['candidates'][0];results.append(value)
                arr.extend([(value['Cmin']-row['target_Cmin'])/.03,(value['Cmax']-row['target_Cmax'])/.08,(value['period']-row['target_period'])/5])
                arr.extend((normalized_shape(value['shape'])-target)/(a.shape_scale*np.sqrt(64)))
            res=np.array(arr);loss=float(res@res)
            if record and loss<best['loss']:
                best={'loss':loss,'x':x.tolist(),'calls':calls,'fold':fold,'checks':checks,'cycles':results,'trace':T.tolist(),'discriminant':disc.tolist()}
                (a.output/'checkpoint.json').write_text(json.dumps(best,indent=2)+'\n')
                candidate=d.copy()
                for key in ('status','scope','optimizer','loss','selected','parameters','event_q','fold_constraint','training_cycles','holdout_cycles'):candidate.pop(key,None)
                candidate['status']='exploratory_fit_not_validated';candidate['numerical_fold']=fold
                (a.output/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
            status='valid'
        except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as e:
            status=str(e);fail[status]=fail.get(status,0)+1
            # Search barrier only, never counted as a valid candidate or reported as its loss.
            res=np.full(nres,100.)+.001*np.resize(x,nres)
        print(a.case,'seed',a.seed,'call',calls,status,'best',best['loss'],flush=True);return res
    rng=np.random.default_rng(a.seed);start=None
    for attempt in range(a.max_init_tries):
        x=reference.copy() if a.seed==0 and attempt==0 else np.clip(reference+rng.normal(0,a.start_sigma,len(free)),-a.bound+.001,a.bound-.001)
        before=best['loss'];r=residual(x)
        if max(abs(r))<50:start=x;break
    if start is None:
        (a.output/'start_failure.json').write_text(json.dumps({'case':a.case,'seed':a.seed,'attempts':a.max_init_tries,'failures':fail},indent=2)+'\n')
        raise RuntimeError('no physical valid start found; this does not prove nonexistence')
    initial_loss=float(r@r)
    sol=least_squares(residual,start,bounds=(np.full(len(free),-a.bound),np.full(len(free),a.bound)),max_nfev=a.max_nfev,diff_step=.001,x_scale='jac',ftol=1e-5,xtol=1e-5,gtol=1e-5)
    # Recompute four held-out points from nearest training solution; unresolved is explicit.
    d=json.loads((a.output/'candidate.json').read_text());holdout=[]
    for row in base['holdout_cycles']:
        near=min(best['cycles'],key=lambda v:abs(v['I']-row['I']));name=near['route']
        result=solve_cycle(d,row['I'],best['fold']['times'],extra_seeds={name:[near['times']]},route_names=(name,))
        if not result['candidates']:result=solve_cycle(d,row['I'],best['fold']['times'],route_names=tuple(v for v in ('both','upper','lower') if v!=name))
        holdout.append({'I':row['I'],'target':{k:row[k] for k in ('target_Cmin','target_Cmax','target_period')},'cycle':result})
    rep={'case':a.case,'seed':a.seed,'scope':'local fit, numeric fold and sampled physical routes; numerical failures not nonexistence',
         'source_sha256':hashlib.sha256(base_path.read_bytes()).hexdigest(),'loss':'sum_i[(dCmin/.03)^2+(dCmax/.08)^2+(dT/5)^2+mean((normalized_shape−target_shape)^2)/shape_scale²]',
         'shape_scale':a.shape_scale,'start_sigma':a.start_sigma,'max_nfev':a.max_nfev,'nfev':int(sol.nfev),'calls':calls,
         'optimizer_success':bool(sol.success),'message':str(sol.message),'initial_loss':initial_loss,'best':best,'holdout':holdout,'failures':fail}
    (a.output/'report.json').write_text(json.dumps(rep,indent=2,allow_nan=False)+'\n')
    print('FINISHED',a.case,'seed',a.seed,'initial',initial_loss,'best',best['loss'],'holdout',sum(bool(x['cycle']['candidates']) for x in holdout),'/',len(holdout),flush=True)

if __name__=='__main__':main()
