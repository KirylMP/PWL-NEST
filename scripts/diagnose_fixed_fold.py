"""Compare original and transformed fields on the published varying input."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from fit_transformed_coordinates import KSETS,lr,equilibrium,features,inverse

ROOT=Path(__file__).resolve().parents[1]
SEGMENTS=[(0.,80.,.2),(80.,160.,.58),(160.,240.,.3),(240.,320.,.75),(320.,400.,.2)]

def solve(case,kind,fold_seeds=None):
    old=json.loads((ROOT/'paper/data/random_selected'/case/'candidate.json').read_text())
    if kind=='physical':
        state=np.array(old['rest_target']);cuts=old['physical_thresholds_C']
        def f(I,y):
            C,q=y;v=np.r_[1.,C,I,np.maximum(C-np.array(cuts),0.)]
            return np.array([old['a']*(q-np.dot(old['alpha'],v)),old['b']*(np.dot(old['beta'],v)-q)])
        physical=lambda y:y[0]
    else:
        if fold_seeds:
            seed=fold_seeds[(case,kind)];model=json.loads((ROOT/'results/fold_anchors'/case/kind/f'seed_{seed}_verification.json').read_text())['candidate_model']
        else:model=json.loads((ROOT/'results/coordinate_experiment'/f'{case}_{kind}.json').read_text())['model']
        state=np.array(model['rest_state']);a=np.array(model['coefficient_rows']);cuts=model['cuts_u']
        def f(I,y):return a@features(y[0],y[1],I,cuts)
        physical=lambda y:inverse(y[0],kind)
    times=[];C=[];U=[];events=[]
    for start,end,I in SEGMENTS:
        def output(t,y):return physical(y)-.2
        output.direction=1
        t=np.linspace(start,end,round((end-start)*20)+1)
        sol=solve_ivp(lambda t,y:f(I,y),(start,end),state,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.05,t_eval=t,events=output)
        if not sol.success or not np.isfinite(sol.y).all():raise RuntimeError(sol.message)
        times.extend(t if not times else t[1:]);C.extend(physical(sol.y) if not C else physical(sol.y)[1:]);U.extend(sol.y[0] if not U else sol.y[0,1:]);events.extend(sol.t_events[0].tolist());state=sol.y[:,-1]
    C=np.array(C);U=np.array(U);times=np.array(times);i=np.argmin(C)
    return {'t':times,'C':C,'u':U,'event_times_s':events,'Cmin_sampled':float(C[i]),'tmin_s':float(times[i]),'umin_sampled':float(np.min(U)), 'u_crosses_zero_sampled':bool(np.min(U)<0<np.max(U))}

def main():
    out={};arrays={};fold_seeds={('AM','log'):6,('AM','square'):3,('AFM','log'):2,('AFM','square'):3,('FM','log'):7,('FM','square'):0}
    for case in KSETS:
        out[case]={}
        for kind in ('physical','log','square'):
            try:
                d=solve(case,kind,fold_seeds); arrays[case+'_time']=d.pop('t');arrays[case+'_'+kind+'_C']=d.pop('C');d.pop('u');out[case][kind]=d
            except Exception as e:out[case][kind]={'error':repr(e)}
            print(case,kind,out[case][kind],flush=True)
    (ROOT/'results/fold_anchors/dynamic.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(ROOT/'results/fold_anchors/dynamic.npz',**arrays)
if __name__=='__main__':main()
