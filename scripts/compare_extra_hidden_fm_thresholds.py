"""One-hinge screens, adding the threshold separately in C, I, or hidden r."""
import json,time
import numpy as np
from scipy.optimize import least_squares
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.models import ROOT
from validate_uploaded_hidden_fm import CACHE,OUT,simulate
from refit_uploaded_hidden_fm import cycle_period

BASE=OUT/'input_archive/four_region_base.json'
BEST=np.array([-0.2569580657533585,-0.0727607481052785,2.31416593231771,.16330558554235758,.09895364464324688])

def main():
    data=json.loads((CACHE/'data.json').read_text());model=CalciumTransformPWA.load(BASE);protocols=[]
    for q in data['train']:
        if q['name'] not in {'pulse_tau_20.0','periodic','promoted_tau_65.0'}:continue
        with np.load(CACHE/f'{q["name"]}.npz') as f:protocols.append((q,f['t'].copy(),f['x'].copy()))
    for ip3 in [.53,.7,1.12,1.13]:
        with np.load(CACHE/f'fixed_{ip3:g}.npz') as f:
            q=dict(name=f'fixed_{ip3:g}',duration=700. if ip3==.53 else 400.,tau=20.,baseline=ip3,jumps=[])
            protocols.append((q,f['t'].copy(),f['x'].copy()))
    knots={'C':(.25-model.coordinate.reference)/model.coordinate.scale,'I':.8,'r':.4}
    result={'base_parameters':BEST.tolist(),'knots':{'C_uM':.25,'I_uM':.8,'r_latent':.4},'base_metrics':{},'variants':{},
        'note':'Each variant adds one hinge to du,dv,dr. The central knot is fixed and only three weights are fitted; this is screening, not global knot optimization.'}
    for q,tr,truth in protocols:
        init=[1.3,.8,q['baseline']] if q['name'].startswith('fixed_') else None
        t,x=simulate(q,model,'hidden',init,parameters=BEST);target=np.column_stack([np.interp(t,tr,truth[:,j]) for j in range(3)])
        cp,rp=cycle_period(t,x),cycle_period(t,target)
        result['base_metrics'][q['name']]={'C_RMSE':float(np.sqrt(np.mean((x[:,0]-target[:,0])**2))),
            'period_s':float(np.median(cp)) if len(cp) else None,'LR_period_s':float(np.median(rp)) if len(rp) else None}
    for mode,knot in knots.items():
        def residual(w):
            pieces=[]
            for q,tr,truth in protocols:
                init=[1.3,.8,q['baseline']] if q['name'].startswith('fixed_') else None
                t,x=simulate(q,model,'hidden',init,parameters=BEST,hinge={'coordinate':mode,'knot':knot,'weights':w})
                target=np.column_stack([np.interp(t,tr,truth[:,j]) for j in range(3)]);idx=np.arange(0,len(t),5)
                pieces.extend(((x[idx,0]-target[idx,0])/(.1*np.sqrt(len(idx)))).tolist())
                pieces.extend(((x[idx,1]-target[idx,1])/(.15*np.sqrt(len(idx)))).tolist())
                cp,rp=cycle_period(t,x),cycle_period(t,target);n=min(len(cp),len(rp))
                pieces.append(2.*(len(cp)-len(rp)))
                if n:pieces.extend(((cp[-n:]-rp[-n:])/(.7*np.sqrt(n))).tolist())
            pieces.extend((np.asarray(w)*.05).tolist())
            return np.asarray(pieces)
        t0=time.time();fit=least_squares(residual,np.full(3,.05),bounds=([-2]*3,[2]*3),
            max_nfev=4,jac='2-point',x_scale=[.2]*3,diff_step=1e-2,ftol=1e-6,xtol=1e-8,gtol=1e-6)
        detail={}
        for q,tr,truth in protocols:
            init=[1.3,.8,q['baseline']] if q['name'].startswith('fixed_') else None
            t,x=simulate(q,model,'hidden',init,parameters=BEST,hinge={'coordinate':mode,'knot':knot,'weights':fit.x})
            target=np.column_stack([np.interp(t,tr,truth[:,j]) for j in range(3)]);cp,rp=cycle_period(t,x),cycle_period(t,target)
            detail[q['name']]={'C_RMSE':float(np.sqrt(np.mean((x[:,0]-target[:,0])**2))),
                'period_s':float(np.median(cp)) if len(cp) else None,'LR_period_s':float(np.median(rp)) if len(rp) else None}
        result['variants'][mode]={'knot_coordinate':float(knot),'weights_du_dv_dr':fit.x.tolist(),
            'training_loss':float(fit.fun@fit.fun),'nfev':int(fit.nfev),'elapsed_s':time.time()-t0,'metrics':detail}
    (OUT/'extra_thresholds.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['variants'],indent=2))

if __name__=='__main__':main()
