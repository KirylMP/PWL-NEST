"""Add one continuous IP3 hinge; calibrate only on autonomous LR equilibria."""
import copy
import hashlib
import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.strict_pwa import lr_field, equilibrium_h

SOURCES={
    'full_AM':'models/server_candidates/AM_sqrt_tails__cube.json',
    'AM_2C':'models/threshold_reduction/selected_AM_2C.json',
    'AM_2C_2I':'models/threshold_reduction/selected_AM_2C_2I.json',
}
OUT=ROOT/'results/high_ip3_hinge'


def fit():
    OUT.mkdir(parents=True,exist_ok=True)
    dest=ROOT/'models/high_ip3_hinge';dest.mkdir(parents=True,exist_ok=True)
    # Chosen before network evaluation; network trajectories/spikes are not fitted.
    ips=np.array([2.,4.,8.,16.,32.,48.])
    states=[]
    for ip in ips:
        c=brentq(lambda c:lr_field([c,equilibrium_h(c,ip),ip],'AM')[0],.001,1.6)
        states.append([c,equilibrium_h(c,ip),ip])
    states=np.asarray(states)
    report=dict(train_IP3=ips.tolist(),train_equilibria=states.tolist(),
                method='One continuous additive hinge w*(IP3-k)+; '
                       'k>=1.3, fit equilibrium latent-coordinate residuals only',models={})
    for name,path in SOURCES.items():
        m=CalciumTransformPWA.load(ROOT/path)
        y=m.to_latent(states)
        q=m.geometry_coefficients().reshape(-1,2,4)[-1]
        a=q[:,1:3]
        def solve(k):
            # Solve equilibrium-coordinate error, not a network spike objective.
            original=np.linalg.solve(a,-q[:,0,None]-q[:,3,None]*ips).T
            h=np.maximum(0,ips-k)
            correction=np.sum(h[:,None]*(y[:,:2]-original),axis=0)/np.sum(h*h)
            w=-a@correction
            pred=original+h[:,None]*correction
            scale=np.maximum(np.ptp(y[:,:2],axis=0),.1)
            loss=float(np.mean(((pred-y[:,:2])/scale)**2))
            return loss,w,pred
        opt=minimize_scalar(lambda k:solve(k)[0],bounds=(1.3,1.9),method='bounded',
                            options=dict(xatol=1e-9))
        choices=[1.3,float(opt.x),1.9]
        k=min(choices,key=lambda k:solve(k)[0]);loss,w,pred=solve(k)
        new=copy.deepcopy(m)
        new.input_thresholds=np.r_[m.input_thresholds,k]
        new.input_weights=np.column_stack([m.input_weights,w])
        new.__post_init__()
        new.metadata['high_IP3_hinge']=dict(threshold=k,weights=w.tolist(),
             source=path,train_IP3=ips.tolist(),unchanged_for_IP3_at_most=1.3)
        output=dest/f'{name}.json';new.save(output)
        # Exact equality of the field over the old input range, independent of C,h.
        probe=y.copy();probe[:,2]=np.linspace(0,1.3,len(y))
        np.testing.assert_array_equal(m.field(probe),new.field(probe))
        report['models'][name]=dict(threshold=k,weights=w.tolist(),train_loss=loss,
             predicted_equilibria_latent=pred.tolist(),source=path,
             source_sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
             output=str(output.relative_to(ROOT)),output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
             unchanged_old_field=True)
        print(name,report['models'][name],flush=True)
    (OUT/'fit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':fit()
