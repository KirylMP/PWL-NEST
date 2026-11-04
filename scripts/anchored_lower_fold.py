"""Exact equilibrium anchors plus fixed-input fold for lower-zone models.

The base has two old C thresholds. The new threshold is strictly below them.
Four coefficients are eliminated by linear equalities: the field is zero at
the specified rest state at I=0, and an equilibrium lies on each old threshold
at its specified input. A fifth coefficient and four flight times are solved
from closure and log(M)=0 of the fold. The fold must remain above the new cut.
"""
import copy
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from continue_physical_fixed import Orbit, REGIONS
from fit_lower_fixed_fold import make_model, rhs

ELIMINATED={('alpha',0),('beta',0),('alpha',2),('alpha',3)}
FREE=[(k,j) for k in ('alpha','beta') for j in range(5) if (k,j) not in ELIMINATED]


def anchor_model(base, changes, cut, da, db):
    d=copy.deepcopy(base)
    for (key,j),v in changes.items():
        if (key,j) in ELIMINATED: raise ValueError('cannot vary eliminated coefficient')
        d[key][j]+=float(v)
    if not 0<cut<base['physical_thresholds_C'][0]:raise ValueError('invalid lower cut')
    model=make_model(d,cut,da,db)
    C0,q0=base['rest_target']; old=base['physical_thresholds_C']; I1,I2=base['fixed_equilibrium_threshold_I']
    n=np.array(model['alpha']);h=np.array(model['beta']); cuts=np.array(model['physical_thresholds_C'])
    def basis(C,I):return np.r_[1.,C,I,np.maximum(C-cuts,0.)]
    v0=basis(C0,0.); v1=basis(old[0],I1);v2=basis(old[1],I2)
    # Unknowns: delta(alpha[0]), delta(beta[0]), delta(alpha[2]),
    # delta(alpha[4]); index 4 is the first old hinge after inserting new cut.
    def equations(n,h):return np.array([n@v0-q0,h@v0-q0,(n-h)@v1,(n-h)@v2])
    A=np.array([[1.,0.,v0[2],v0[4]],[0.,1.,0.,0.],
                [1.,-1.,v1[2],v1[4]],[1.,-1.,v2[2],v2[4]]])
    delta=np.linalg.solve(A,-equations(n,h))
    n[0]+=delta[0];h[0]+=delta[1];n[2]+=delta[2];n[4]+=delta[3]
    model['alpha']=n.tolist();model['beta']=h.tolist()
    # Recompute matrices after coefficient elimination.
    a,b=model['a'],model['b']; mats=[];slopes=[]
    for j in range(4):
        ns=n[1]+sum(n[3:3+j]);hs=h[1]+sum(h[3:3+j]);
        mats.append(np.array([[-a*ns,a],[b*hs,-b]]))
        if abs(ns-hs)<1e-12:raise ValueError('degenerate equilibrium slope')
        slopes.append(float(-(n[2]-h[2])/(ns-hs)))
    model['trace']=[float(np.trace(A)) for A in mats]
    model['determinant']=[float(np.linalg.det(A)) for A in mats]
    residual=equations(n,h)
    if np.max(np.abs(residual))>1e-9:raise ValueError('equilibrium anchor error')
    # Local equilibrium crosses the old cuts in the original direction, with
    # stable lower regions, unstable middle and stable upper region.
    if not (all(x>1e-8 for x in model['determinant']) and all(x>1e-8 for x in slopes)
            and model['trace'][0]<-1e-8 and model['trace'][1]<-1e-8
            and model['trace'][2]>1e-8 and model['trace'][3]<-1e-8):
        raise ValueError('regional stability or equilibrium crossing orientation changed')
    # For C>=cut absorb new hinge into the low-region intercept/slope.
    three=copy.deepcopy(base)
    for key,arr in [('alpha',n),('beta',h)]:
        three[key]=[float(arr[0]-arr[3]*cut),float(arr[1]+arr[3]),float(arr[2]),float(arr[4]),float(arr[5])]
    three['trace']=model['trace'][1:];three['determinant']=model['determinant'][1:]
    checks={'anchor_residuals':residual.tolist(),'rest_rhs':rhs(model,0.,base['rest_target']),
            'equilibrium_slopes':slopes,'region_trace':model['trace'],'region_determinant':model['determinant']}
    return three,model,checks


def anchored_fold(base,changes,pivot,cut,da,db):
    if pivot not in FREE:raise ValueError('pivot is not free')
    I=base['fold_constraint']['I']; original=np.array(base['fold_constraint']['crossing_times'])
    seed=np.r_[original,float(changes.get(pivot,0.))]
    scale=np.r_[np.maximum(original,.2),max(abs(base[pivot[0]][pivot[1]]),.5)]
    def equations(z):
        trial=dict(changes);trial[pivot]=z[4]
        try:
            three,_,_=anchor_model(base,trial,cut,da,db);orb=Orbit(three)
            return np.r_[orb.match(z[:4],I),np.dot(np.array(three['trace'])[list(REGIONS)],z[:4])]
        except (ValueError,np.linalg.LinAlgError,OverflowError,FloatingPointError):return np.full(5,1e3)
    lo=np.r_[np.full(4,1e-5),seed[4]-10*scale[4]];hi=np.r_[np.full(4,120.),seed[4]+10*scale[4]]
    result=least_squares(equations,seed,bounds=(lo,hi),x_scale=scale,
                         ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=90)
    residual=equations(result.x)
    if not result.success or np.max(abs(residual))>2e-7:raise ValueError('fold equations unresolved')
    trial=dict(changes);trial[pivot]=float(result.x[4]);three,model,checks=anchor_model(base,trial,cut,da,db)
    orb=Orbit(three);_,z,q,mats=orb.match(result.x[:4],I,True);minimum=float('inf');speeds=[]
    for i,(region,A) in enumerate(zip(REGIONS,mats)):
        y=np.array([z[i],q[i],1.]);speed=float((A@y)[0]);speeds.append(speed)
        if (1 if i<2 else -1)*speed<1e-7:raise ValueError('fold non-transverse')
        samples=np.array([(expm(A*t)@y)[0] for t in np.linspace(0,result.x[i],81)])
        minimum=min(minimum,float(samples.min()))
        low=-np.inf if region==0 else orb.cuts[region-1];high=np.inf if region==2 else orb.cuts[region]
        if np.any(samples<low-1e-7) or np.any(samples>high+1e-7):raise ValueError('fold itinerary violation')
    if minimum<=cut+1e-4:raise ValueError('fold enters added lower zone')
    fold={'times':result.x[:4].tolist(),'I':I,'Cmin_sampled':minimum,
          'matching_max':float(max(abs(residual[:4]))),'logM':float(residual[4]),
          'speeds':speeds,'pivot':list(pivot),'pivot_delta':float(result.x[4])}
    return three,model,checks,fold
