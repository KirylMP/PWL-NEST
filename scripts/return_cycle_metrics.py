"""Find candidate stable cycles by piecewise-affine return equations.

Finite starts and sampled route checks; unresolved does not prove nonexistence.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares, brentq
from physical_fold_routes import Route


def route_for(model,name):
    r=Route(model,'upper' if name=='upper' else 'both')
    if name=='lower':r.regions=(1,0);r.boundaries=(0,0,0)
    return r


def inspect(route,t,I,require_positive=True):
    res,z,qs,mats=route.match(t,I,True);C=[];speeds=[];extrema=[];segments=[];elapsed=0.
    if max(abs(res))>1e-7:raise ValueError('matching unresolved')
    for j,(region,A) in enumerate(zip(route.regions,mats)):
        y=np.r_[z[j],qs[j],1.];grid=np.linspace(0,t[j],81);segments.append((elapsed,elapsed+t[j],A,y));elapsed+=t[j]
        states=np.array([expm(A*s)@y for s in grid]);vel=np.array([(A@state)[0] for state in states]);speeds.append(float(vel[0]))
        direction=1 if j<len(t)//2 else -1
        if direction*vel[0]<1e-8:raise ValueError('wrong start direction or grazing')
        lo=-np.inf if region==0 else route.cut[region-1];hi=np.inf if region==2 else route.cut[region]
        if np.min(states[:,0])<lo-1e-7 or np.max(states[:,0])>hi+1e-7:raise ValueError('route violation')
        vals=list(states[:,0]);extrema.extend([(float(states[0,0]),segments[-1][0]),(float(states[-1,0]),segments[-1][1])])
        for k in range(len(grid)-1):
            if vel[k]*vel[k+1]<0:
                root=brentq(lambda t0:(A@expm(A*t0)@y)[0],grid[k],grid[k+1]);value=float((expm(A*root)@y)[0]);vals.append(value);extrema.append((value,segments[-1][0]+root))
        C.extend(vals)
    logm=float(route.T[list(route.regions)]@t)
    if max(C)-min(C)<1e-5:raise ValueError('trivial')
    if require_positive and min(C)<=0:raise ValueError('nonphysical')
    if logm>=-1e-7:raise ValueError('not a stable cycle')
    period=float(sum(t)); peak_time=max(extrema,key=lambda item:item[0])[1];shape=[]
    for phase in np.arange(64)/64:
        tt=(peak_time+phase*period)%period
        j=min(np.searchsorted([seg[1] for seg in segments],tt,side='right'),len(segments)-1)
        start,end,A,y=segments[j];shape.append(float((expm(A*(tt-start))@y)[0]))
    return {'I':float(I),'Cmin':float(min(C)),'Cmax':float(max(C)),'period':float(sum(t)),
            'shape':shape,'times':t.tolist(),'physical_sampled':bool(min(C)>0),'logM':logm,'matching':float(max(abs(res))),'crossing_speeds':speeds,
            'scope':'finite starts and 81 samples per segment; first-exit not rigorously certified'}


def solve_cycle(model,I,fold_times,extra_seeds=None,max_nfev=100,route_names=('both','upper','lower'),require_positive=True):
    tf=np.asarray(fold_times,float)
    default={'both':tf,'upper':np.array([tf[1],tf[0]+tf[2]]),'lower':np.array([max(.1,tf[0]+tf[2]),tf[3]])}
    attempts=[];found=[]
    for name in route_names:
        route=route_for(model,name);n=len(route.regions)
        seeds=([] if extra_seeds is None else extra_seeds.get(name,[]))+[default[name]]
        for seed in seeds:
            def fun(t):
                try:
                    v=route.match(t,I)
                    return v if np.all(np.isfinite(v)) else np.full(n,1e3)
                except (ValueError,OverflowError,FloatingPointError):return np.full(n,1e3)
            try:
                sol=least_squares(fun,np.clip(seed,1e-4,119.),bounds=(np.full(n,1e-5),np.full(n,120.)),
                    max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale='jac')
                item=inspect(route,sol.x,I,require_positive=require_positive);item['route']=name;found.append(item)
                attempts.append({'route':name,'status':'stable_cycle','nfev':int(sol.nfev)})
                break
            except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as e:
                attempts.append({'route':name,'status':'unresolved_or_rejected','reason':str(e)})
    return {'candidates':found,'attempts':attempts}
