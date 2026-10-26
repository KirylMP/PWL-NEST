"""Four-region FM experiment: fixed physical cuts, hard types, then return-map fit.
Run with OPENBLAS_NUM_THREADS=1 python3 fit_four.py [geometry|poincare|validate].
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'support'))
sys.path.insert(0,str(P.parent))
from loss_weights import poincare_scale
POINCARE_SCALE=poincare_scale()
from fit import phi,dphi
from geometry_reference import truth,gauss
from lr_reference import eqs,jac,gate,input_at_C,CMAX
from counts import ret,cycles
TC=np.array([.0624450437666669,.13693821763740716,.3623042888621253])
OLD=json.loads((P/'previous_selected.json').read_text())['candidate']
SH=np.array([OLD['transform'][k] for k in ['p','e','n','k']])
# x: log(a,b), four T parameters, four D parameters, N/H at C_ref,I=.8,
# globally affine I coefficients, map(p,e,n,k). Types hold by construction.
LO=np.r_[np.log([.2,.03]),[-5,-6,-5,-5],[-8,-9,-9,-9],[-2,-2,-2,-2],[-4,-6,.15,-4]]
HI=np.r_[np.log([40,1.2]),[3,6,2,2],[8,1,1,1],[3,3,2,2],[5,3,2.5,4]]

def unpack(x):
    a,b=np.exp(x[:2]);T=np.array([-np.exp(x[2]),x[3],np.exp(x[4]),-np.exp(x[5])])
    D=np.array([T[0]**2/4*expit(x[6]),-np.exp(x[7]),T[2]**2/4+np.exp(x[8]),T[3]**2/4+np.exp(x[9])])
    n=-(T+b)/a;q=n-D/(a*b);shape=x[14:];th=phi(TC,shape)
    N=np.r_[0,n[0],x[12],np.diff(n)];H=np.r_[0,q[0],x[13],np.diff(q)]
    at0=np.r_[1,0,.8,np.maximum(-th,0)]
    N[0]=x[10]-at0@N;H[0]=x[11]-at0@H
    co=np.zeros((2,11));idx=[0,1,3,4,5,6]
    co[0,idx]=-a*N;co[0,2]=a;co[1,idx]=b*H;co[1,2]=-b
    padded=np.r_[th,[100,101,102,103]]
    return dict(a=a,b=b,T=T,D=D,N=N,H=H,shape=shape,th=th,co=co,padded=padded)

def basis(c,I,m):
    z=phi(np.asarray(c),m['shape']);z,I=np.broadcast_arrays(z,I)
    return np.stack([np.ones_like(z),z,I,*[np.maximum(z-t,0) for t in m['th']]],axis=-1)

def events(m):
    bb=basis(TC,0,m);return (bb@(m['N']-m['H']))/(m['H'][2]-m['N'][2])

def inverse(z,m):
    z=np.asarray(z);lo=np.full_like(z,-700.);hi=np.full_like(z,3.)
    for _ in range(60):
        mid=(lo+hi)/2;mask=phi(.5*np.exp(mid),m['shape'])<z
        lo=np.where(mask,mid,lo);hi=np.where(mask,hi,mid)
    return .5*np.exp((lo+hi)/2)

def equilibrium(I,m):
    n=m['N'][1]+np.r_[0,np.cumsum(m['N'][3:])];q=m['H'][1]+np.r_[0,np.cumsum(m['H'][3:])]
    nc=m['N'][0]-np.r_[0,np.cumsum(m['N'][3:]*m['th'])];hc=m['H'][0]-np.r_[0,np.cumsum(m['H'][3:]*m['th'])]
    z=(hc-nc+(m['H'][2]-m['N'][2])*I)/(n-q);h=nc+n*z+m['N'][2]*I
    valid=(z>np.r_[-np.inf,m['th']])&(z<=np.r_[m['th'],np.inf])
    return [dict(z=[float(z[r]),float(h[r])],C=float(inverse(z[r],m)),region=int(r),stable=bool(r in [0,3])) for r in np.where(valid & (h>0)&(h<1))[0] if 0<inverse(z[r],m)<CMAX]

def initialize():
    x=np.r_[np.log([OLD['a'],OLD['b']]),np.log(1.5),3.,np.log(.68),np.log(.72),0.,np.log(.096),np.log(.19),np.log(.294),.6,.6,OLD['alpha'][2],OLD['beta'][2],SH]
    omth=phi(np.array(OLD['physical_thresholds_C']),SH);bb=np.r_[1,0,.8,np.maximum(-omth,0)]
    x[10:12]=[bb@OLD['alpha'],bb@OLD['beta']]
    return x

def data():
    cp=[gauss(18,a,b) for a,b in zip([.03,*TC],[*TC,1.15])];c=np.concatenate([q[0] for q in cp]);wc=np.concatenate([q[1] for q in cp])/1.12
    ii,wi=gauss(8,.4,1.2);C,I=np.meshgrid(c,ii,indexing='ij');w=np.sqrt(wc[:,None]*wi[None,:]/.8)
    ce=np.r_[np.linspace(.033,.061,15),np.linspace(.064,.134,12),np.linspace(.14,.36,22),np.linspace(.365,.75,22)]
    ie=np.array([input_at_C(c) for c in ce]);he=gate(ce,ie)
    spec=np.array([[np.trace(jac([c,h],i)),np.linalg.det(jac([c,h],i))] for c,h,i in zip(ce,he,ie)])
    return C,I,w,truth(C,I),ce,ie,he,spec
DATA=data()

def geometric(x):
    m=unpack(x);C,I,w,y,ce,ie,he,sp=DATA;bb=basis(C,I,m)
    geo=np.stack([bb@m['N'],bb@m['H']],axis=-1)-y
    be=basis(ce,ie,m);er=np.stack([be@m['N']-he,be@m['H']-he],axis=-1)
    regs=np.searchsorted(TC,ce);sr=np.column_stack([(m['T'][regs]-sp[:,0]),(m['D'][regs]-sp[:,1])/.15])
    # Fold/BEB locations: lower node-saddle fold, saddle-focus fold, upper focus-focus.
    ev=(events(m)-[.52907,.482304,.860021])/.04
    return np.r_[(geo*w[:,:,None]/.025).ravel(),(er/.04/np.sqrt(len(ce))).ravel(),.35*sr.ravel()/np.sqrt(len(ce)),2*ev]

def write(x,path,extra=None):
    m=unpack(x);out=dict(schema_version=1,regions=4,threshold_count_m=3,physical_thresholds_C=TC.tolist(),alpha=m['N'].tolist(),beta=m['H'].tolist(),a=m['a'],b=m['b'],transform=dict(zip(['p','e','n','k'],m['shape'])),regional_types=['stable_node','saddle','unstable_focus','stable_focus'],trace=m['T'].tolist(),determinant=m['D'].tolist(),discriminant=(m['T']**2-4*m['D']).tolist(),equilibrium_threshold_I=events(m).tolist(),parameters=x.tolist(),accepted=False,notes=['Fixed physical cuts; hard regional types by parameterization','Numerical experiment; no proof of minimality or cycle completeness'])
    out['poincare_weight_scale']=POINCARE_SCALE
    out['training_cycle_inputs']=sorted({q['I'] for q in TARGETS})
    out['training_cycle_count']=len(TARGETS)
    out['training_cycle_residual_scale']=float(TARGET_NORMALIZATION)
    out['transform']['C_ref']=.5
    if extra:out.update(extra)
    (P/path).write_text(json.dumps(out,indent=2))

def geometry():
    rng=np.random.default_rng(20261026);base=initialize();runs=[];best=None
    for j in range(10):
        x=base.copy() if j==0 else base+rng.normal(size=18)*np.r_[.25,.2,.5,.6,.4,.4,1.,.5,.5,.5,.04,.04,.03,.03,.3,.3,.1,.3]
        x=np.clip(x,LO+1e-5,HI-1e-5)
        o=least_squares(geometric,x,bounds=(LO,HI),max_nfev=1300,ftol=1e-9,xtol=1e-9,gtol=1e-7,x_scale='jac')
        v=float(o.fun@o.fun);runs.append(dict(run=j,loss=v,nfev=o.nfev,optimality=o.optimality,message=o.message))
        print('GEOMETRY',runs[-1],events(unpack(o.x)),flush=True)
        if best is None or v<best[0]:best=v,o.x;write(o.x,'geometry.json',dict(loss=v,optimizer=runs[-1]))
    (P/'geometry_runs.json').write_text(json.dumps(runs,indent=2))

def prepare_targets():
    source=json.loads((P/'LR_cycles_reference.json').read_text())['rows'];targets=[]
    rows=[r for r in source if r['cycles']]
    explicit=os.environ.get('FM_TARGET_I')
    if explicit:
        desired=[float(v.strip()) for v in explicit.split(',') if v.strip()]
        if not desired or not all(np.isfinite(v) for v in desired):raise ValueError('FM_TARGET_I must contain finite I values')
        for I in desired:
            if min(abs(r['I']-I) for r in rows)>1e-8:raise ValueError(f'FM_TARGET_I={I} is not an LR reference point with cycles')
    else:
        count=int(os.environ.get('FM_TARGET_GRID_SIZE','25'))
        if count<2:raise ValueError('FM_TARGET_GRID_SIZE must be at least 2')
        desired=np.r_[np.linspace(.53,1.12,count),[.55,.65,.8,.9,1.,1.1]]
    selected={min(rows,key=lambda r:abs(r['I']-I))['I'] for I in desired}
    for I in sorted(selected):
        row=min(rows,key=lambda r:abs(r['I']-I))
        for q in row['cycles']:
            targets.append(dict(I=I,h=q['s'],C=q['section'],T=q['T'],Cmin=q['min'],Cmax=q['max'],stable=q['stable'],log_multiplier=q['log_multiplier']))
    return targets

TARGETS=prepare_targets()
TARGET_NORMALIZATION=np.sqrt(9./len(TARGETS))

def p_residual(x,detail=False):
    m=unpack(x);bounds=phi(np.array([1e-12,CMAX]),m['shape']);res=[];rows=[]
    cg=np.geomspace(1e-12,CMAX,4096);zg=phi(cg,m['shape'])
    for q in TARGETS:
        I=q['I'];z=float(phi(q['C'],m['shape']));h=q['h'];back=not q['stable']
        status,v=ret(I,h,z,m['co'],m['padded'],bounds,back=back,tol=3e-8,fast=True)
        if status==1:
            # Root constraint and time of return; stable and unstable flows use reliable directions.
            rr=[v[0]/.015,(v[1]-q['T'])/8]
            if os.environ.get('FIT_EXTREMA')=='1':
                ext=np.interp(v[3:5],zg,cg)
                rr.extend([(ext[0]-q['Cmin'])/.025,(ext[1]-q['Cmax'])/.15])
        else:
            # Smooth directional guidance until a physical return exists.
            n=float(basis(q['C'],I,m)@m['N']);rr=[5+20*max(n-h,0),5.]
            if os.environ.get('FIT_EXTREMA')=='1':rr.extend([5.,5.])
        res.extend(rr);rows.append(dict(target=q,status=int(status),return_data=v.tolist(),residual=rr))
    return rows if detail else TARGET_NORMALIZATION*np.array(res)

def poincare():
    seed='candidate.json' if os.environ.get('REFINE')=='1' else 'geometry.json'
    x=np.array(json.loads((P/seed).read_text())['parameters']);best=None
    for base_weight in [1.,4.,12.]:
        weight=base_weight*POINCARE_SCALE
        calls=0;start=time.time()
        def fun(v):
            nonlocal calls,best
            calls+=1;g=geometric(v);r=p_residual(v);out=np.r_[g,np.sqrt(weight)*r]
            cost=float(out@out)
            if calls%100==0:print('RETURN',weight,calls,'loss',round(cost,5),'geom',round(float(g@g),5),'p',round(float(r@r),5),'sec',round(time.time()-start),flush=True)
            return out
        o=least_squares(fun,x,bounds=(LO,HI),max_nfev=160 if os.environ.get('REFINE')=='1' else 100,diff_step=3e-4,ftol=None if os.environ.get('REFINE')=='1' else 2e-5,xtol=None if os.environ.get('REFINE')=='1' else 2e-5,gtol=1e-5,x_scale='jac')
        x=o.x;write(x,f'poincare_{base_weight:g}.json',dict(loss=float(o.fun@o.fun),poincare_weight=weight,optimizer=dict(nfev=o.nfev,optimality=o.optimality,message=o.message),return_targets=p_residual(x,True)))
        print('STAGE DONE',weight,o.nfev,o.fun@o.fun,flush=True)
    write(x,'candidate.json',dict(return_targets=p_residual(x,True),fit_scope='hard types + nullclines + source-equilibrium anchors + soft spectrum + threshold events + Poincare orbit closure and periods'))

def scan_job(args):
    I,x=args;m=unpack(x);ee=equilibrium(I,m);qq,diag=cycles(I,ee,m['co'],m['padded'],phi(np.array([1e-12,CMAX]),m['shape']),n=96 if os.environ.get('VALIDATE_DENSE')=='1' else 64,tol=5e-10 if os.environ.get('VALIDATE_DENSE')=='1' else 1e-9)
    for q in qq:q.update(Cmin=float(inverse(q['min'],m)),Cmax=float(inverse(q['max'],m)))
    return dict(I=float(I),equilibria=ee,cycles=qq,diagnostics=diag)

def orbit_fit():
    from scipy.integrate import solve_ivp
    from lr_reference import rhs
    cache=P/'orbit_samples.npz'
    if cache.exists():
        dd=np.load(cache);C,I,h,vel=[dd[k] for k in ['C','I','h','vel']]
    else:
        states=[];ivals=[]
        for q in TARGETS:
            direction=1 if q['stable'] else -1
            sol=solve_ivp(lambda t,y:direction*rhs(y,q['I']),[0,q['T']],[q['C'],q['h']],t_eval=np.linspace(0,q['T'],96,endpoint=False),method='DOP853',rtol=2e-11,atol=1e-13,max_step=.06)
            states.extend(sol.y.T);ivals.extend([q['I']]*len(sol.t))
        states=np.array(states);C,h=states.T;I=np.array(ivals);vel=np.array([rhs([c,hh],ii) for c,hh,ii in zip(C,h,I)])
        np.savez(cache,C=C,I=I,h=h,vel=vel)
    # Continue from the just-refined return-map candidate when available. The
    # archived closure-only model remains a fallback for standalone orbit runs.
    seed_path=P/'poincare_12.json'
    if not seed_path.exists():seed_path=P/'closure_only.json'
    seed=json.loads(seed_path.read_text());x=np.array(seed['parameters'])
    def field_res(v):
        m=unpack(v);bb=basis(C,I,m)
        pred=np.column_stack([m['a']*(h-bb@m['N'])/dphi(C,m['shape']),m['b']*(bb@m['H']-h)])
        return ((pred-vel)/[.12,.012]/np.sqrt(len(C))).ravel()
    for w in [1.,5.,20.,60.]:
        o=least_squares(lambda v:np.r_[geometric(v),np.sqrt(w)*field_res(v)],x,bounds=(LO,HI),max_nfev=900,ftol=1e-9,xtol=1e-9,gtol=1e-7,x_scale='jac')
        x=o.x;write(x,f'orbit_{w:g}.json',dict(loss=float(o.fun@o.fun),orbit_velocity_weight=w,optimizer=dict(nfev=o.nfev,optimality=o.optimality,message=o.message)))
        print('ORBIT',w,o.nfev,'loss',float(o.fun@o.fun),'events',events(unpack(x)),flush=True)
    write(x,'candidate.json',dict(fit_scope='hard types + geometry + equilibrium anchors + soft spectrum + threshold events + physical vector-field error on LR periodic orbits',orbit_fit_seed=seed_path.name,orbit_velocity_weight=60.,optimizer=dict(nfev=o.nfev,optimality=o.optimality,message=o.message)))

def powell_fit():
    from scipy.optimize import minimize
    os.environ['FIT_EXTREMA']='1'
    seed=json.loads((P/'orbit_60.json').read_text());base=np.array(seed['parameters'])
    scale=np.array([.4,.3,.4,.5,.4,.4,1.2,.7,1.2,.7,.04,.04,.06,.06,.6,.6,.3,.5])
    bounds=list(zip(np.maximum(-1.5,(LO-base)/scale),np.minimum(1.5,(HI-base)/scale)))
    calls=0;best=np.inf;start=time.time()
    def objective(y):
        nonlocal calls,best
        x=base+scale*y;g=geometric(x);p=p_residual(x)
        cost=float(g@g+6*POINCARE_SCALE*(p@p));calls+=1
        if cost<best:
            best=cost;write(x,'powell_best.json',dict(loss=cost,poincare_weight=6.*POINCARE_SCALE,fit_scope='hard regional types + geometry/equilibria/spectrum/events + Poincare closure, period, extrema; derivative-free local refinement',calls=calls))
        if calls%200==0:print('POWELL',calls,'best',round(best,5),'sec',round(time.time()-start),flush=True)
        return cost
    o=minimize(objective,np.zeros(18),method='Powell',bounds=bounds,options=dict(maxfev=6500,xtol=2e-3,ftol=3e-5))
    (P/'powell_optimizer.json').write_text(json.dumps(dict(nfev=o.nfev,fun=float(o.fun),success=bool(o.success),message=o.message),indent=2))
    bestmodel=json.loads((P/'powell_best.json').read_text());(P/'candidate.json').write_text(json.dumps(bestmodel,indent=2))

def validate():
    from concurrent.futures import ProcessPoolExecutor
    name=sys.argv[2] if len(sys.argv)>2 else 'candidate';x=np.array(json.loads((P/(name+'.json')).read_text())['parameters']);m=unpack(x)
    custom_size=os.environ.get('FM_VALIDATE_GRID_SIZE')
    if custom_size:
        n=int(custom_size)
        if n<2:raise ValueError('FM_VALIDATE_GRID_SIZE must be at least 2')
        grid=np.linspace(0,2,n)
    else:
        grid=np.unique(np.r_[np.linspace(0,2,81),np.arange(.47,.571,.0025),np.arange(.8,.921,.005),np.arange(1.05,1.151,.005),[e+d for e in events(m) for d in [-.001,-.0001,.0001,.001]]])
        if os.environ.get('VALIDATE_DENSE')=='1':grid=np.unique(np.round(np.r_[grid,np.arange(.45,1.25001,.0025),[e+d for e in events(m) for d in [-1e-5,1e-5]]],11))
    rows=[]
    with ProcessPoolExecutor(int(os.environ.get('WORKERS','4'))) as pool:
        for j,r in enumerate(pool.map(scan_job,[(float(I),x) for I in grid])):
            rows.append(r)
            if j%20==0:print('SCAN',name,j,len(grid),r['I'],len(r['cycles']),flush=True)
    (P/(name+'_branches.json')).write_text(json.dumps(dict(rows=rows,events=events(m).tolist(),method=('96' if os.environ.get('VALIDATE_DENSE')=='1' else '64')+' seeds per equilibrium section, forward stable and reverse unstable; finite scan'),indent=2))

if __name__=='__main__':
    print(f'FM training: {len(TARGETS)} cycles at {len(set(q["I"] for q in TARGETS))} I values; residual scale={TARGET_NORMALIZATION:.6g}',flush=True)
    if sys.argv[1]=='closed':
        from closed_fit import run
        run()
    else:
        {'geometry':geometry,'poincare':poincare,'validate':validate,'orbit':orbit_fit,'powell':powell_fit}[sys.argv[1]]()
