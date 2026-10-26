"""AM fit with two frozen physical calcium thresholds.
MAP_KIND=mixed_power|mixed|affine python fit_am.py geometry|orbits|poincare|phase|closed|validate
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'support'))
sys.path.insert(0,str(ROOT.parent))
from loss_weights import poincare_scale,closed_components
POINCARE_SCALE=poincare_scale()
from coordinates import phi,dphi
from lr_reference import eq,eqs,jac,rhs,CMAX
from counts import ret,cycles
KIND=os.environ.get('MAP_KIND','mixed_power');P=ROOT/'fits'/f'{KIND}_fixed';P.mkdir(parents=True,exist_ok=True)
FIXED_TC=np.array([.15880902539171288,.32705145045357475])
ACTIVE=np.r_[np.arange(12),np.arange(12,16) if KIND!='affine' else []].astype(int)
LO=np.r_[np.log([.15,.03]),[-5,-6,-5],[-8,-9,-9],[-1,-1,-2,-5],[-5,-6,.15,-4]]
HI=np.r_[np.log([40,1.5]),[3,2,2],[2,1,1],[2,2,2,1],[5,3,3.,4]]
EVENTS=np.array([.35453381865657685,.6368818163953428])

def unpack(x):
    a,b=np.exp(x[:2]);T=np.array([-np.exp(x[2]),np.exp(x[3]),-np.exp(x[4])]);D=np.array([np.exp(x[5]),T[1]**2/4+np.exp(x[6]),T[2]**2/4+np.exp(x[7])])
    n=-(T+b)/a;q=n-D/(a*b);shape=x[12:16] if KIND!='affine' else None
    tc=FIXED_TC.copy();th=phi(tc,shape)
    N=np.r_[0,n[0],x[10],np.diff(n)];H=np.r_[0,q[0],x[10]+np.exp(x[11]),np.diff(q)]
    bb=np.r_[1,0,.5,np.maximum(-th,0)];N[0]=x[8]-bb@N;H[0]=x[9]-bb@H
    co=np.zeros((2,11));idx=[0,1,3,4,5];co[0,idx]=-a*N;co[0,2]=a;co[1,idx]=b*H;co[1,2]=-b
    return dict(a=a,b=b,T=T,D=D,N=N,H=H,shape=shape,tc=tc,th=th,co=co,padded=np.r_[th,[100,101,102,103,104]])

def basis(c,I,m):
    z,I=np.broadcast_arrays(phi(np.asarray(c),m['shape']),I)
    return np.stack([np.ones_like(z),z,I,*[np.maximum(z-t,0) for t in m['th']]],axis=-1)

def events(m):return basis(m['tc'],0,m)@(m['N']-m['H'])/(m['H'][2]-m['N'][2])

def inverse(z,m):
    if m['shape'] is None:return .5*(np.asarray(z)+1)
    z=np.asarray(z);lo=np.full_like(z,-700.);hi=np.full_like(z,3.)
    for _ in range(62):
        mid=(lo+hi)/2;mask=phi(.5*np.exp(mid),m['shape'])<z;lo=np.where(mask,mid,lo);hi=np.where(mask,hi,mid)
    return .5*np.exp((lo+hi)/2)

def equilibria(I,m):
    n=m['N'][1]+np.r_[0,np.cumsum(m['N'][3:])];q=m['H'][1]+np.r_[0,np.cumsum(m['H'][3:])]
    nc=m['N'][0]-np.r_[0,np.cumsum(m['N'][3:]*m['th'])];hc=m['H'][0]-np.r_[0,np.cumsum(m['H'][3:]*m['th'])]
    z=(hc-nc+(m['H'][2]-m['N'][2])*I)/(n-q);h=nc+n*z+m['N'][2]*I
    valid=(z>np.r_[-np.inf,m['th']])&(z<=np.r_[m['th'],np.inf]);out=[]
    for r in np.where(valid)[0]:
        c=float(inverse(z[r],m))
        if 0<c<CMAX and 0<h[r]<1:out.append(dict(z=[float(z[r]),float(h[r])],C=c,h=float(h[r]),region=int(r),stable=bool(m['T'][r]<0)))
    return out

def truth(c,I):
    q=1.049*(I+.13)/(I+.9434)
    return np.stack([np.cbrt((.9*c*c/(c*c+.01)-.11*(2-1.185*c))/(6*(I/(I+.13))**3*(c/(c+.08234))**3*(2-1.185*c))),q/(q+c)],axis=-1)

def data():
    from numpy.polynomial.legendre import leggauss
    c,w=leggauss(90);c=.06+(c+1)*.59/2;w=w/2
    ii,wi=leggauss(12);ii=.25+(ii+1)*.5/2;wi=wi/2
    C,I=np.meshgrid(c,ii,indexing='ij');W=np.sqrt(w[:,None]*wi[None,:])
    ie=np.linspace(.2,.8,71);ee=np.array([eq(i) for i in ie]);jj=np.array([jac(e,i) for e,i in zip(ee,ie)]);sp=np.column_stack([np.trace(jj,axis1=1,axis2=2),np.linalg.det(jj)])
    source=np.genfromtxt(ROOT/'LR_stable_cycles.csv',delimiter=',',names=True,dtype=None,encoding='utf8');targets=[]
    for i in [.36,.38,.42,.46,.5,.54,.58,.62,.634]:
        q=source[np.argmin(abs(source['I']-i))]
        targets.append(dict(I=float(q['I']),h=float(q['h0']),C=float(q['section_C']),T=float(q['T']),Cmin=float(q['Cmin']),Cmax=float(q['Cmax'])))
    return C,I,W,truth(C,I),ie,ee,sp,targets
DATA=data();TARGETS=DATA[-1]

def geometric(x):
    m=unpack(x);C,I,W,target,ie,ee,sp,_=DATA;bb=basis(C,I,m)
    geo=np.stack([bb@m['N'],bb@m['H']],axis=-1)-target
    be=basis(ee[:,0],ie,m);er=np.stack([be@m['N']-ee[:,1],be@m['H']-ee[:,1]],axis=-1)
    regs=np.searchsorted(m['tc'],ee[:,0]);sr=np.column_stack([m['T'][regs]-sp[:,0],(m['D'][regs]-sp[:,1])/.15])
    return np.r_[(geo*W[:,:,None]/.04).ravel(),(er/.025/np.sqrt(len(ie))).ravel(),.18*sr.ravel()/np.sqrt(len(ie)),3*(events(m)-EVENTS)/.025]

def initial():
    saved=ROOT/'mixed_power'/'candidate.json'
    if saved.exists():
        candidate=json.loads(saved.read_text())
        if np.allclose(candidate.get('physical_thresholds_C',[]),FIXED_TC,rtol=0,atol=1e-10):
            return np.asarray(candidate['parameters'][:16],dtype=float)
    # The fitted coordinates are [12 dynamical parameters, 4 map parameters].
    # Threshold logits from the archived moving-cut fit are deliberately absent.
    return np.r_[np.log([5.,.2]),np.log([1.,.12,.25]),np.log([.35,.24,.35]),.58,.58,-.2,np.log(.25),[0.,-3.,.5,-1.]]

def write(x,path,extra=None):
    m=unpack(x);out=dict(case='AM',K_ER=.1,coordinate_family=KIND,threshold_count_m=2,regions=3,physical_thresholds_C=m['tc'].tolist(),threshold_policy='fixed_before_fit_from_mixed_two_thresholds_seed',thresholds_optimized=False,regional_type_constraints=['stable (negative trace), positive determinant','unstable (positive trace), positive determinant','stable (negative trace), positive determinant'],I_thresholds=[],alpha=m['N'].tolist(),beta=m['H'].tolist(),a=m['a'],b=m['b'],transform=None if m['shape'] is None else dict(zip(['p','e','n','k'],m['shape'])),trace=m['T'].tolist(),determinant=m['D'].tolist(),discriminant=(m['T']**2-4*m['D']).tolist(),equilibrium_threshold_I=events(m).tolist(),parameters=x.tolist(),accepted=False)
    out['poincare_weight_scale']=POINCARE_SCALE
    if out['transform'] is not None:out['transform']['C_ref']=.5
    if extra:out.update(extra)
    (P/path).write_text(json.dumps(out,indent=2))

def solve(fun,x,nfev=600):
    base=x.copy()
    def expand(v):
        q=base.copy();q[ACTIVE]=v;return q
    o=least_squares(lambda v:fun(expand(v)),x[ACTIVE],bounds=(LO[ACTIVE],HI[ACTIVE]),max_nfev=nfev,ftol=1e-9,xtol=1e-9,gtol=1e-7,x_scale='jac')
    return expand(o.x),o

def geometry():
    rng=np.random.default_rng(261026);base=initial();best=np.inf;runs=[]
    for k in range(8 if KIND=='mixed' else 4):
        x=base.copy() if k==0 else base+rng.normal(size=16)*np.r_[[.3]*8,[.03,.03,.04,.3],[.7,.7,.3,.7]]
        x=np.clip(x,LO+1e-6,HI-1e-6);x,o=solve(geometric,x)
        cost=float(o.fun@o.fun);runs.append(dict(run=k,loss=cost,nfev=o.nfev,optimality=float(o.optimality),message=o.message))
        print('GEOMETRY',KIND,runs[-1],events(unpack(x)),unpack(x)['tc'],flush=True)
        if cost<best:best=cost;write(x,'geometry.json',dict(loss=cost,optimizer=runs[-1]))
    (P/'geometry_runs.json').write_text(json.dumps(runs,indent=2))

def orbit_samples():
    cache=ROOT/'orbit_samples.npz'
    if cache.exists():
        d=np.load(cache);return [d[k] for k in ['C','h','I','vel']]
    st=[];ivals=[]
    for q in TARGETS:
        sol=solve_ivp(lambda t,y:rhs(y,q['I']),[0,q['T']],[q['C'],q['h']],t_eval=np.linspace(0,q['T'],96,endpoint=False),method='DOP853',rtol=2e-11,atol=1e-13,max_step=.06)
        st.extend(sol.y.T);ivals.extend([q['I']]*len(sol.t))
    C,h=np.array(st).T;I=np.array(ivals);vel=np.array([rhs([c,hh],ii) for c,hh,ii in zip(C,h,I)])
    np.savez(cache,C=C,h=h,I=I,vel=vel);return C,h,I,vel

def orbits():
    x=np.array(json.loads((P/'geometry.json').read_text())['parameters']);C,h,I,vel=orbit_samples()
    def field(v):
        m=unpack(v);bb=basis(C,I,m);pred=np.column_stack([m['a']*(h-bb@m['N'])/dphi(C,m['shape']),m['b']*(bb@m['H']-h)])
        return ((pred-vel)/[.035,.006]/np.sqrt(len(C))).ravel()
    for weight in [1.,5.,20.]:
        x,o=solve(lambda v:np.r_[geometric(v),np.sqrt(weight)*field(v)],x,800)
        write(x,f'orbit_{weight:g}.json',dict(loss=float(o.fun@o.fun),velocity_weight=weight,optimizer=dict(nfev=o.nfev,optimality=float(o.optimality),message=o.message)))
        print('ORBITS',KIND,weight,'loss',float(o.fun@o.fun),'cuts',unpack(x)['tc'],'events',events(unpack(x)),flush=True)
    write(x,'candidate.json',dict(fit_scope='hard stability signs; mixed geometry/equilibria/spectrum/events + field on source periodic orbits'))

def p_residual(x,detail=False,h_values=None):
    m=unpack(x);bounds=phi(np.array([1e-12,CMAX]),m['shape']);cg=np.geomspace(1e-12,CMAX,4096);zg=phi(cg,m['shape']);rows=[];r=[]
    for j,q in enumerate(TARGETS):
        h=q['h'] if h_values is None else h_values[j]
        sec=float(phi(q['C'],m['shape']));status,v=ret(q['I'],h,sec,m['co'],m['padded'],bounds,tol=3e-8,fast=True)
        if status==1:
            ext=np.interp(v[3:5],zg,cg);rr=[v[0]/.008,(v[1]-q['T'])/2,(ext[0]-q['Cmin'])/.025,(ext[1]-q['Cmax'])/.04]
        else:
            n=float(basis(q['C'],q['I'],m)@m['N']);rr=[5+20*max(n-h,0),5.,5.,5.]
        rows.append(dict(target=q,status=int(status),return_data=v.tolist(),residual=rr));r.extend(rr)
    return rows if detail else np.array(r)

def poincare():
    seed=json.loads((P/'orbit_20.json').read_text());base=np.array(seed['parameters']);scale=np.r_[[.35]*8,[.025,.025,.07,.25],[.7,.7,.35,.7]]
    low=np.maximum(-1.5,(LO-base)/scale);high=np.minimum(1.5,(HI-base)/scale);calls=0;best=np.inf;start=time.time()
    def objective(y):
        nonlocal calls,best
        x=base.copy();x[ACTIVE]+=scale[ACTIVE]*y;g=geometric(x);r=p_residual(x);cost=float(g@g+4*POINCARE_SCALE*(r@r));calls+=1
        if cost<best:
            best=cost;write(x,'candidate.json',dict(loss=cost,fit_scope='hard stability signs + geometry/equilibria/soft spectrum/events + Poincare closure/period/extrema',poincare_weight=4.*POINCARE_SCALE,calls=calls))
        if calls%200==0:print('POINCARE',KIND,calls,'best',round(best,5),'seconds',round(time.time()-start),flush=True)
        return cost
    o=minimize(objective,np.zeros(len(ACTIVE)),method='Powell',bounds=list(zip(low[ACTIVE],high[ACTIVE])),options=dict(maxfev=6500,xtol=2e-3,ftol=3e-5))
    (P/'optimizer.json').write_text(json.dumps(dict(nfev=o.nfev,fun=float(o.fun),success=bool(o.success),message=o.message),indent=2))

def scan_job(arg):
    I,x=arg;m=unpack(x);ee=equilibria(I,m);qq,diag=cycles(float(I),ee,m['co'],m['padded'],phi(np.array([1e-12,CMAX]),m['shape']),n=80,tol=7e-10)
    for q in qq:q.update(Cmin=float(inverse(q['min'],m)),Cmax=float(inverse(q['max'],m)))
    return dict(I=float(I),equilibria=ee,cycles=qq,diagnostics=diag)

def phase_fit():
    from scipy.optimize import brentq
    seed=json.loads((P/'candidate.json').read_text());(P/'before_phase.json').write_text(json.dumps(seed,indent=2));base=np.array(seed['parameters'])
    m=unpack(base)
    # Include the uppermost training orbit in the initial oscillatory window.
    base[8]+=max(0,.6375-events(m)[1])*(m['H'][2]-m['N'][2]);m=unpack(base)
    h0=[];bounds=phi(np.array([1e-12,CMAX]),m['shape'])
    for q in TARGETS:
        sec=float(phi(q['C'],m['shape']));n=float(basis(q['C'],q['I'],m)@m['N']);grid=n+(1-n)*np.r_[np.geomspace(1e-5,.1,16),np.linspace(.1,.98,20)]
        def fun(h):
            status,v=ret(q['I'],h,sec,m['co'],m['padded'],bounds,tol=1e-8,fast=True)
            return v[0] if status==1 else np.nan
        vals=[fun(h) for h in grid];roots=[]
        for a,b,va,vb in zip(grid[:-1],grid[1:],vals[:-1],vals[1:]):
            if np.isfinite(va) and np.isfinite(vb) and va*vb<0:
                try:roots.append(brentq(fun,a,b,xtol=1e-9))
                except ValueError:pass
        h0.append(max(roots) if roots else q['h'])
    h0=np.array(h0);scale=np.r_[[.35]*8,[.025,.025,.07,.25],[.8,.8,.4,.8]]
    low=np.maximum(-1.5,(LO-base)/scale);high=np.minimum(1.5,(HI-base)/scale)
    low=np.r_[low[ACTIVE],np.maximum(-1.5,(.01-h0)/.025)];high=np.r_[high[ACTIVE],np.minimum(1.5,(.99-h0)/.025)]
    calls=0;best=np.inf;start=time.time();nn=len(ACTIVE)
    def objective(y):
        nonlocal calls,best
        x=base.copy();x[ACTIVE]+=scale[ACTIVE]*y[:nn];h=h0+.025*y[nn:];g=geometric(x);r=p_residual(x,h_values=h).reshape(-1,4)*[2,1,1.25,1.6]
        cost=float(g@g+4*POINCARE_SCALE*np.sum(r*r));calls+=1
        if cost<best:
            best=cost;write(x,'candidate.json',dict(loss=cost,fit_scope='hard stability + geometry/equilibria/events + Poincare closure/period/extrema with free section heights',poincare_weight=4.*POINCARE_SCALE,section_heights=h.tolist(),initial_section_heights=h0.tolist(),calls=calls))
        if calls%300==0:print('PHASE',KIND,calls,'best',round(best,5),'seconds',round(time.time()-start),flush=True)
        return cost
    o=minimize(objective,np.zeros(nn+len(TARGETS)),method='Powell',bounds=list(zip(low,high)),options=dict(maxfev=9500,xtol=1e-3,ftol=2e-5))
    (P/'phase_optimizer.json').write_text(json.dumps(dict(nfev=o.nfev,fun=float(o.fun),success=bool(o.success),message=o.message),indent=2))

def closed_fit():
    from scipy.optimize import brentq
    sys.path.insert(0,str(ROOT.parent))
    from equilibrium_loss import model_cost
    eq_source=[(float(I),eqs(float(I))) for I in np.linspace(.2,.8,25)]
    seed=json.loads((P/'candidate.json').read_text());(P/'before_closed.json').write_text(json.dumps(seed,indent=2));base=np.array(seed['parameters'])
    phase=np.array(seed.get('section_heights',[q['h'] for q in TARGETS]));scale=np.r_[[.18]*8,[.012,.012,.04,.15],[.45,.45,.2,.45]]
    low=np.maximum(-1.5,(LO-base)/scale);high=np.minimum(1.5,(HI-base)/scale);calls=0;best=np.inf;start=time.time()
    def actual(x):
        m=unpack(x);bounds=phi(np.array([1e-12,CMAX]),m['shape']);cg=np.geomspace(1e-12,CMAX,4096);zg=phi(cg,m['shape']);res=[];heights=[];details=[]
        for j,q in enumerate(TARGETS):
            sec=float(phi(q['C'],m['shape']));n=float(basis(q['C'],q['I'],m)@m['N']);h=float(np.clip(phase[j],n+1e-6,.999));ok=False;v=np.zeros(7)
            cache={}
            def calc(h):
                if h not in cache:cache[h]=ret(q['I'],h,sec,m['co'],m['padded'],bounds,tol=3e-9,fast=False)
                return cache[h]
            for it in range(14):
                st,v=calc(h)
                if st!=1:break
                if abs(v[0])<8e-9 and v[2]<-1e-6:ok=True;break
                der=np.expm1(np.clip(v[2],-700,50))
                if abs(der)<1e-7:break
                step=float(np.clip(-v[0]/der,-.025,.025));hn=float(np.clip(h+step,n+1e-6,.999))
                if abs(hn-h)<1e-10:break
                h=hn
            if not ok:
                grid=n+(1-n)*np.r_[np.geomspace(1e-5,.12,12),np.linspace(.12,.95,12)];vals=[]
                for s in grid:
                    st,vv=calc(float(s));vals.append(vv[0] if st==1 else np.nan)
                for a,b,va,vb in zip(grid[:-1],grid[1:],vals[:-1],vals[1:]):
                    if np.isfinite(va) and np.isfinite(vb) and va*vb<0:
                        def fun(s):
                            st,vv=calc(float(s));return vv[0] if st==1 else np.nan
                        try:h=float(brentq(fun,a,b,xtol=2e-9));st,v=calc(h);ok=st==1 and abs(v[0])<3e-8 and v[2]<-1e-6
                        except ValueError:continue
                        if ok:break
            if ok:
                ext=np.interp(v[3:5],zg,cg);amp=ext[1]-ext[0];targetamp=q['Cmax']-q['Cmin']
                rr=[(v[1]-q['T'])/2,(ext[0]-q['Cmin'])/.025,(ext[1]-q['Cmax'])/.035,(amp-targetamp)/.02]
            else:rr=[8.,8.,8.,8.]
            res.extend(rr);heights.append(h);details.append(dict(I=q['I'],found=bool(ok),h=h,return_data=v.tolist(),residual=rr))
        return np.array(res),heights,details
    def objective(y):
        nonlocal calls,best
        x=base.copy();x[ACTIVE]+=scale[ACTIVE]*y;g=geometric(x);r,heights,details=actual(x);eq_loss=model_cost(unpack(x),eq_source,equilibria);parts=closed_components(g@g,eq_loss,r@r,2.,POINCARE_SCALE);cost=parts['total'];calls+=1
        if cost<best:
            best=cost;write(x,'candidate.json',dict(loss=cost,actual_equilibrium_set_loss=eq_loss,fit_scope='hard stability + geometry/equilibria/events + actual equilibrium set matching + actual stable Poincare roots, periods, extrema and amplitude',cycle_weight=2.*POINCARE_SCALE,loss_components=parts,section_heights=heights,closed_orbit_training=details,calls=calls))
        if calls==1 or calls%150==0:print('CLOSED',KIND,calls,'best',round(best,5),'components',parts,'seconds',round(time.time()-start),flush=True)
        return cost
    o=minimize(objective,np.zeros(len(ACTIVE)),method='Powell',bounds=list(zip(low[ACTIVE],high[ACTIVE])),options=dict(maxfev=4500,xtol=2e-3,ftol=3e-5))
    (P/'closed_optimizer.json').write_text(json.dumps(dict(nfev=o.nfev,fun=float(o.fun),success=bool(o.success),message=o.message),indent=2))

def validate():
    from concurrent.futures import ProcessPoolExecutor
    name=sys.argv[2] if len(sys.argv)>2 else 'candidate';x=np.array(json.loads((P/(name+'.json')).read_text())['parameters']);m=unpack(x)
    grid=np.unique(np.round(np.r_[np.linspace(0,1,101),np.arange(.345,.661,.0025),[q['I'] for q in TARGETS],[e+d for e in events(m) for d in [-.003,-.001,-.0001,-1e-5,1e-5,.0001,.001,.003]]],11))
    rows=[]
    with ProcessPoolExecutor(int(os.environ.get('WORKERS','4'))) as pool:
        for k,q in enumerate(pool.map(scan_job,[(float(I),x) for I in grid])):
            rows.append(q)
            if k%25==0:print('SCAN',KIND,k,len(grid),q['I'],len(q['cycles']),flush=True)
    (P/(name+'_branches.json')).write_text(json.dumps(dict(rows=rows,events=events(m).tolist(),method='80 seeds at all physical equilibrium sections; forward and reverse return roots; finite scan'),indent=2))

if __name__=='__main__':{'geometry':geometry,'orbits':orbits,'poincare':poincare,'validate':validate,'phase':phase_fit,'closed':closed_fit}[sys.argv[1]]()
