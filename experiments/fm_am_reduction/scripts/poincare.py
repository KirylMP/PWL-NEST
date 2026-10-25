"""Oriented first returns, event-corrected sensitivities, and cycle counters."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import ndtr
from fit_stage import phi, hinge_design
from prepare_reference_data import lr_field, jacobian


@dataclass
class Section:
    C: float
    h0: float
    orientation: int = 1


class LRSystem:
    nparams = 0
    def __init__(self, p, ip3):
        self.p=p; self.ip3=ip3
        self.bounds=np.array([[1e-8,p['C_0']/(1+p['c_1'])-1e-8],[1e-8,1-1e-8]])
    def coordinate(self,c):return float(c)
    def coordinate_gradient(self,c):return np.empty(0)
    def physical(self,z):return float(z)
    def field(self,y):return np.asarray(lr_field(y[0],y[1],self.ip3,self.p))
    def jac(self,y):return jacobian(y[0],y[1],self.ip3,self.p)
    def partials(self,y):return np.empty((2,0))
    def nullcline(self,c):
        p=self.p;i=self.ip3
        rad=(p['v_ER']*c*c/((c*c+p['K_ER']**2)*(p['C_0']-(1+p['c_1'])*c))-p['r_L'])/(p['r_C']*(i/(i+p['d_1']))**3*(c/(c+p['d_5']))**3)
        return float(np.cbrt(rad)) if i>0 else None


class PWASystem:
    def __init__(self, theta, thresholds, ip3, c_ref, cmax):
        self.theta=np.asarray(theta);self.thresholds=np.asarray(thresholds)
        self.ncoef=len(thresholds)+3;self.nparams=len(theta);self.ip3=ip3;self.c_ref=c_ref
        self.alpha=self.theta[:self.ncoef];self.beta=self.theta[self.ncoef:2*self.ncoef]
        self.a,self.b=np.exp(self.theta[2*self.ncoef:2*self.ncoef+2]);self.eta=self.theta[-4:]
        self.zt=phi(self.thresholds,self.eta,c_ref)
        self.dzt=self._coordinate_eta(self.thresholds)
        self.ns=self.alpha[1]+np.r_[0.,np.cumsum(self.alpha[3:])]
        self.hs=self.beta[1]+np.r_[0.,np.cumsum(self.beta[3:])]
        self.ni=self.alpha[0]+self.alpha[2]*ip3-np.r_[0.,np.cumsum(self.alpha[3:]*self.zt)]
        self.hi=self.beta[0]+self.beta[2]*ip3-np.r_[0.,np.cumsum(self.beta[3:]*self.zt)]
        self.cmin=1e-8;self.cmax=cmax-1e-8
        self.bounds=np.array([[self.coordinate(self.cmin),self.coordinate(self.cmax)],[1e-8,1-1e-8]])
    def _coordinate_eta(self,c):
        # Geometry-only derivatives; no trajectory differencing is used here.
        result=[]
        for j in range(4):
            step=2e-6*max(1.,abs(self.eta[j]));d=np.eye(4)[j]*step
            result.append((phi(np.atleast_1d(c),self.eta+d,self.c_ref)-phi(np.atleast_1d(c),self.eta-d,self.c_ref))/(2*step))
        return np.array(result).T
    def coordinate(self,c):return float(phi(np.array([c]),self.eta,self.c_ref)[0])
    def coordinate_gradient(self,c):
        out=np.zeros(self.nparams);out[-4:]=self._coordinate_eta(c)[0];return out
    def physical(self,z):
        if z<=self.bounds[0,0]:return self.cmin
        if z>=self.bounds[0,1]:return self.cmax
        return float(brentq(lambda c:self.coordinate(c)-z,self.cmin,self.cmax,xtol=1e-12))
    def field(self,y):
        z,h=y;r=np.searchsorted(self.zt,z,side='right')
        return np.array([self.a*(h-self.ni[r]-self.ns[r]*z),self.b*(self.hi[r]+self.hs[r]*z-h)])
    def jac(self,y):
        r=np.searchsorted(self.zt,y[0],side='right')
        return np.array([[-self.a*self.ns[r],self.a],[self.b*self.hs[r],-self.b]])
    def partials(self,y):
        z=y[0];base=np.r_[1.,z,self.ip3,np.maximum(z-self.zt,0)]
        out=np.zeros((2,self.nparams));n=self.ncoef
        out[0,:n]=-self.a*base;out[1,n:2*n]=self.b*base
        f=self.field(y);out[0,2*n]=f[0];out[1,2*n+1]=f[1]
        active=z>self.zt
        out[0,-4:]=self.a*(self.alpha[3:]*active)@self.dzt
        out[1,-4:]=-self.b*(self.beta[3:]*active)@self.dzt
        return out
    def nullcline(self,c):
        return float(hinge_design(np.array([self.coordinate(c)]),np.array([self.ip3]),self.zt)[0]@self.alpha)


def first_return(system, section, u, sensitivities=False, *, max_time=400.,rtol=2e-8,
                 atol=2e-10,max_step=.5,max_evaluations=100000, capture_points=(),method="DOP853"):
    """First opposite crossing then first crossing in the initial orientation.

    The sensitivity matrix is propagated at fixed time across continuous-field
    switches; the moving-section correction is applied at the final return.
    """
    h=section.h0+(1-section.h0)*float(u);zsec=system.coordinate(section.C)
    y0=np.array([zsec,h]);n=system.nparams if sensitivities else 0
    orientation=section.orientation
    if not 0<u or section.h0>=1 or np.any(y0<=system.bounds[:,0]) or np.any(y0>=system.bounds[:,1]):
        return dict(status='initial_outside_domain',u=float(u))
    initial_field=system.field(y0);normal=orientation*initial_field[0]
    if normal<=1e-9:
        return dict(status='wrong_orientation_or_tangency',u=float(u),initial_normal=float(normal))
    # Last column is the derivative with respect to the section coordinate h.
    S=np.zeros((2,n+1));S[1,-1]=1.
    dzsec=system.coordinate_gradient(section.C) if n else np.empty(0)
    if n:S[0,:n]=dzsec
    initial=np.r_[y0,S.ravel(),0.];evaluations=0;paths=[];times=[]
    def rhs(t,v):
        nonlocal evaluations
        evaluations+=1
        if evaluations>max_evaluations:raise RuntimeError('evaluation_budget')
        y=v[:2];mat=v[2:-1].reshape(2,n+1)
        deriv=system.jac(y)@mat
        if n:deriv[:,:n]+=system.partials(y)
        return np.r_[system.field(y),deriv.ravel(),np.trace(system.jac(y))]
    def boundary(t,v):return min(*(v[:2]-system.bounds[:,0]),*(system.bounds[:,1]-v[:2]))
    boundary.direction=-1;boundary.terminal=True
    def extremum(t,v):return system.field(v[:2])[0]
    extremum.direction=0;extremum.terminal=False
    def capture(t,v):
        if not capture_points:return 1.
        return min(np.linalg.norm(v[:2]-p) for p in capture_points)-1e-8
    capture.direction=-1;capture.terminal=True
    now=0.;state=initial
    try:
        # Separating crossings removes the t=0 root without an arbitrary time nudge.
        for direction in [-orientation,orientation]:
            def crossing(t,v):return v[0]-zsec
            crossing.direction=direction;crossing.terminal=True
            sol=solve_ivp(rhs,[now,max_time],state,method=method,events=[crossing,boundary,capture,extremum],
                          rtol=rtol,atol=atol,max_step=max_step)
            paths.extend(sol.y[:2].T.tolist());times.extend(sol.t.tolist())
            if len(sol.y_events[3]):paths.extend(sol.y_events[3][:,:2].tolist())
            if not sol.success:return dict(status='solver_failure',u=float(u),message=sol.message)
            if len(sol.t_events[1]):return dict(status='domain_exit',u=float(u))
            if len(sol.t_events[2]):return dict(status='equilibrium_capture',u=float(u))
            if not len(sol.t_events[0]):return dict(status='timeout',u=float(u))
            now=float(sol.t_events[0][0]);state=sol.y_events[0][0]
        fy=system.field(state[:2]);normal_final=orientation*fy[0]
        if normal_final<1e-9:return dict(status='return_tangency',u=float(u))
        sens=state[2:-1].reshape(2,n+1)
        corrected=sens[1].copy()-fy[1]/fy[0]*sens[0]
        if n:corrected[:n]+=fy[1]/fy[0]*dzsec
        log_multiplier=float(np.log(abs(initial_field[0]/fy[0]))+state[-1])
        multiplier=float(np.exp(np.clip(log_multiplier,-700,700)))
        if not np.isfinite(multiplier) or multiplier<=0:return dict(status='nonregular_multiplier',u=float(u))
        returned=float(state[1]);den=h-section.h0
        zs=np.array(paths)[:,0]
        return dict(status='valid',u=float(u),s=float(h),P=returned,G=(returned-h)/den,
                    dG=(corrected[:n]/den).tolist(),multiplier=multiplier,log_multiplier=log_multiplier,period=now,
                    Cmin=system.physical(float(zs.min())),Cmax=system.physical(float(zs.max())),
                    initial_normal=float(normal),return_normal=float(normal_final))
    except (RuntimeError,ValueError,FloatingPointError,OverflowError) as exc:
        return dict(status='integration_error',u=float(u),message=str(exc))


def return_table(system,section,nodes,sensitivities=False,**settings):
    return [first_return(system,section,u,sensitivities,**settings) for u in nodes]


def smooth_counts(records,epsilon,xi=1e-7,nparams=0):
    """Discrete variation: never connect a pair across a failed return node."""
    values=np.zeros(2);grad=np.zeros((2,nparams));edges=0
    for left,right in zip(records[:-1],records[1:]):
        if left['status']!='valid' or right['status']!='valid':continue
        a,b=left['G']/epsilon,right['G']/epsilon
        v=float(ndtr(a)-ndtr(b));root=np.sqrt(v*v+xi*xi)
        values+=np.array([.5*(v+root),.5*(-v+root)])
        if nparams:
            da=np.exp(-.5*a*a)/np.sqrt(2*np.pi)/epsilon*np.asarray(left['dG'])
            db=np.exp(-.5*b*b)/np.sqrt(2*np.pi)/epsilon*np.asarray(right['dG'])
            grad[0]+=.5*(1+v/root)*(da-db)
            grad[1]+=.5*(-1+v/root)*(da-db)
        edges+=1
    return values,grad,edges


def isolate_cycles(system,section,nodes,settings,root_tolerance=1e-8):
    table=return_table(system,section,nodes,False,**settings);cycles=[];suspects=[]
    for left,right in zip(table[:-1],table[1:]):
        if left['status']!='valid' or right['status']!='valid':continue
        if left['G']*right['G']>0:
            if min(abs(left['G']),abs(right['G']))<1e-4:suspects.append([left['u'],right['u']])
            continue
        def displacement(u):
            result=first_return(system,section,u,False,**settings)
            if result['status']!='valid':raise ValueError('return_domain_gap_inside_bracket')
            return result['G']
        try:
            u=brentq(displacement,left['u'],right['u'],xtol=root_tolerance)
            result=first_return(system,section,u,False,**settings)
            if result['status']!='valid':continue
            lam=result['multiplier'];kind='stable' if lam<1-1e-4 else ('unstable' if lam>1+1e-4 else 'degenerate')
            if not any(abs(u-q['u'])<1e-6 for q in cycles):
                cycles.append(dict(**result,kind=kind,section_C=section.C,section_h0=section.h0))
        except ValueError as exc:suspects.append(dict(interval=[left['u'],right['u']],message=str(exc)))
    return dict(cycles=cycles,records=table,suspect_intervals=suspects,
                regular_edges=sum(a['status']==b['status']=='valid' for a,b in zip(table[:-1],table[1:])))
