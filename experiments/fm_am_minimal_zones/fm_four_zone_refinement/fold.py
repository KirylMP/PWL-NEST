"""Boundary matching for a cycle traversing all fixed thresholds.

A solution is a local fold on this particular transverse itinerary. It does
not establish global absence of other cycles or folds.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

class FoldError(ValueError):
    pass

class BoundaryFold:
    def __init__(self, model, target_I, first=0, last=None):
        self.model=model; self.I=float(target_I)
        last=len(model.cuts)-1 if last is None else last
        if not 0<=first<=last<len(model.cuts):raise ValueError('Invalid crossing interval')
        self.boundaries=tuple(range(first,last+1))+tuple(range(last,first-1,-1))+(first,)
        self.regions=tuple(range(first+1,last+2))+tuple(range(last,first-1,-1))
        self.legs=2*(last-first+1)

    def matching(self,x,times,details=False):
        m=self.model.unpack(x); z=m['th'][list(self.boundaries)]
        heights=[]; ends=[]; matrices=[]
        for j,r in enumerate(self.regions):
            co=m['co']; M=np.zeros((3,3))
            M[:2,:2]=np.column_stack((co[:,1]+co[:,4:4+r].sum(axis=1),co[:,2]))
            M[:2,2]=co[:,0]+co[:,3]*self.I-co[:,4:4+r]@m['th'][:r]
            Q=expm(M*times[j]);E=Q[:2,:2];g=Q[:2,2]
            if not np.isfinite(E).all() or abs(E[0,1])<1e-12:raise FoldError('Singular boundary segment')
            h=(z[j+1]-E[0,0]*z[j]-g[0])/E[0,1]
            heights.append(h);ends.append(E[1,0]*z[j]+E[1,1]*h+g[1]);matrices.append(M)
        residual=np.asarray(ends)-np.roll(heights,-1)
        return (residual,m,z,np.asarray(heights),matrices) if details else residual

    def solve(self,raw,times,index,max_nfev=200):
        model=self.model; base=model.constrain_events(raw); n=self.legs
        lower=np.r_[np.full(n,1e-5),model.lo[index]+1e-8]
        upper=np.r_[np.full(n,30.),model.hi_bounds[index]-1e-8]
        def reconstruct(v):
            x=base.copy();x[index]=v[-1];return model.constrain_events(x)
        def fun(v):
            try:
                x=reconstruct(v);m=model.unpack(x)
                result=np.r_[self.matching(x,v[:n]),sum(m['T'][r]*t for r,t in zip(self.regions,v[:n]))/10]
                return result if np.isfinite(result).all() else np.full(n+1,1e6)
            except (ValueError,FloatingPointError,OverflowError):return np.full(n+1,1e6)
        guess=np.clip(np.r_[times,base[index]],lower+1e-9,upper-1e-9)
        sol=least_squares(fun,guess,bounds=(lower,upper),max_nfev=max_nfev,ftol=1e-11,xtol=1e-11,gtol=1e-11,x_scale='jac')
        x=reconstruct(sol.x);t=sol.x[:n];err=fun(sol.x)
        if max(abs(err[:n]))>1e-8 or abs(err[n])>1e-8:raise FoldError(f'Fold not resolved: residual={max(abs(err)):.3g}')
        if min(sol.x[-1]-lower[-1],upper[-1]-sol.x[-1])<1e-7:raise FoldError('Dependent parameter at bound')
        if not model.physically_valid_events(x):raise FoldError('Nonphysical fixed-event height')
        residual,m,z,h,matrices=self.matching(x,t,True)
        all_states=[]
        for j,(r,M) in enumerate(zip(self.regions,matrices)):
            y0=np.r_[z[j],h[j],1.]; speed=(M@y0)[0]
            if (1 if j<n//2 else -1)*speed<=1e-8:raise FoldError('Wrong or grazing crossing')
            states=np.array([(expm(M*s)@y0)[:2] for s in np.linspace(0,t[j],65)])
            lo=-np.inf if r==0 else m['th'][r-1];hi=np.inf if r==model.n-1 else m['th'][r]
            if not np.isfinite(states).all() or min(states[:,0])<lo-1e-7 or max(states[:,0])>hi+1e-7:raise FoldError('Segment leaves region')
            if min(states[:,1])<=0 or max(states[:,1])>=1:raise FoldError('Segment leaves physical h domain')
            all_states.extend(states)
        state=np.asarray(all_states)
        if hasattr(model,'f'):
            reference=model.f
        else:
            from model import reference
        zlo,zhi=reference.phi(np.array([1e-12,reference.CMAX]),m['shape'])
        if min(state[:,0])<=zlo or max(state[:,0])>=zhi:raise FoldError('Segment leaves physical C domain')
        eps=1e-5;eye=np.eye(n)
        J=np.column_stack([(self.matching(x,t+eps*e)-self.matching(x,t-eps*e))/(2*eps) for e in eye])
        U,s,V=np.linalg.svd(J);left=U[:,-1];right=V[-1]
        step=1e-4
        curvatures=[float(left@(self.matching(x,t+step*right)-2*residual+self.matching(x,t-step*right))/step**2) for step in (1e-3,3e-4,1e-4)]
        curvature=curvatures[1]
        xp=x.copy();xm=x.copy();xp[index]+=1e-6;xm[index]-=1e-6
        dp=float(left@(self.matching(model.constrain_events(xp),t)-self.matching(model.constrain_events(xm),t))/(2e-6))
        dI=1e-6; saved=self.I
        try:
            self.I=saved+dI;plus=self.matching(x,t);self.I=saved-dI;minus=self.matching(x,t)
        finally:self.I=saved
        unfold=float(left@(plus-minus)/(2*dI))
        if s[-1]>1e-6*s[0] or s[-2]<1e-8*s[0] or min(abs(v) for v in curvatures)<1e-9 or len({np.sign(v) for v in curvatures})!=1 or abs(unfold)<1e-8 or abs(dp)<1e-8 or curvature*unfold<=0:
            raise FoldError('Fold degeneracy or upper-bound orientation unresolved')
        return x,dict(I=self.I,parameter_index=index,crossing_times=t.tolist(),residual=float(max(abs(err[:n]))),trace_integral=float(err[n]*10),singular_values=s.tolist(),curvature_samples=curvatures,input_derivative=unfold,parameter_derivative=dp,scope='local fold on all-threshold itinerary; finite segment sampling')
