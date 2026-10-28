"""AM/FM nested PWA refinements with exact rest and threshold events."""
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]


class Model:
    def __init__(self,case,coordinate='mixed',construction='refined',fm_lower_type='node'):
        if case not in ('am','fm') or coordinate not in ('mixed','affine') or construction not in ('refined','base','split','split_fixed'):raise ValueError('Invalid case, coordinate or construction')
        if fm_lower_type not in ('node','focus') or (fm_lower_type=='focus' and (case!='fm' or construction not in ('base','split','split_fixed'))) or (construction in ('split','split_fixed') and (case!='fm' or fm_lower_type!='focus')):raise ValueError('Split construction requires FM with lower focus')
        self.case=case;self.coordinate=coordinate;self.construction=construction;self.fm_lower_type=fm_lower_type
        if case=='am':
            import os
            os.environ['MAP_KIND']='affine' if coordinate=='affine' else 'mixed_power'
            sys.path.insert(0,str(ROOT/'am_two_thresholds'))
            import fit_am as f
            self.f=f;self.base=np.asarray(f.FIXED_TC);self.event_I=np.asarray(f.EVENTS);self.rest=np.asarray(f.eq(0.))
            self.kind=np.array([0,1,1,1,2]);self.event_indices=[0,3]
            self.default_parent=ROOT/'am_two_thresholds/fits/mixed_power_fixed/candidate.json'
        else:
            import os
            os.environ['FM_COORDINATE']=coordinate
            sys.path.insert(0,str(ROOT/'fm_four_zone_fit'))
            import fit_four as f
            self.f=f;self.base=np.asarray(f.TC);self.event_I=np.asarray(f.FIXED_EVENT_I);self.rest=np.asarray(f.eqs(0.)[0])
            self.kind=np.array([0,1,1,1,2,2,2,3]);self.event_indices=[0,6]
            self.default_parent=ROOT/'fm_four_zone_fit/candidate.json'
        cuts=list(self.base)
        intervals=[] if construction in ('base','split','split_fixed') else ([(self.base[0],self.base[1])] if case=='am' else [(self.base[0],self.base[1]),(self.base[1],self.base[2])])
        if construction=='split':cuts.append(0.05214116272147648)
        if construction=='split_fixed':
            cuts=[0.05214116272147648,0.06128930910964862,0.1357042423799119,float(self.base[2])]
            self.event_I=np.array([0.5290700389163683,0.5439919432478553,0.48230398233782223,0.8600205557570602])
        for lo,hi in intervals:cuts.extend([lo+(hi-lo)/3,lo+2*(hi-lo)/3])
        self.cuts=np.sort(cuts);self.n=len(self.cuts)+1
        if construction=='base':
            self.kind=np.arange(self.n);self.event_indices=[0,len(self.cuts)-1]
        if construction in ('split','split_fixed'):
            self.kind=np.array([0,2,1,2,3]);self.event_indices=[0,len(self.cuts)-1]
        if construction=='split_fixed':self.event_indices=list(range(4))
        self.n0=2+2*self.n;self.h0=self.n0+1;self.ni=self.n0+2;self.hi=self.n0+3;self.shape=self.n0+4
        self.size=self.shape+4;self.dep_region=2 if construction in ('split','split_fixed') else 1;self.dep_det=2+self.n+self.dep_region
        self.derived=([2+self.n+r for r in (1,2,3)] if construction=='split_fixed' else [self.dep_det])+[self.n0,self.h0,self.hi]
        self.active=np.array([i for i in range(self.size) if i not in self.derived and not (coordinate=='affine' and i>=self.shape)],dtype=int)
        if case=='am':
            lo_u=np.array([-5.,-6.,-5.])[self.kind];hi_u=np.array([3.,2.,2.])[self.kind]
            lo_d=np.array([-8.,-9.,-9.])[self.kind];hi_d=np.array([2.,1.,1.])[self.kind]
            self.lo=np.r_[np.log([.15,.03]),lo_u,lo_d,[-1,-1,-2,-5],[-5,-6,.15,-4]]
            self.upper=np.r_[np.log([40,1.5]),hi_u,hi_d,[2,2,2,1],[5,3,3.,4]]
        else:
            lo_u=np.array([-5.,-6.,-5.,-5.])[self.kind];hi_u=np.array([3.,6.,2.,2.])[self.kind]
            lo_d=np.array([-8.,-9.,-9.,-9.])[self.kind];hi_d=np.array([8.,1.,1.,1.])[self.kind]
            self.lo=np.r_[np.log([.2,.03]),lo_u,lo_d,[-2,-2,-2,-2],[-4,-6,.15,-4]]
            self.upper=np.r_[np.log([40,1.2]),hi_u,hi_d,[3,3,2,2],[5,3,2.5,4]]
            if fm_lower_type=='focus':self.lo[2+self.n]=-9.;self.upper[2+self.n]=1.
        self.hi_bounds=self.upper;self.fixed=self.event_indices

    def unpack(self,x):
        x=np.asarray(x,float)
        if x.shape!=(self.size,):raise ValueError('Bad parameter vector size')
        a,b=np.exp(x[:2]);u=x[2:2+self.n];v=x[2+self.n:2+2*self.n]
        if self.case=='am':
            T=np.where(self.kind==1,np.exp(u),-np.exp(u))
            D=np.where(self.kind==0,np.exp(v),T*T/4+np.exp(v))
        else:
            T=np.where(self.kind==1,u,np.where(self.kind==2,np.exp(u),-np.exp(u)))
            D=np.where(self.kind==0,T*T/4*expit(v),np.where(self.kind==1,-np.exp(v),T*T/4+np.exp(v)))
            if self.fm_lower_type=='focus':D[0]=T[0]**2/4+np.exp(v[0])
        ns=-(T+b)/a;hs=ns-D/(a*b)
        shape=None if self.coordinate=='affine' else x[self.shape:]
        th=self.f.phi(self.cuts,shape)
        N=np.r_[0.,ns[0],x[self.ni],np.diff(ns)];H=np.r_[0.,hs[0],x[self.hi],np.diff(hs)]
        at=np.r_[1.,0.,.8,np.maximum(-th,0.)];N[0]=x[self.n0]-at@N;H[0]=x[self.h0]-at@H
        co=np.zeros((2,11));idx=[0,1,3,*range(4,4+len(th))]
        co[0,idx]=-a*N;co[0,2]=a;co[1,idx]=b*H;co[1,2]=-b
        return dict(a=a,b=b,T=T,D=D,N=N,H=H,shape=shape,th=th,tc=self.cuts,co=co,padded=np.r_[th,np.full(7-len(th),1e100)])

    def basis(self,c,I,m):
        z,I=np.broadcast_arrays(self.f.phi(np.asarray(c),m['shape']),I)
        return np.stack([np.ones_like(z),z,I,*[np.maximum(z-t,0.) for t in m['th']]],axis=-1)

    def constrain_events(self,x):
        x=np.asarray(x,float).copy();m=self.unpack(x);zr=float(self.f.phi(self.rest[0],m['shape']))
        z=np.r_[zr,m['th']];last=self.event_indices[-1]
        if not zr<m['th'][0]:raise ValueError('LR rest is not in first region')
        D=m['D'];width=np.diff(z)
        if self.construction=='split_fixed':
            I0=self.event_I[0]
            target=D[0]*width[0]*np.diff(self.event_I)/(I0*width[1:4])
            for r,value in enumerate(target,1):
                q=-value if self.kind[r]==1 else value-m['T'][r]**2/4
                if not np.isfinite(q) or q<=0:raise ValueError('Fixed events incompatible with regional types')
                idx=2+self.n+r;x[idx]=np.log(q)
                if not self.lo[idx]<x[idx]<self.upper[idx]:raise ValueError('Derived determinant outside bounds')
            x[self.hi]=x[self.ni]+D[0]*width[0]/(m['a']*m['b']*I0)
            m=self.unpack(x);rest_basis=self.basis(self.rest[0],0.,m)
            x[self.n0]+=self.rest[1]-rest_basis@m['N'];x[self.h0]+=self.rest[1]-rest_basis@m['H']
            return x
        I0,I1=self.event_I
        dep=self.dep_region
        required=(D[0]*width[0]*(I1/I0-1)-sum(D[r]*width[r] for r in range(1,last+1) if r!=dep))/width[dep]
        if self.case=='am':
            q=required-m['T'][1]**2/4
            if not np.isfinite(q) or q<=0:raise ValueError('Rest/event constraints incompatible with unstable focus')
            x[self.dep_det]=np.log(q)
        else:
            if not np.isfinite(required) or required>=0:raise ValueError('Rest/event constraints incompatible with saddle')
            x[self.dep_det]=np.log(-required)
        if not self.lo[self.dep_det]<x[self.dep_det]<self.upper[self.dep_det]:raise ValueError('Derived determinant outside bounds')
        x[self.hi]=x[self.ni]+D[0]*width[0]/(m['a']*m['b']*I0)
        m=self.unpack(x);rest_basis=self.basis(self.rest[0],0.,m)
        x[self.n0]+=self.rest[1]-rest_basis@m['N'];x[self.h0]+=self.rest[1]-rest_basis@m['H']
        return x

    def event_heights(self,x):
        m=self.unpack(x)
        return np.array([self.basis(self.cuts[j],I,m)@m['N'] for j,I in zip(self.event_indices,self.event_I)])

    def physically_valid_events(self,x):
        h=self.event_heights(x);return bool(np.isfinite(h).all() and np.all((h>0)&(h<1)))

    def equilibrium(self,I,m):
        n=m['N'][1]+np.r_[0,np.cumsum(m['N'][3:])];h=m['H'][1]+np.r_[0,np.cumsum(m['H'][3:])]
        nc=m['N'][0]-np.r_[0,np.cumsum(m['N'][3:]*m['th'])];hc=m['H'][0]-np.r_[0,np.cumsum(m['H'][3:]*m['th'])]
        z=(hc-nc+(m['H'][2]-m['N'][2])*I)/(n-h);y=nc+n*z+m['N'][2]*I
        valid=(z>np.r_[-np.inf,m['th']])&(z<=np.r_[m['th'],np.inf])&(y>0)&(y<1);out=[]
        for r in np.flatnonzero(valid):
            c=float(self.f.inverse(np.asarray(z[r]),m))
            if 0<c<self.f.CMAX:out.append(dict(z=[float(z[r]),float(y[r])],C=c,region=int(r),stable=bool(m['T'][r]<0 and m['D'][r]>0)))
        return out

    def lift_parent(self,parent):
        if not np.allclose(parent['physical_thresholds_C'],self.base,rtol=0,atol=1e-10):raise ValueError('Parent physical thresholds mismatch')
        if (parent.get('transform') is None)!=(self.coordinate=='affine'):raise ValueError('Parent coordinate family mismatch')
        old=np.asarray(parent['parameters'],float);nold=len(self.base)+1;idx=self.kind
        x=np.r_[old[:2],old[2:2+nold][idx],old[2+nold:2+2*nold][idx],old[2+2*nold:6+2*nold],old[6+2*nold:10+2*nold]]
        if self.fm_lower_type=='focus' and parent.get('fm_lower_type')!='focus':x[2+self.n]=-6.
        if self.construction=='split_fixed':return self.initialize_fixed_events(x)
        try:
            constrained=self.constrain_events(x)
            if self.physically_valid_events(constrained):return constrained
        except ValueError as exc:
            if str(exc) not in ('Rest/event constraints incompatible with saddle','Rest/event constraints incompatible with unstable focus','Derived determinant outside bounds'):
                raise
        # The old fit did not fix the rest point; require admissible event
        # heights as well as a saddle determinant during initialization.
        # Keeping the cuts and transform fixed, vary regional slopes and the
        # input coefficient; the remaining coefficients are derived exactly.
        if self.case in ('am','fm'):
            from scipy.optimize import minimize
            last=self.event_indices[-1]
            indices=np.array([2+self.n+r for r in range(last+1) if r!=self.dep_region]+[2+r for r in range(last+1)]+[self.ni])
            original=x[indices].copy()
            bounds=list(zip(self.lo[indices]+1e-7,self.upper[indices]-1e-7))
            initial=np.clip(original,*np.asarray(bounds).T)
            def required(y):
                trial=x.copy();trial[indices]=y;m=self.unpack(trial)
                z=np.r_[float(self.f.phi(self.rest[0],m['shape'])),m['th']]
                width=np.diff(z);I0,I1=self.event_I;last=self.event_indices[-1]
                dep=self.dep_region
                return (m['D'][0]*width[0]*(I1/I0-1)-sum(m['D'][r]*width[r] for r in range(1,last+1) if r!=dep))/width[dep]
            def heights(y):
                trial=x.copy();trial[indices]=y;m=self.unpack(trial)
                trial[self.n0]+=self.rest[1]-self.basis(self.rest[0],0.,m)@m['N']
                return self.event_heights(trial)
            def difference(y):
                if self.case=='fm':return -required(y)
                trial=x.copy();trial[indices]=y
                return required(y)-self.unpack(trial)['T'][1]**2/4
            lower=np.exp(self.lo[self.dep_det]+1e-5);upper=np.exp(self.upper[self.dep_det]-1e-5)
            result=minimize(lambda y: float(np.sum((y-original)**2)),initial,method='SLSQP',bounds=bounds,
                constraints=[{'type':'ineq','fun':lambda y:difference(y)-lower},
                             {'type':'ineq','fun':lambda y:upper-difference(y)},
                             {'type':'ineq','fun':lambda y:heights(y)-1e-4},
                             {'type':'ineq','fun':lambda y:1.-1e-4-heights(y)}],
                options={'maxiter':1000,'ftol':1e-12})
            x[indices]=result.x
            try:
                x=self.constrain_events(x)
            except ValueError as failure:
                raise ValueError(f'{self.case.upper()} initialization could not satisfy rest and event constraints: {result.message}; {failure}') from failure
            if not self.physically_valid_events(x):
                raise ValueError(f'{self.case.upper()} initialization could not find physical event heights: {self.event_heights(x).tolist()}; {result.message}')
            print(f'{self.case.upper()} initialization adjusted regional parameters: distance={np.linalg.norm(result.x-original):.6g}, event heights={self.event_heights(x).tolist()}, optimizer={result.message}',flush=True)
            return x

    def initialize_fixed_events(self,x):
        from scipy.optimize import minimize
        indices=np.r_[[2+self.n],np.arange(2,2+4),[self.ni],np.arange(self.shape,self.shape+4)]
        original=x[indices].copy();bounds=list(zip(self.lo[indices]+1e-6,self.upper[indices]-1e-6))
        initial=np.clip(original,*np.asarray(bounds).T)
        def trial(y):
            q=x.copy();q[indices]=y;return q
        def margins(y):
            q=trial(y);m=self.unpack(q);z=np.r_[float(self.f.phi(self.rest[0],m['shape'])),m['th']]
            widths=np.diff(z);det=m['D'][0]*widths[0]*np.diff(self.event_I)/(self.event_I[0]*widths[1:4])
            vals=np.array([det[0]-m['T'][1]**2/4,-det[1],det[2]-m['T'][3]**2/4])
            lo=np.exp(self.lo[[2+self.n+r for r in (1,2,3)]]+1e-4)
            hi=np.exp(self.upper[[2+self.n+r for r in (1,2,3)]]-1e-4)
            q[self.n0]+=self.rest[1]-self.basis(self.rest[0],0.,m)@m['N']
            heights=self.event_heights(q)
            return np.r_[vals-lo,hi-vals,heights-1e-4,1.-1e-4-heights]
        result=minimize(lambda y:float(np.sum((y-original)**2)),initial,method='SLSQP',bounds=bounds,
                        constraints=[dict(type='ineq',fun=margins)],options=dict(maxiter=1500,ftol=1e-12))
        try:
            final=self.constrain_events(trial(result.x))
            if not self.physically_valid_events(final):raise ValueError('Nonphysical fixed event heights')
        except ValueError as exc:
            raise ValueError(f'Cannot initialize all four exact events: {result.message}; {exc}; min margin={np.min(margins(result.x)):.4g}') from exc
        print(f'FM four-event initialization: distance={np.linalg.norm(result.x-original):.6g} heights={self.event_heights(final).tolist()} status={result.message}',flush=True)
        return final
