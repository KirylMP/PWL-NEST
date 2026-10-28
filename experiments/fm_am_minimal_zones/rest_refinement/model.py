"""AM/FM nested PWA refinements with exact rest and threshold events."""
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]


class Model:
    def __init__(self,case,coordinate='mixed',construction='refined',fm_lower_type='node'):
        upper_modes=('upper70','upper100','upper_both','upper_three')
        if case not in ('am','fm') or coordinate not in ('mixed','affine') or construction not in ('refined','base','split','split_fixed','six_fixed','full_fixed',*upper_modes):raise ValueError('Invalid case, coordinate or construction')
        if fm_lower_type not in ('node','focus') or (fm_lower_type=='focus' and (case!='fm' or construction not in ('base','split','split_fixed'))) or (construction in ('split','split_fixed') and (case!='fm' or fm_lower_type!='focus')) or (construction in ('six_fixed','full_fixed',*upper_modes) and (case!='fm' or fm_lower_type!='node')):raise ValueError('Invalid FM lower type for construction')
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
        intervals=[] if construction in ('base','split','split_fixed','six_fixed','full_fixed',*upper_modes) else ([(self.base[0],self.base[1])] if case=='am' else [(self.base[0],self.base[1]),(self.base[1],self.base[2])])
        if construction=='split':cuts.append(0.05214116272147648)
        if construction=='split_fixed':
            cuts=[0.05214116272147648,0.06128930910964862,0.1357042423799119,float(self.base[2])]
            self.event_I=np.array([0.5290700389163683,0.5439919432478553,0.48230398233782223,0.8600205557570602])
        if construction=='six_fixed':
            cuts=[0.04861935111371004,0.06128930910964862,0.1357042423799119,0.25181841575270236,0.3926251934101712]
            self.event_I=np.array([0.5107261652060441,0.5439919432478553,0.48230398233782223,0.5747315355622029,0.8600205557570602])
            self.default_parent=ROOT/'fm_four_zone_fit/previous_selected.json'
        if construction in ('full_fixed',*upper_modes):
            cuts=[0.04861935111371004,0.05214116272147648,0.05484469312764684,0.06128930910964862,0.1357042423799119,0.25181841575270236,0.3926251934101712]
            self.event_I=np.array([0.5107261652060441,0.5290700389163682,0.5373937039464933,0.5439919432478553,0.48230398233782223,0.5747315355622029,0.8600205557570602])
            self.default_parent=ROOT/'fm_four_zone_fit/previous_selected.json'
            cuts+=({'upper70':[.7],'upper100':[1.0],'upper_both':[.7,1.0],'upper_three':[.7,.95,1.1]}.get(construction,[]))
        for lo,hi in intervals:cuts.extend([lo+(hi-lo)/3,lo+2*(hi-lo)/3])
        self.cuts=np.sort(cuts);self.n=len(self.cuts)+1
        if construction=='base':
            self.kind=np.arange(self.n);self.event_indices=[0,len(self.cuts)-1]
        if construction in ('split','split_fixed'):
            self.kind=np.array([0,2,1,2,3]);self.event_indices=[0,len(self.cuts)-1]
        if construction=='split_fixed':self.event_indices=list(range(4))
        if construction=='six_fixed':self.kind=np.array([0,3,1,4,2,3]);self.event_indices=list(range(5))
        if construction in ('full_fixed',*upper_modes):self.kind=np.r_[[0,3,2,4,1,4,2,3],np.full(len(self.cuts)-7,3)].astype(int);self.event_indices=list(range(7))
        self.n0=2+2*self.n;self.h0=self.n0+1;self.ni=self.n0+2;self.hi=self.n0+3;self.shape=self.n0+4
        self.cut_start=self.shape+4;self.extra_cuts=len(self.cuts)-7 if construction in upper_modes else 0
        self.size=self.cut_start+self.extra_cuts;self.dep_region=2 if construction in ('split','split_fixed') else 1;self.dep_det=2+self.n+self.dep_region
        self.derived=([2+self.n+r for r in range(1,len(self.event_I))] if construction in ('split_fixed','six_fixed','full_fixed',*upper_modes) else [self.dep_det])+[self.n0,self.h0,self.hi]
        self.active=np.array([i for i in range(self.size) if i not in self.derived and not (coordinate=='affine' and self.shape<=i<self.cut_start)],dtype=int)
        if case=='am':
            lo_u=np.array([-5.,-6.,-5.])[self.kind];hi_u=np.array([3.,2.,2.])[self.kind]
            lo_d=np.array([-8.,-9.,-9.])[self.kind];hi_d=np.array([2.,1.,1.])[self.kind]
            self.lo=np.r_[np.log([.15,.03]),lo_u,lo_d,[-1,-1,-2,-5],[-5,-6,.15,-4]]
            self.upper=np.r_[np.log([40,1.5]),hi_u,hi_d,[2,2,2,1],[5,3,3.,4]]
        else:
            lo_u=np.array([-5.,-6.,-5.,-5.,-5.])[self.kind];hi_u=np.array([3.,6.,2.,2.,2.])[self.kind]
            lo_d=np.array([-8.,-9.,-9.,-9.,-8.])[self.kind];hi_d=np.array([8.,1.,1.,1.,8.])[self.kind]
            self.lo=np.r_[np.log([.2,.03]),lo_u,lo_d,[-2,-2,-2,-2],[-4,-6,.15,-4]]
            self.upper=np.r_[np.log([40,1.2]),hi_u,hi_d,[3,3,2,2],[5,3,2.5,4]]
            if fm_lower_type=='focus':self.lo[2+self.n]=-9.;self.upper[2+self.n]=1.
        if self.extra_cuts:
            if self.extra_cuts==1:lo,hi=[.45],[1.35]
            elif self.extra_cuts==2:lo,hi=[.45,.905],[.895,1.35]
            else:lo,hi=[.45,.905,1.055],[.895,1.045,1.35]
            self.lo=np.r_[self.lo,lo];self.upper=np.r_[self.upper,hi]
        self.hi_bounds=self.upper;self.fixed=self.event_indices

    def unpack(self,x):
        x=np.asarray(x,float)
        if x.shape!=(self.size,):raise ValueError('Bad parameter vector size')
        a,b=np.exp(x[:2]);u=x[2:2+self.n];v=x[2+self.n:2+2*self.n]
        if self.case=='am':
            T=np.where(self.kind==1,np.exp(u),-np.exp(u))
            D=np.where(self.kind==0,np.exp(v),T*T/4+np.exp(v))
        else:
            T=np.where(self.kind==1,u,np.where((self.kind==2)|(self.kind==4),np.exp(u),-np.exp(u)))
            D=np.where((self.kind==0)|(self.kind==4),T*T/4*expit(v),np.where(self.kind==1,-np.exp(v),T*T/4+np.exp(v)))
            if self.fm_lower_type=='focus':D[0]=T[0]**2/4+np.exp(v[0])
        ns=-(T+b)/a;hs=ns-D/(a*b)
        shape=None if self.coordinate=='affine' else x[self.shape:self.cut_start]
        cuts=np.r_[self.cuts[:7],x[self.cut_start:]] if self.extra_cuts else self.cuts
        if not np.all(np.diff(cuts)>0):raise ValueError('Calcium thresholds are not strictly ordered')
        th=self.f.phi(cuts,shape)
        N=np.r_[0.,ns[0],x[self.ni],np.diff(ns)];H=np.r_[0.,hs[0],x[self.hi],np.diff(hs)]
        at=np.r_[1.,0.,.8,np.maximum(-th,0.)];N[0]=x[self.n0]-at@N;H[0]=x[self.h0]-at@H
        capacity=10 if self.extra_cuts==3 else (9 if self.extra_cuts else 7)
        co=np.zeros((2,4+capacity));idx=[0,1,3,*range(4,4+len(th))]
        co[0,idx]=-a*N;co[0,2]=a;co[1,idx]=b*H;co[1,2]=-b
        return dict(a=a,b=b,T=T,D=D,N=N,H=H,shape=shape,th=th,tc=cuts,co=co,padded=np.r_[th,np.full(capacity-len(th),1e100)])

    def basis(self,c,I,m):
        z,I=np.broadcast_arrays(self.f.phi(np.asarray(c),m['shape']),I)
        return np.stack([np.ones_like(z),z,I,*[np.maximum(z-t,0.) for t in m['th']]],axis=-1)

    def constrain_events(self,x):
        x=np.asarray(x,float).copy();m=self.unpack(x);zr=float(self.f.phi(self.rest[0],m['shape']))
        z=np.r_[zr,m['th']];last=self.event_indices[-1]
        if not zr<m['th'][0]:raise ValueError('LR rest is not in first region')
        D=m['D'];width=np.diff(z)
        if self.construction in ('split_fixed','six_fixed','full_fixed','upper70','upper100','upper_both','upper_three'):
            I0=self.event_I[0]
            target=D[0]*width[0]*np.diff(self.event_I)/(I0*width[1:len(self.event_I)])
            for r,value in enumerate(target,1):
                if self.kind[r]==4:
                    ratio=value/(m['T'][r]**2/4)
                    if not np.isfinite(ratio) or not 0<ratio<1:raise ValueError('Fixed events incompatible with unstable node')
                    param=np.log(ratio/(1-ratio))
                else:
                    q=-value if self.kind[r]==1 else value-m['T'][r]**2/4
                    if not np.isfinite(q) or q<=0:raise ValueError('Fixed events incompatible with regional types')
                    param=np.log(q)
                idx=2+self.n+r;x[idx]=param
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
        return np.array([self.basis(m['tc'][j],I,m)@m['N'] for j,I in zip(self.event_indices,self.event_I)])

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
        if self.extra_cuts:
            if parent.get('case')!='fm' or parent.get('construction')!='full_fixed' or parent.get('stage')!='closed':raise ValueError('Upper refinement requires a closed full_fixed FM candidate')
            if not np.allclose(parent['physical_thresholds_C'],self.cuts[:7],rtol=0,atol=1e-10):raise ValueError('Parent fixed threshold mismatch')
            old=np.asarray(parent['parameters'],float);nold=8
            if old.shape!=(2+2*nold+8,):raise ValueError('Parent parameter layout mismatch')
            idx=np.r_[np.arange(nold),np.full(self.extra_cuts,nold-1)].astype(int)
            x=np.r_[old[:2],old[2:2+nold][idx],old[2+nold:2+2*nold][idx],old[2+2*nold:],self.cuts[7:]]
            x=self.constrain_events(x)
            if not self.physically_valid_events(x):raise ValueError('Parent has nonphysical fixed events')
            return x
        if self.construction in ('six_fixed','full_fixed') and ('candidate' in parent or len(parent.get('physical_thresholds_C',[]))==7):
            parent=parent.get('candidate',parent)
            if len(parent['physical_thresholds_C'])!=7 or parent.get('transform') is None:raise ValueError('Expected transformed 8-region FM parent')
            a=float(parent['a']);b=float(parent['b']);alpha=np.asarray(parent['alpha'],float);beta=np.asarray(parent['beta'],float)
            shape=np.array([parent['transform'][k] for k in ('p','e','n','k')],float)
            old_th=self.f.phi(np.asarray(parent['physical_thresholds_C'],float),shape)
            selected=[0,3,6,5,4,5,6,7] if self.construction=='full_fixed' else [0,3,4,5,6,7];u=[];v=[]
            for r,kind in zip(selected,self.kind):
                ns=alpha[1]+sum(alpha[3:3+r]);hs=beta[1]+sum(beta[3:3+r]);T=-a*ns-b;D=a*b*(ns-hs)
                if kind==1:
                    if D>=0:raise ValueError('Parent saddle type mismatch')
                    u.append(T);v.append(np.log(-D))
                elif kind in (0,4):
                    ratio=D/(T*T/4)
                    if not 0<ratio<1 or (T>=0)!=(kind==4):raise ValueError('Parent node type mismatch')
                    u.append(np.log(abs(T)));v.append(np.log(ratio/(1-ratio)))
                else:
                    q=D-T*T/4
                    if q<=0 or (T>=0)!=(kind==2):raise ValueError('Parent focus type mismatch')
                    u.append(np.log(abs(T)));v.append(np.log(q))
            at=np.r_[1.,0.,.8,np.maximum(-old_th,0.)]
            x=np.r_[np.log([a,b]),u,v,alpha@at,beta@at,alpha[2],beta[2],shape]
            return self.initialize_fixed_events(x)
        if self.construction=='full_fixed':raise ValueError('Full FM initialization requires the 8-region candidate')
        if not np.allclose(parent['physical_thresholds_C'],self.base,rtol=0,atol=1e-10):raise ValueError('Parent physical thresholds mismatch')
        if (parent.get('transform') is None)!=(self.coordinate=='affine'):raise ValueError('Parent coordinate family mismatch')
        old=np.asarray(parent['parameters'],float);nold=len(self.base)+1;idx=np.array([0,3,1,2,2,3]) if self.construction=='six_fixed' else self.kind
        x=np.r_[old[:2],old[2:2+nold][idx],old[2+nold:2+2*nold][idx],old[2+2*nold:6+2*nold],old[6+2*nold:10+2*nold]]
        if self.fm_lower_type=='focus' and parent.get('fm_lower_type')!='focus':x[2+self.n]=-6.
        if self.construction in ('split_fixed','six_fixed','full_fixed'):return self.initialize_fixed_events(x)
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
        indices=np.r_[[2+self.n],np.arange(2,2+len(self.event_I)),[self.ni],np.arange(self.shape,self.shape+4)]
        original=x[indices].copy();bounds=list(zip(self.lo[indices]+1e-6,self.upper[indices]-1e-6))
        initial=np.clip(original,*np.asarray(bounds).T)
        def trial(y):
            q=x.copy();q[indices]=y;return q
        def margins(y):
            q=trial(y);m=self.unpack(q);z=np.r_[float(self.f.phi(self.rest[0],m['shape'])),m['th']]
            widths=np.diff(z);det=m['D'][0]*widths[0]*np.diff(self.event_I)/(self.event_I[0]*widths[1:len(self.event_I)])
            vals=[];lo=[];hi=[]
            for r,d in enumerate(det,1):
                idx=2+self.n+r
                if self.kind[r]==4:
                    vals.append(d);lo.append(m['T'][r]**2/4*expit(self.lo[idx]+1e-4));hi.append(m['T'][r]**2/4*expit(self.upper[idx]-1e-4))
                else:
                    vals.append(-d if self.kind[r]==1 else d-m['T'][r]**2/4)
                    lo.append(np.exp(self.lo[idx]+1e-4));hi.append(np.exp(self.upper[idx]-1e-4))
            vals=np.asarray(vals);lo=np.asarray(lo);hi=np.asarray(hi)
            q[self.n0]+=self.rest[1]-self.basis(self.rest[0],0.,m)@m['N']
            heights=self.event_heights(q)
            return np.r_[vals-lo,hi-vals,heights-1e-4,1.-1e-4-heights]
        result=minimize(lambda y:float(np.sum((y-original)**2)),initial,method='SLSQP',bounds=bounds,
                        constraints=[dict(type='ineq',fun=margins)],options=dict(maxiter=1500,ftol=1e-12))
        try:
            final=self.constrain_events(trial(result.x))
            if not self.physically_valid_events(final):raise ValueError('Nonphysical fixed event heights')
        except ValueError as exc:
            raise ValueError(f'Cannot initialize {len(self.event_I)} exact events: {result.message}; {exc}; min margin={np.min(margins(result.x)):.4g}') from exc
        print(f'FM {len(self.event_I)}-event initialization: distance={np.linalg.norm(result.x-original):.6g} heights={self.event_heights(final).tolist()} status={result.message}',flush=True)
        return final
