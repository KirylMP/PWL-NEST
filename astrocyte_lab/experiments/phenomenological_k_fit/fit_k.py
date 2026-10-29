#!/usr/bin/env python3
"""Two-threshold physical-C family with prescribed LR fold amplitude.

The fold is constrained by exact affine flow matching, integrated divergence,
and extrema conditions. Local shooting checks do not prove branch completeness.
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import root, brentq, least_squares, minimize
from scipy.linalg import expm
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'astrocyte_lab/scripts'))
from fit_physical_calcium import PhysicalModel
from fold import BoundaryFold

TARGETS=Path(__file__).resolve().parent/'lr_fold_targets.json'

def lr_rhs(C,h,I,K):
    q=1.049*(I+.13)/(I+.9434);m=I/(I+.13);n=C/(C+.08234)
    return np.array([(6*m**3*n**3*h**3+.11)*(2-1.185*C)-.9*C*C/(C*C+K*K),.2*(q*(1-h)-C*h)])

def lr_trace(C,h,I,K):
    q=1.049*(I+.13)/(I+.9434);m=I/(I+.13);n=C/(C+.08234)
    return 6*m**3*h**3*(3*n*n*.08234/(C+.08234)**2*(2-1.185*C)-1.185*n**3)-.13035-1.8*C*K*K/(C*C+K*K)**2-.2*(q+C)

def gate(C,I):
    q=1.049*(I+.13)/(I+.9434);return q/(q+C)

def hopf(K,start):
    def fun(z):
        c,I=z
        if c<=0 or I<=0:return [1e3,1e3]
        h=gate(c,I);return [lr_rhs(c,h,I,K)[0],lr_trace(c,h,I,K)]
    sol=root(fun,start,tol=1e-11)
    if not sol.success or np.max(np.abs(fun(sol.x)))>1e-8:raise ValueError(f'LR Hopf root unresolved at K={K}: {sol.message}')
    return sol.x

def setup(K,kind):
    model=PhysicalModel(kind,'identity')
    rest=brentq(lambda c:lr_rhs(c,gate(c,0),0,K)[0],1e-7,.3)
    lower=np.array([.05214116272147648,.5290700389163683]); upper=np.array([.3926251934101712,.8600205557570602])
    for k in np.linspace(.05,K,max(2,int(abs(K-.05)/.002)+1)):
        lower=hopf(k,lower);upper=hopf(k,upper)
    model.rest=np.array([rest,gate(rest,0)]);model.cuts=np.array([lower[0],upper[0]]);model.events=np.array([lower[1],upper[1]])
    if not 0<rest<lower[0]<upper[0] or not 0<lower[1]<upper[1]:raise ValueError('Unexpected event ordering')
    return model

def flow_state(M,z,h,t):return (expm(M*t)@np.array([z,h,1.]))[:2]

def run(K,kind,out,starts,maxiter,itinerary):
    out.mkdir(parents=True,exist_ok=True);t0=time.monotonic();model=setup(K,kind)
    targets=json.loads(TARGETS.read_text())['rows']; r=min(targets,key=lambda z:abs(z['K']-K))
    if abs(r['K']-K)>1e-8:raise ValueError('K must match a stored LR fold target; rerun LR fold continuation with that K')
    if r['nfev']>=500:raise ValueError('LR fold target did not converge; refusing to fit')
    first=0 if itinerary=='both' or r['C_min']<model.cuts[0] else 1
    fold=BoundaryFold(model,r['I'],first=first,last=1); n=fold.legs
    gap=float(model.cuts[1]-model.cuts[0])
    if first==0 and r['amplitude']<=gap:
        result=dict(K=K,kind=kind,LR_target=r,rest=model.rest.tolist(),cuts=model.cuts.tolist(),events=model.events.tolist(),itinerary=itinerary,itinerary_regions=list(fold.regions),status='amplitude_incompatible_with_both_thresholds',threshold_gap=gap,reason='A transverse cycle through both fixed C thresholds has amplitude strictly larger than their separation.')
        (out/'parameters.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('SAVED',out/'parameters.json',result['status'],flush=True);return
    seedcase='fm' if first==0 else ('fm' if K<.075 else 'am');seedpath=ROOT/f'astrocyte_lab/models/{seedcase}_2thr_traj_identity_20261029/parameters.json'
    seed=json.loads(seedpath.read_text());base=np.array(seed['parameters'],float);active=model.active;lo=model.lo[active]+1e-7;hi=model.hi_bounds[active]-1e-7
    alternate_path=ROOT/f'astrocyte_lab/models/{"am" if seedcase=="fm" else "fm"}_2thr_traj_identity_20261029/parameters.json'
    alternate=np.asarray(json.loads(alternate_path.read_text())['parameters'],float)
    Igrid=np.linspace(max(.05,model.events[0]*.8),min(1.5,r['I']*1.05),9);Cgrid=np.linspace(max(.01,model.rest[0]*.5),min(1.25,r['C_max']*1.1),34); C,I=np.meshgrid(Cgrid,Igrid)
    h=np.linspace(.2,.95,7);CC,II,HH=np.broadcast_arrays(C[...,None],I[...,None],h)
    ref=lr_rhs(CC,HH,II,K);ref=np.moveaxis(ref,0,-1)
    def geometry(x):
        m=model.unpack(x);B=model.basis(CC,II,m);q=HH
        pred=np.stack([m['a']*(q-B@m['N']),m['b']*(B@m['H']-q)],axis=-1)
        return np.mean(((pred-ref)/[.5,.08])**2)
    v0=np.r_[base[active],np.full(n,r['period']/n),.5,.5]
    if first==0 and seedcase=='fm' and seed.get('fold_constraint'):
        ts=seed['fold_constraint'].get('crossing_times',[])
        if len(ts)==n:v0[len(active):len(active)+n]=ts
    lower=np.r_[lo,np.full(n,.0001),.001,.001];upper=np.r_[hi,np.full(n,30.),.999,.999];v0=np.clip(v0,lower+1e-6,upper-1e-6)
    def expand(v):
        x=base.copy();x[active]=v[:len(active)];return model.constrain_events(x),v[len(active):len(active)+n],v[-2:]
    def eqs(v):
        try:
            x,t,fr=expand(v);res,m,z,h,M=fold.matching(x,t,True)
            imax=n//2-1;imin=n-1
            ymax=flow_state(M[imax],z[imax],h[imax],t[imax]*fr[0]);ymin=flow_state(M[imin],z[imin],h[imin],t[imin]*fr[1])
            speedmax=(M[imax]@np.r_[ymax,1])[0];speedmin=(M[imin]@np.r_[ymin,1])[0]
            return np.r_[res,sum(m['T'][idx]*dt for idx,dt in zip(fold.regions,t))/10,speedmax,speedmin,(ymax[0]-ymin[0]-r['amplitude'])]
        except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError):return np.full(n+4,1e3)
    def valid(v):
        x,_,_=expand(v);m=model.unpack(x);q=model.basis(model.cuts,model.events,m)@m['N'];return np.r_[q-1e-6,1-1e-6-q]
    rng=np.random.default_rng(7189);attempts=[];choices=[]
    for j in range(starts):
        start=v0.copy()
        if j%2:start[:len(active)]=alternate[active]
        if j>=2:start[:len(active)]+=rng.normal(0,.2,len(active));start[len(active):len(active)+n]*=np.exp(rng.normal(0,.08,n))
        start=np.clip(start,lower+1e-7,upper-1e-7)
        init=least_squares(lambda v:np.r_[eqs(v)*10, np.maximum(-valid(v),0)*10,1e-4*(v-start)],start,bounds=(lower,upper),max_nfev=maxiter,x_scale='jac',ftol=1e-9,xtol=1e-9,gtol=1e-9)
        print(f'K={K:.4f} {kind} start={j+1}/{starts} init residual={max(abs(eqs(init.x))):.4g} nfev={init.nfev} elapsed={time.monotonic()-t0:.1f}s',flush=True)
        sol=minimize(lambda v:geometry(expand(v)[0]),init.x,method='SLSQP',bounds=list(zip(lower,upper)),constraints=[dict(type='eq',fun=eqs),dict(type='ineq',fun=valid)],options=dict(maxiter=maxiter,ftol=1e-8))
        err=float(max(abs(eqs(sol.x))));x,t,fr=expand(sol.x);cost=float(geometry(x));attempts.append(dict(start=j+1,success=bool(sol.success),message=str(sol.message),nfev=int(sol.nfev),loss=cost,constraint_residual=err))
        print(f'K={K:.4f} {kind} start={j+1} loss={cost:.5g} constraint={err:.3g} success={sol.success}',flush=True)
        if err<1e-6 and min(valid(sol.x))>=-1e-7:choices.append((cost,sol.x))
    result=dict(K=K,kind=kind,LR_target=r,rest=model.rest.tolist(),cuts=model.cuts.tolist(),events=model.events.tolist(),itinerary=itinerary,itinerary_regions=list(fold.regions),attempts=attempts,scope='Exact rest/events by construction; constrained fold matching and amplitude. Global branch completeness not proved.',status='no_constraint_solution')
    if choices:
        _,v=min(choices,key=lambda z:z[0]);x,t,fr=expand(v);err,m,z,h,M=fold.matching(x,t,True);Cmin=10.;Cmax=-1.;admissible=True
        for j,(reg,mat) in enumerate(zip(fold.regions,M)):
            states=np.array([flow_state(mat,z[j],h[j],s) for s in np.linspace(0,t[j],101)])
            loC=0 if reg==0 else model.cuts[reg-1];hiC=model.CMAX if reg==2 else model.cuts[reg]
            direction=1 if j<n//2 else -1
            speed=(mat@np.r_[z[j],h[j],1.])[0]
            admissible &= bool(direction*speed>1e-8 and np.min(states[:,0])>=loC-1e-6 and np.max(states[:,0])<=hiC+1e-6 and np.min(states[:,1])>0 and np.max(states[:,1])<1)
            Cmin=min(Cmin,float(np.min(states[:,0])));Cmax=max(Cmax,float(np.max(states[:,0])))
        def rhs(ti,y):
            C,q=y;B=model.basis(C,r['I'],m);return [m['a']*(q-B@m['N']),m['b']*(B@m['H']-q)]
        integ=solve_ivp(rhs,(0,float(sum(t))),[z[0],h[0]],method='DOP853',rtol=1e-10,atol=1e-12,max_step=.01)
        close=float(np.linalg.norm(integ.y[:,-1]-[z[0],h[0]]));targetres=float(max(abs(eqs(v))))
        sampled_amplitude=Cmax-Cmin
        trajectory_ok=admissible and integ.success and close<1e-5 and abs(sampled_amplitude-r['amplitude'])<.002
        step=1e-5;eye=np.eye(n)
        J=np.column_stack([(fold.matching(x,t+step*e)-fold.matching(x,t-step*e))/(2*step) for e in eye])
        U,s,V=np.linalg.svd(J);left=U[:,-1];right=V[-1];base_res=fold.matching(x,t)
        curvatures=[float(left@(fold.matching(x,t+d*right)-2*base_res+fold.matching(x,t-d*right))/(d*d)) for d in (1e-3,3e-4,1e-4)]
        dI=1e-6;fold.I=r['I']+dI;plus=fold.matching(x,t);fold.I=r['I']-dI;minus=fold.matching(x,t);fold.I=r['I']
        unfold=float(left@(plus-minus)/(2*dI))
        fold_ok=bool(s[0]>0 and s[-1]<1e-5*s[0] and s[-2]>1e-8*s[0] and min(abs(c) for c in curvatures)>1e-9 and len({np.sign(c) for c in curvatures})==1 and abs(unfold)>1e-8 and curvatures[1]*unfold>0)
        status='verified_local' if trajectory_ok and fold_ok else ('fold_nondegeneracy_unresolved' if trajectory_ok else 'failed_trajectory_check')
        result.update(status=status,parameters=x.tolist(),a=m['a'],b=m['b'],alpha=m['N'].tolist(),beta=m['H'].tolist(),trace=m['T'].tolist(),determinant=m['D'].tolist(),loss=geometry(x),fold=dict(I=r['I'],target_amplitude=r['amplitude'],sampled_amplitude=sampled_amplitude,sampled_Cmin=Cmin,sampled_Cmax=Cmax,crossing_times=t.tolist(),closure=close,constraint_residual=targetres,admissible_segments=admissible,nondegeneracy_passed=fold_ok,singular_values=s.tolist(),curvature_samples=curvatures,input_unfolding_derivative=unfold))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        reference=solve_ivp(lambda ti,y:lr_rhs(y[0],y[1],r['I'],K),(0,r['period']),[r['C_max'],r['h_at_max']],method='DOP853',rtol=1e-10,atol=1e-12,max_step=.02)
        fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        ax[0].plot(reference.y[0],reference.y[1],label='LR');ax[0].plot(integ.y[0],integ.y[1],label='PWA')
        ax[0].set_xlabel('C (µM)');ax[0].set_ylabel('h / q');ax[0].legend()
        ax[1].plot(reference.t,reference.y[0],label='LR');ax[1].plot(integ.t,integ.y[0],label='PWA (different phase)');ax[1].set_xlabel('time (s)');ax[1].set_ylabel('C (µM)');ax[1].legend()
        for axis in ax:axis.grid(alpha=.25)
        fig.suptitle(f'K_ER={K:.4f}, {kind}, {result["status"]}: fold orbit');fig.savefig(out/'fold_orbit.png',dpi=150);plt.close(fig)
    (out/'parameters.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SAVED',out/'parameters.json',result['status'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--K',type=float,required=True);p.add_argument('--kind',choices=['am','fm'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--starts',type=int,default=4);p.add_argument('--maxiter',type=int,default=1200);p.add_argument('--itinerary',choices=['both','auto'],default='both');a=p.parse_args();run(a.K,a.kind,a.out,a.starts,a.maxiter,a.itinerary)
if __name__=='__main__':main()
