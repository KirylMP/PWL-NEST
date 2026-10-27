#!/usr/bin/env python3
"""Experimental analytic boundary-matching solve for an FM cycle fold.

This is a standalone candidate transformation, not a constraint inside the
main optimizer. It assumes the transverse itinerary R2,R3,R4,R3,R2,R1 and
checks that the candidate actually follows it. Nothing is written to the
input candidate file.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import least_squares

import fit_four as f
from counts import cycles


REGIONS=(1,2,3,2,1,0)  # zero-based indices: R2,R3,R4,R3,R2,R1
BOUNDARIES=(0,1,2,2,1,0,0)


def region_augmented(m,region,I):
    co=m['co']
    A=np.column_stack([co[:,1]+np.sum(co[:,4:4+region],axis=1),co[:,2]])
    b=co[:,0]+co[:,3]*I-co[:,4:4+region]@m['th'][:region]
    aug=np.zeros((3,3));aug[:2,:2]=A;aug[:2,2]=b
    return aug


def matching(times,I,x,details=False):
    """Six height matching residuals after exact elimination of entry heights."""
    m=f.unpack(x);z=m['th'][list(BOUNDARIES)];heights=[];exits=[];matrices=[]
    for j,region in enumerate(REGIONS):
        M=region_augmented(m,region,I);Q=expm(M*times[j]);E=Q[:2,:2];g=Q[:2,2]
        if not np.isfinite(E).all() or abs(E[0,1])<1e-12:
            raise ValueError('A segment has a singular E12; use a different parameterization')
        h=(z[j+1]-E[0,0]*z[j]-g[0])/E[0,1]
        heights.append(float(h));exits.append(float(E[1,0]*z[j]+E[1,1]*h+g[1]));matrices.append(M)
    residual=np.asarray(exits)-np.roll(heights,-1)
    return (residual,m,z,np.asarray(heights),matrices) if details else residual


def starting_times(x,probe_I):
    m=f.unpack(x);bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
    found,diag=cycles(probe_I,f.equilibrium(probe_I,m),m['co'],m['padded'],bounds,n=256,tol=2e-10)
    stable=[q for q in found if q['stable'] and q['min']<m['th'][0] and q['max']>m['th'][2]]
    if not stable:raise ValueError(f'No stable cycle crossing all 3 thresholds at I={probe_I}; diagnostics={diag}')
    orbit=max(stable,key=lambda q:q['max']-q['min'])
    def rhs(t,y):return m['co']@np.r_[1,y[0],y[1],probe_I,np.maximum(y[0]-m['padded'],0)]
    events=[]
    for cut in m['th']:
        def ev(t,y,cut=cut):return y[0]-cut
        ev.direction=0;ev.terminal=False;events.append(ev)
    sol=solve_ivp(rhs,(0,2.2*orbit['T']),[orbit['section'],orbit['s']],method='DOP853',
                  rtol=3e-11,atol=1e-13,max_step=.04,events=events)
    hits=sorted([(float(t),i,y) for i,(tt,yy) in enumerate(zip(sol.t_events,sol.y_events))
                 for t,y in zip(tt,yy)],key=lambda x:x[0])
    upward=[j for j,(t,i,y) in enumerate(hits) if i==0 and rhs(t,y)[0]>0]
    for a,b in zip(upward[:-1],upward[1:]):
        seq=hits[a:b+1]
        if tuple(q[1] for q in seq)==BOUNDARIES:
            return np.diff([q[0] for q in seq]),diag
    raise ValueError(f'Cycle at I={probe_I} does not follow itinerary {BOUNDARIES}')


def verify_trajectory(I,x,times):
    err,m,z,h,matrices=matching(times,I,x,True);samples=[];normal=[]
    for j,(region,M) in enumerate(zip(REGIONS,matrices)):
        Y=np.array([(expm(M*t)@np.r_[z[j],h[j],1.])[:2] for t in np.linspace(0,times[j],401)])
        lo=-np.inf if region==0 else m['th'][region-1]
        hi=np.inf if region==3 else m['th'][region]
        violation=max(float(np.max(lo-Y[:,0])),float(np.max(Y[:,0]-hi)),0.)
        if violation>1e-7:raise ValueError(f'Leg {j+1} exits its assumed region by {violation}')
        samples.extend(Y);normal.append(float((M@np.r_[z[j],h[j],1.])[0]))
    samples=np.asarray(samples)
    if np.min(samples[:,1])<=0 or np.max(samples[:,1])>=1:raise ValueError('Orbit leaves physical h interval')
    z_bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
    if np.min(samples[:,0])<=z_bounds[0] or np.max(samples[:,0])>=z_bounds[1]:raise ValueError('Orbit leaves physical C interval')
    if min(abs(v) for v in normal)<1e-8:raise ValueError('Boundary crossing is nontransverse')
    if not f.physically_valid_events(x):raise ValueError('Fixed equilibrium crossings leave physical h interval')
    co=m['co'];y0=np.array([z[0],h[0],0.]);eps=1e-7
    def rhs(t,y):
        flow=co@np.r_[1,y[0],y[1],I,np.maximum(y[0]-m['padded'],0)]
        trace=co[0,1]+co[1,2]+sum(co[0,4+j] for j in range(3) if y[0]>m['th'][j])
        return np.r_[flow,trace]
    initial=solve_ivp(rhs,(0,eps),y0,method='DOP853',rtol=3e-12,atol=1e-14).y[:,-1]
    def crossing(t,y):return y[0]-z[0]
    crossing.direction=1;crossing.terminal=True
    sol=solve_ivp(rhs,(eps,1.2*sum(times)),initial,method='DOP853',rtol=3e-12,atol=1e-14,
                  max_step=.01,events=crossing)
    if not len(sol.t_events[0]):raise ValueError('Independent integrator did not return to the section')
    end=sol.y_events[0][0];period=float(sol.t_events[0][0]);closure=float(end[1]-h[0]);log_mu=float(end[2])
    if abs(closure)>1e-7 or abs(period-sum(times))>1e-5 or abs(log_mu)>1e-4:
        raise ValueError(f'Independent check failed: closure={closure}, period={period}, log_multiplier={log_mu}')
    return dict(max_matching_residual=float(max(abs(err))),section_C=float(f.TC[0]),section_h=float(h[0]),
        C_min=float(f.inverse(min(samples[:,0]),m)),C_max=float(f.inverse(max(samples[:,0]),m)),
        h_min=float(min(samples[:,1])),h_max=float(max(samples[:,1])),normal_speeds=normal,
        independent_period=period,independent_closure_h=closure,independent_log_multiplier=log_mu,
        caveat='Region and physical bounds checked by finite sampling; not a global proof.')


def local_nondegeneracy(times,I,x,index):
    base=matching(times,I,x);eps=1e-5;eye=np.eye(6)
    J=np.column_stack([(matching(times+eps*e,I,x)-matching(times-eps*e,I,x))/(2*eps) for e in eye])
    U,s,V=np.linalg.svd(J);left=U[:,-1];right=V[-1,:]
    dI=(matching(times,I+1e-6,x)-matching(times,I-1e-6,x))/(2e-6)
    xp=x.copy();xm=x.copy();xp[index]+=1e-6;xm[index]-=1e-6
    dx=(matching(times,I,f.constrain_events(xp))-matching(times,I,f.constrain_events(xm)))/(2e-6)
    curv=[]
    for h in [1e-3,3e-4,1e-4]:
        second=(matching(times+h*right,I,x)-2*base+matching(times-h*right,I,x))/(h*h)
        curv.append(float(left@second))
    return dict(singular_values=s.tolist(),input_derivative=float(left@dI),
        parameter_derivative=float(left@dx),curvature_samples=curv,
        roots_locally_on_lower_I=bool((left@dI)*curv[1]>0),
        caveat='Finite-difference local diagnostic; refine and continue branches for stronger evidence.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,default=f.P/'candidate.json')
    p.add_argument('--output',type=Path,default=f.P/'fold_fixed_candidate.json')
    p.add_argument('--target-I',type=float,default=1.1231503495944923)
    p.add_argument('--probe-I',type=float,default=1.1)
    p.add_argument('--parameter-index',type=int,default=4)
    p.add_argument('--max-evaluations',type=int,default=500)
    a=p.parse_args()
    if not np.isfinite(a.target_I) or a.target_I<0 or not np.isfinite(a.probe_I) or a.probe_I<0:
        p.error('target-I and probe-I must be finite and nonnegative')
    if a.max_evaluations<1:p.error('max-evaluations must be positive')
    candidate_path=a.candidate.resolve();output_path=a.output.resolve()
    if output_path==candidate_path:p.error('--output must differ from input candidate')
    if output_path.exists():p.error('--output already exists; choose a new filename to preserve earlier results')
    if a.parameter_index not in f.ACTIVE or a.parameter_index in (10,12):
        p.error('parameter must be active and cannot be a common h-translation parameter (10 or 12)')
    old=json.loads(candidate_path.read_text());seed=f.constrain_events(old['parameters'])
    times0,diag=starting_times(seed,a.probe_I);index=a.parameter_index
    def objective(v):
        t=v[:6];x=seed.copy();x[index]=v[6];x=f.constrain_events(x);m=f.unpack(x)
        try:g=matching(t,a.target_I,x)
        except (ValueError,FloatingPointError):return np.full(7,1e6)
        trace=sum(m['T'][r]*dt for r,dt in zip(REGIONS,t))
        return np.r_[g,trace/10]
    start=np.r_[times0,seed[index]];lo=np.r_[np.full(6,1e-5),f.LO[index]+1e-8];hi=np.r_[np.full(6,30.),f.HI[index]-1e-8]
    if np.any(start<=lo) or np.any(start>=hi):raise ValueError('Initial values violate numerical search bounds')
    result=least_squares(objective,start,bounds=(lo,hi),max_nfev=a.max_evaluations,
                         ftol=1e-12,xtol=1e-12,gtol=1e-12,x_scale='jac')
    residual=objective(result.x);error=float(max(abs(residual)))
    print(f'optimizer success={result.success} nfev={result.nfev} max_residual={error:.3g}',flush=True)
    if not np.isfinite(error) or error>1e-8:raise ValueError('No accurate fold solution found; candidate was not written')
    x=seed.copy();x[index]=result.x[6];x=f.constrain_events(x);times=result.x[:6]
    verification=verify_trajectory(a.target_I,x,times)
    diagnostic=local_nondegeneracy(times,a.target_I,x,index)
    ss=diagnostic['singular_values'];curv=diagnostic['curvature_samples']
    if (ss[-1]>1e-6*ss[0] or abs(diagnostic['input_derivative'])<1e-8 or
        abs(diagnostic['parameter_derivative'])<1e-8 or min(abs(q) for q in curv)<1e-9 or
        len({np.sign(q) for q in curv})!=1 or not diagnostic['roots_locally_on_lower_I']):
        raise ValueError(f'Local fold/nondegeneracy check failed: {diagnostic}')
    fold=dict(target_I=a.target_I,source=str(candidate_path),probe_I=a.probe_I,probe_diagnostics=diag,
        parameter_index=index,old_parameter=float(seed[index]),new_parameter=float(x[index]),
        crossing_times=times.tolist(),period=float(sum(times)),verification=verification,
        local_nondegeneracy=diagnostic,optimizer=dict(success=bool(result.success),
        message=result.message,nfev=result.nfev,max_residual=error),
        note='Experimental one-candidate fold solve; the main optimizer does not enforce this condition.')
    f.write(x,output_path,dict(fold_condition_experiment=fold))
    print(json.dumps(fold,indent=2),flush=True)
    print(f'Candidate saved separately to {output_path}',flush=True)


if __name__=='__main__':main()
