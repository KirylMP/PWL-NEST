"""Numerical elimination of one FM parameter using the large-cycle fold.

The projection follows a single, transverse six-segment boundary itinerary.
It returns no candidate on failure; every successful result is checked against
the boundary equations, physical domain, crossing order and local fold tests.
"""
import os
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

import fit_four as f
from boundary_fold_experiment import (REGIONS, matching, starting_times,
                                      local_nondegeneracy, verify_trajectory)


# Use one dependent parameter for the entire fit. Try regional parameters first,
# then the relative kinetic scales log(a), log(b). A common scaling of a and b
# only changes time, but changing either one alone can move the cycle fold.
FOLD_PARAMETER_CANDIDATES=(4,5,3,2,6,7,9,8,0,1,14,16,17,15)
DEFAULT_TARGET_I=1.1231503495944923  # numerical LR estimate, not an exact constant


class FoldProjectionError(ValueError):
    pass


class FoldProjector:
    def __init__(self,seed,target_I=DEFAULT_TARGET_I,probe_I=1.1,max_nfev=120):
        if not np.isfinite(target_I) or target_I<0 or not np.isfinite(probe_I) or probe_I<0:
            raise FoldProjectionError('Fold target and probe I must be finite and nonnegative')
        if max_nfev<1:raise FoldProjectionError('Fold solver budget must be positive')
        self.target_I=float(target_I);self.probe_I=float(probe_I);self.max_nfev=int(max_nfev)
        self.attempts=0;self.failures=0;self.seconds=0.;self.last_error=None
        seed=f.constrain_events(seed)
        errors=[];times_cache={};probes=list(dict.fromkeys([self.probe_I,1.05,1.0,.9,.8,.65,.55]))
        for index in FOLD_PARAMETER_CANDIDATES:
            if index not in f.ACTIVE:continue
            self.parameter_index=index
            for I in probes:
                if I not in times_cache:
                    print(f'FOLD INIT: searching for a large orbit at I={I}',flush=True)
                    try:times_cache[I]=starting_times(seed,I)[0]
                    except (ValueError,StopIteration) as e:
                        times_cache[I]=None;errors.append(f'probe I={I}: {e}')
                        print(f'FOLD INIT: no usable orbit at I={I}: {e}',flush=True)
                times0=times_cache[I]
                if times0 is None:continue
                print(f'FOLD INIT: trying parameter[{index}] from I={I}',flush=True)
                try:
                    x,metadata=self._solve(seed,np.r_[times0,seed[index]])
                    metadata['independent_check']=verify_trajectory(self.target_I,x,np.asarray(metadata['crossing_times']))
                except (FoldProjectionError,ValueError,FloatingPointError) as e:
                    errors.append(f'parameter[{index}], probe I={I}: {e}')
                    print(f'FOLD INIT: rejected parameter[{index}] at I={I}: {e}',flush=True)
                    continue
                self.used_probe_I=float(I);self.active=np.array([j for j in f.ACTIVE if j!=index],dtype=int)
                self.anchor=np.r_[metadata['crossing_times'],x[index]]
                self.initial=x;self.initial_metadata=metadata
                print(f'FOLD INIT: selected parameter[{index}], probe I={I}',flush=True)
                return
        # The seed can be too far from the requested fold for any one
        # parameter to resolve the boundary equations. Jointly adjust the
        # active coefficients to reach the fold, then freeze all but one
        # regular parameter for the main closed-orbit fit.
        nearby=sorted(times_cache.items(),key=lambda q:abs(q[0]-self.target_I))[:3]
        for I,times0 in nearby:
            if times0 is None:continue
            print(f'FOLD INIT: locating the seed model fold from I={I}',flush=True)
            try:
                fold_x,fold_I,fold_times,free_meta=self._solve_free_input(seed,times0,I)
            except FoldProjectionError as e:
                errors.append(f'free-I fold, probe I={I}: {e}')
                print(f'FOLD INIT: free-I fold rejected from I={I}: {e}',flush=True)
                continue
            print(f'FOLD INIT: seed-model fold at I={fold_I:.10g}',flush=True)
            ranked=[]
            for index in FOLD_PARAMETER_CANDIDATES:
                if index not in f.ACTIVE:continue
                try:d=local_nondegeneracy(fold_times,fold_I,fold_x,index)
                except (ValueError,FloatingPointError,np.linalg.LinAlgError):continue
                if (d['roots_locally_on_lower_I'] and abs(d['input_derivative'])>1e-8
                        and abs(d['parameter_derivative'])>1e-8):
                    sensitivity=abs(d['parameter_derivative']/d['input_derivative'])
                    ranked.append((-sensitivity,index))
            for _,index in sorted(ranked):
                print(f'FOLD INIT: continuing seed fold with parameter[{index}]',flush=True)
                try:
                    x,metadata=self._continue_fold_to_target(fold_x,fold_times,fold_I,index)
                    metadata['independent_check']=verify_trajectory(self.target_I,x,np.asarray(metadata['crossing_times']))
                except (FoldProjectionError,ValueError,FloatingPointError) as e:
                    errors.append(f'fold continuation parameter[{index}], seed I={fold_I:.8g}: {e}')
                    print(f'FOLD INIT: continuation rejected for parameter[{index}]: {e}',flush=True)
                    continue
                self.used_probe_I=float(I);self.parameter_index=index
                self.active=np.array([j for j in f.ACTIVE if j!=index],dtype=int)
                self.anchor=np.r_[metadata['crossing_times'],x[index]]
                self.initial=x;self.initial_metadata=metadata
                print(f'FOLD INIT: selected continued fold, parameter[{index}], I={self.target_I:.10g}',flush=True)
                return
            if not ranked:errors.append(f'free-I fold at probe I={I} had no regular active continuation parameter')

        # If continuation from the fitted model's own fold is unavailable, try
        # the joint projection as a fallback.
        for I,times0 in nearby:
            if times0 is None:continue
            print(f'FOLD INIT: joint parameter projection from I={I}',flush=True)
            try:joint_x,joint_times=self._solve_joint(seed,times0)
            except FoldProjectionError as e:
                errors.append(f'joint projection, probe I={I}: {e}')
                print(f'FOLD INIT: joint projection rejected at I={I}: {e}',flush=True)
                continue
            for index in FOLD_PARAMETER_CANDIDATES:
                if index not in f.ACTIVE:continue
                self.parameter_index=index
                try:
                    x,metadata=self._solve(joint_x,np.r_[joint_times,joint_x[index]])
                    metadata['independent_check']=verify_trajectory(self.target_I,x,np.asarray(metadata['crossing_times']))
                except (FoldProjectionError,ValueError,FloatingPointError) as e:
                    errors.append(f'joint seed parameter[{index}], probe I={I}: {e}')
                    continue
                self.used_probe_I=float(I);self.parameter_index=index
                self.active=np.array([j for j in f.ACTIVE if j!=index],dtype=int)
                self.anchor=np.r_[metadata['crossing_times'],x[index]]
                self.initial=x;self.initial_metadata=metadata
                print(f'FOLD INIT: selected joint projection, parameter[{index}], probe I={I}',flush=True)
                return
        raise FoldProjectionError('Could not initialize a verified fold using the tested parameters and probes: '+'; '.join(errors))

    def _solve_joint(self,seed,times0):
        """Project the fitted model onto the fold equations with minimal drift."""
        base=f.constrain_events(seed);active=np.asarray(f.ACTIVE,dtype=int)
        scales=np.array([.2,.15,.2,.25,.2,.2,.6,.35,.6,.35,.02,.02,.03,.03,.3,.3,.15,.25])[active]
        lo=np.r_[np.full(6,1e-5),f.LO[active]+1e-8]
        hi=np.r_[np.full(6,30.),f.HI[active]-1e-8]
        start=np.clip(np.r_[times0,base[active]],lo+1e-9,hi-1e-9)

        def unpack(v):
            x=base.copy();x[active]=v[6:]
            return f.constrain_events(x)

        best=None
        for regularizer in (1e-4,1e-6,1e-8):
            def objective(v):
                try:
                    x=unpack(v);m=f.unpack(x);g=matching(v[:6],self.target_I,x)
                    trace=sum(m['T'][r]*dt for r,dt in zip(REGIONS,v[:6]))
                    out=np.r_[g,trace/10.,regularizer*(v[6:]-base[active])/scales]
                    if not np.all(np.isfinite(out)):return np.full(7+len(active),1e6)
                    return out
                except (ValueError,FloatingPointError,OverflowError):
                    return np.full(7+len(active),1e6)

            sol=least_squares(objective,start,bounds=(lo,hi),max_nfev=max(self.max_nfev,300),
                ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac')
            start=sol.x;x=unpack(sol.x);res=objective(sol.x)
            match_error=float(np.max(np.abs(res[:6])));trace_error=float(abs(res[6]*10.))
            best=(match_error,trace_error,int(sol.nfev),regularizer,sol.x.copy())
            print(f'FOLD INIT: joint regularizer={regularizer:g} matching={match_error:.3g} trace={trace_error:.3g} nfev={sol.nfev}',flush=True)
            if match_error<=1e-8 and trace_error<=1e-7:
                if not f.physically_valid_events(x):
                    raise FoldProjectionError('Joint fold projection found nonphysical fixed-event heights')
                return x,np.asarray(sol.x[:6])
        assert best is not None
        # The proximity penalty can leave a small but unacceptable closure
        # residual. Starting at that near-fold point, solve only the seven
        # fold equations so proximity no longer trades away exact closure.
        start=best[4]
        def fold_equations(v):
            try:
                x=unpack(v);m=f.unpack(x);g=matching(v[:6],self.target_I,x)
                trace=sum(m['T'][r]*dt for r,dt in zip(REGIONS,v[:6]))
                out=np.r_[g,trace/10.]
                if not np.all(np.isfinite(out)):return np.full(7,1e6)
                return out
            except (ValueError,FloatingPointError,OverflowError):
                return np.full(7,1e6)
        sol=least_squares(fold_equations,start,bounds=(lo,hi),
            max_nfev=max(5*self.max_nfev,500),ftol=1e-13,xtol=1e-13,
            gtol=1e-13,x_scale='jac')
        x=unpack(sol.x);res=fold_equations(sol.x)
        match_error=float(np.max(np.abs(res[:6])));trace_error=float(abs(res[6]*10.))
        print(f'FOLD INIT: unregularized correction matching={match_error:.3g} trace={trace_error:.3g} nfev={sol.nfev}',flush=True)
        if match_error<=1e-8 and trace_error<=1e-7:
            if not f.physically_valid_events(x):
                raise FoldProjectionError('Unregularized fold correction found nonphysical fixed-event heights')
            return x,np.asarray(sol.x[:6])
        raise FoldProjectionError(f'Joint fold equations not resolved after unregularized correction (matching={match_error:.3g}, trace={trace_error:.3g}, nfev={sol.nfev}; regularized start matching={best[0]:.3g}, trace={best[1]:.3g}, regularizer={best[3]:g})')

    def _solve_free_input(self,seed,times0,probe_I):
        """Find this parameter set's own cycle fold with input I as an unknown."""
        base=f.constrain_events(seed)
        lo=np.r_[np.full(6,1e-5),1e-6]
        hi=np.r_[np.full(6,30.),2.]
        start=np.clip(np.r_[times0,probe_I],lo+1e-9,hi-1e-9)

        def objective(v):
            try:
                m=f.unpack(base);g=matching(v[:6],v[6],base)
                trace=sum(m['T'][r]*dt for r,dt in zip(REGIONS,v[:6]))
                out=np.r_[g,trace/10.]
                if not np.all(np.isfinite(out)):return np.full(7,1e6)
                return out
            except (ValueError,FloatingPointError,OverflowError):
                return np.full(7,1e6)

        sol=least_squares(objective,start,bounds=(lo,hi),max_nfev=max(5*self.max_nfev,500),
            ftol=1e-12,xtol=1e-12,gtol=1e-12,x_scale='jac')
        res=objective(sol.x);match_error=float(np.max(np.abs(res[:6])))
        trace_error=float(abs(res[6]*10.));I=float(sol.x[6]);times=np.asarray(sol.x[:6])
        print(f'FOLD INIT: free-I solve I={I:.10g} matching={match_error:.3g} trace={trace_error:.3g} nfev={sol.nfev}',flush=True)
        if match_error>1e-8 or trace_error>1e-7:
            raise FoldProjectionError(f'Free-I fold equations not resolved (matching={match_error:.3g}, trace={trace_error:.3g}, nfev={sol.nfev})')
        if not f.physically_valid_events(base):raise FoldProjectionError('Free-I fold has nonphysical fixed-event heights')
        self._check_itinerary(base,times,I)
        meta=dict(input_I=I,crossing_times=times.tolist(),matching_residual=match_error,
                  trace_integral=trace_error,solver_nfev=int(sol.nfev))
        return base,I,times,meta

    def _continue_fold_to_target(self,seed,times0,start_I,index):
        """Continue a verified fold in I while varying one dependent parameter."""
        self.parameter_index=index
        x=f.constrain_events(seed);times=np.asarray(times0,dtype=float)
        I=float(start_I)
        # Validate that this parameter is a regular coordinate at the seed fold.
        x,meta=self._solve(x,np.r_[times,x[index]],target_I=I)
        delta=self.target_I-I
        if abs(delta)<1e-10:return x,meta
        direction=float(np.sign(delta));step=min(.025,abs(delta));fraction=0.
        while fraction<1.-1e-12:
            next_I=I+direction*min(step,abs(self.target_I-I))
            try:
                x_new,meta_new=self._solve(x,np.r_[times,x[index]],target_I=next_I,
                                           check_nondegeneracy=abs(next_I-self.target_I)<1e-12)
                times=np.asarray(meta_new['crossing_times']);x=x_new;I=next_I
                fraction=(I-start_I)/delta
                step=min(step*1.5,.05)
                print(f'FOLD CONTINUE: I={I:.9g} parameter[{index}]={x[index]:.9g}',flush=True)
            except FoldProjectionError:
                step*=.5
                if step<1e-4:raise FoldProjectionError(f'Fold continuation stalled at I={I:.9g}')
        return x,meta

    def _solve(self,raw,initial,target_I=None,check_nondegeneracy=True):
        started=time.monotonic();self.attempts+=1
        try:
            base=f.constrain_events(raw);index=self.parameter_index
            target=self.target_I if target_I is None else float(target_I)
            lo=np.r_[np.full(6,1e-5),f.LO[index]+1e-8]
            hi=np.r_[np.full(6,30.),f.HI[index]-1e-8]
            start=np.clip(np.asarray(initial,dtype=float),lo+1e-9,hi-1e-9)
            def reconstruct(v):
                x=base.copy();x[index]=v[6];return f.constrain_events(x)
            def objective(v):
                try:
                    x=reconstruct(v);m=f.unpack(x);g=matching(v[:6],target,x)
                    trace=sum(m['T'][r]*dt for r,dt in zip(REGIONS,v[:6]))
                    out=np.r_[g,trace/10.]
                    if not np.all(np.isfinite(out)):return np.full(7,1e6)
                    return out
                except (ValueError,FloatingPointError,OverflowError):
                    return np.full(7,1e6)
            sol=least_squares(objective,start,bounds=(lo,hi),max_nfev=self.max_nfev,
                              ftol=1e-11,xtol=1e-11,gtol=1e-11,x_scale='jac')
            x=reconstruct(sol.x);times=sol.x[:6];res=objective(sol.x)
            match_error=float(np.max(np.abs(res[:6])));trace_error=float(abs(res[6]*10.))
            if match_error>1e-8 or trace_error>1e-7:
                raise FoldProjectionError(f'Fold equations not resolved (matching={match_error:.3g}, trace={trace_error:.3g}, nfev={sol.nfev})')
            if min(sol.x[6]-lo[6],hi[6]-sol.x[6])<1e-7:
                raise FoldProjectionError('Dependent fold parameter hit its search bound')
            if not f.physically_valid_events(x):raise FoldProjectionError('Fixed equilibrium crossings outside physical h interval')
            self._check_itinerary(x,times,target)
            diagnostic=None
            if check_nondegeneracy:
                diagnostic=local_nondegeneracy(times,target,x,index)
                ss=diagnostic['singular_values'];curv=diagnostic['curvature_samples']
                if (not np.all(np.isfinite(ss)) or not np.all(np.isfinite(curv)) or
                    not np.isfinite(diagnostic['input_derivative']) or not np.isfinite(diagnostic['parameter_derivative']) or
                    ss[0]<=0 or ss[-1]>1e-6*ss[0] or ss[-2]<1e-8*ss[0] or abs(diagnostic['input_derivative'])<1e-8 or
                    abs(diagnostic['parameter_derivative'])<1e-8 or min(abs(v) for v in curv)<1e-9 or
                    len({np.sign(v) for v in curv})!=1 or not diagnostic['roots_locally_on_lower_I']):
                    raise FoldProjectionError('Fold nondegeneracy or upper-bound orientation unresolved')
            meta=dict(target_I=target,parameter_index=index,parameter_value=float(x[index]),
                crossing_times=times.tolist(),matching_residual=match_error,trace_integral=trace_error,
                solver_nfev=int(sol.nfev),solver_success=bool(sol.success),
                local_nondegeneracy=diagnostic,
                note='Numerical constrained fold on assumed itinerary; finite sampling is not a global proof.')
            return x,meta
        except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:
            self.failures+=1;self.last_error=f'{type(e).__name__}: {e}'
            if isinstance(e,FoldProjectionError):raise
            raise FoldProjectionError(self.last_error) from e
        finally:self.seconds+=time.monotonic()-started

    def _check_itinerary(self,x,times,I=None):
        I=self.target_I if I is None else float(I)
        _,m,z,h,matrices=matching(times,I,x,True)
        lower,upper=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
        expected=(1,1,1,-1,-1,-1)
        for j,(r,M) in enumerate(zip(REGIONS,matrices)):
            y0=np.r_[z[j],h[j],1.];normal=float((M@y0)[0])
            if expected[j]*normal<=1e-8:raise FoldProjectionError(f'Leg {j+1} has wrong or grazing crossing')
            states=np.array([(expm(M*t)@y0)[:2] for t in np.linspace(0,times[j],65)])
            if not np.all(np.isfinite(states)):raise FoldProjectionError(f'Leg {j+1} contains nonfinite states')
            lo=-np.inf if r==0 else m['th'][r-1];hi=np.inf if r==3 else m['th'][r]
            if np.min(states[:,0])<lo-1e-7 or np.max(states[:,0])>hi+1e-7:
                raise FoldProjectionError(f'Leg {j+1} left its expected region')
            if np.min(states[:,0])<=lower or np.max(states[:,0])>=upper or np.min(states[:,1])<=0 or np.max(states[:,1])>=1:
                raise FoldProjectionError(f'Leg {j+1} left physical state domain')

    def project(self,raw):
        # A fixed anchor keeps the projection deterministic for repeated
        # optimizer inputs. Failed projections never fall back to an old x.
        return self._solve(raw,self.anchor)

    def independent_check(self,x,meta):
        return verify_trajectory(self.target_I,x,np.asarray(meta['crossing_times']))
