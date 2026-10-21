"""Physical peak morphology constraints; independent of absolute peak timing."""
import numpy as np
from .rest_protocols import peak_summary


def peak_shape_errors(t,x,target_t,target_x,onset=20.):
    truth=peak_summary(target_t,target_x,onset);candidate=peak_summary(t,x,onset)
    n,m=len(truth['times']),len(candidate['times']);paired=min(n,m)
    height=np.full(n,1.);width=np.full(n,1.)
    height[:paired]=candidate['heights'][:paired]/truth['heights'][:paired]-1.
    width[:paired]=candidate['widths'][:paired]/truth['widths'][:paired]-1.
    return truth,candidate,height,width,n==m


def peak_shape_residual(t,x,target_t,target_x,onset=20.,tolerance=.1):
    """Continuous shape errors plus strong penalties beyond fixed tolerances.

    Peak-centered waveform samples describe morphology only. They supplement,
    rather than replace, the input-locked timing and raw waveform criteria.
    Missing peaks keep the residual length fixed and cannot satisfy the gate.
    """
    if not 0<tolerance<1:raise ValueError('Shape tolerance must lie in (0,1)')
    truth,candidate,height,width,_=peak_shape_errors(t,x,target_t,target_x,onset)
    offsets=np.linspace(-4.,7.,45);n=len(height);paired=min(n,len(candidate['times']))
    waveform=np.full((n,len(offsets)),25.)
    for j in range(paired):
        a=truth['times'][j]+onset+offsets;b=candidate['times'][j]+onset+offsets
        # Fixed reference window; out-of-range candidate samples are penalized.
        valid=(a>=target_t[0])&(a<=target_t[-1])
        good=valid&(b>=t[0])&(b<=t[-1])
        waveform[j,~valid]=0.
        waveform[j,good]=(np.interp(b[good],t,x[:,0])-np.interp(a[good],target_t,target_x[:,0]))/.03
    excess_h=np.maximum(abs(height)-tolerance,0.)/.01
    excess_w=np.maximum(abs(width)-tolerance,0.)/.01
    return np.r_[height/.03,width/.03,10*excess_h,10*excess_w,waveform.ravel()/np.sqrt(len(offsets))]


def peak_shape_check(t,x,target_t,target_x,onset=20.,tolerance=.1):
    truth,candidate,height,width,counts_match=peak_shape_errors(t,x,target_t,target_x,onset)
    mh=float(np.max(abs(height))) if len(height) else 0.
    mw=float(np.max(abs(width))) if len(width) else 0.
    return dict(reference_peaks=len(truth['times']),candidate_peaks=len(candidate['times']),
        counts_match=bool(counts_match),max_height_relative_error=mh,max_width_relative_error=mw,
        tolerance=tolerance,passes=bool(counts_match and mh<=tolerance and mw<=tolerance))


def autonomous_shape_summary(t,x):
    """Physical cycle morphology in the last 400 s, excluding truncated peaks."""
    from scipy.signal import find_peaks
    t=np.asarray(t);c=np.asarray(x)[:,0];tail=t>=t[-1]-400.
    indices=find_peaks(c,prominence=.02)[0]
    indices=indices[(t[indices]>=t[-1]-400.)&(t[indices]<=t[-1]-10.)]
    period=float(np.median(np.diff(t[indices]))) if len(indices)>=4 else None
    from scipy.signal import peak_widths
    widths=None
    if len(indices):
        _,_,left,right=peak_widths(c,indices,rel_height=.5)
        widths=float(np.median(np.interp(right,np.arange(len(t)),t)-np.interp(left,np.arange(len(t)),t)))
    return dict(amplitude=float(np.ptp(c[tail])),height=float(np.max(c[tail])),
                period=period,width=widths,C_rest=float(np.mean(c[tail])))


def autonomous_shape_check(t,x,target_t,target_x,cycling,tolerance=.1,period_tolerance=.2):
    truth=autonomous_shape_summary(target_t,target_x);candidate=autonomous_shape_summary(t,x)
    if cycling:
        errors=[]
        for key in ['amplitude','period','width']:
            a,b=truth[key],candidate[key]
            errors.append(float(b/a-1.) if a is not None and b is not None and a>0 else 1.)
        tolerances=[tolerance,period_tolerance,tolerance]
    else:
        errors=[float(candidate['C_rest']/truth['C_rest']-1.)];tolerances=[tolerance]
    return dict(reference=truth,candidate=candidate,relative_errors=errors,
                passes=bool(all(abs(e)<=q for e,q in zip(errors,tolerances))),tolerances=tolerances)
