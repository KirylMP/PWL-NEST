"""Own-equilibrium starts and event features for reproducible coordinate grids."""
import numpy as np
from scipy.signal import find_peaks,peak_widths
from .strict_pwa import C_MAX
from .calcium_coordinates import NativeCoordinateDynamic


def stable_rest(model,baseline,target):
    effective=model.at_input(baseline);limits=np.r_[-np.inf,effective.latent.thresholds,np.inf];roots=[]
    for j,p in enumerate(effective.latent.local_coefficients()):
        try:
            uv=np.linalg.solve(p[:,1:3],-p[:,0]-baseline*p[:,3])
            if not limits[j]<=uv[0]<=limits[j+1] or np.max(np.linalg.eigvals(p[:,1:3]).real)>=-1e-10:continue
            x=model.to_physical(np.r_[uv,baseline])
            if 0<x[0]<C_MAX and 0<x[1]<1:roots.append(x)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError):continue
    if not roots:raise ValueError('No stable physical rest at baseline')
    selected=min(roots,key=lambda x:np.linalg.norm((x[:2]-np.asarray(target)[:2])/[.05,.1]))
    return selected,len(roots)


def native_events(model,initial,duration,tau,baseline,jumps,dt=.25,maxdt=.005,native=None):
    native=native or NativeCoordinateDynamic();state=np.array(initial,dtype=float);left=0.;times=[];paths=[]
    for right,delta in list(jumps)+[(duration,0.)]:
        count=int(np.ceil((right-left)/dt))+1
        if count<2:raise ValueError('Nonpositive input interval')
        y=native.run(model,state,right-left,tau,baseline,outputs=count,maxdt=maxdt)
        if y is None:raise ValueError('Native event trajectory failed')
        grid=np.linspace(left,right,count);ip3=baseline+(state[2]-baseline)*np.exp(-(grid-left)/tau)
        x=model.to_physical(np.column_stack([y,ip3]))
        times.append(grid if not paths else grid[1:]);paths.append(x if not paths else x[1:])
        state=x[-1].copy();state[2]+=delta;left=right
        if not 0<=state[2]<=1.3:raise ValueError('IP3 outside fitted domain')
    return np.concatenate(times),np.vstack(paths),{}


def peak_summary(t,x,onset=20.):
    peaks=find_peaks(x[:,0],prominence=.02)[0];peaks=peaks[t[peaks]>=onset]
    _,_,left,right=peak_widths(x[:,0],peaks,rel_height=.5)
    index=np.arange(len(t));width=np.interp(right,index,t)-np.interp(left,index,t)
    return dict(times=t[peaks]-onset,heights=x[peaks,0],widths=width)


def event_residual(t,x,target_t,target_x,onset=20.):
    truth=peak_summary(target_t,target_x,onset);candidate=peak_summary(t,x,onset)
    n,m=len(truth['times']),len(candidate['times']);paired=min(n,m)
    heights=np.full(n,25.);widths=np.full(n,25.);intervals=np.full(max(0,n-1),25.)
    heights[:paired]=(candidate['heights'][:paired]-truth['heights'][:paired])/.05
    widths[:paired]=(candidate['widths'][:paired]-truth['widths'][:paired])/.5
    if paired>1:intervals[:paired-1]=(np.diff(candidate['times'][:paired])-np.diff(truth['times'][:paired]))/2.
    latency=(candidate['times'][0]-truth['times'][0])/2. if paired else (25. if n else 0.)
    common=np.column_stack([np.interp(target_t,t,x[:,d]) for d in [0,1]])
    waveform=(.1*(common-target_x[:,:2])/[.03,.05]/np.sqrt(len(target_t))).ravel()
    return np.r_[heights,widths,intervals,latency,25.*(m-n),
        max(0.,float(x[:,0].max())-C_MAX)/.002,waveform]
