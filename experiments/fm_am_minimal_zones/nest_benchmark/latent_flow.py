"""AM/FM transformed-coordinate adapter for the existing native PWA kernel.

This is a standalone building block. It is not a NEST module.
Time unit: seconds; calcium and input: micromolar.
"""
import ctypes
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[3]
CMAX=2/1.185

class Candidate:
    def __init__(self,path):
        self.path=Path(path); data=json.loads(self.path.read_text())
        self.thresholds=np.asarray(data['physical_thresholds_C'],float)
        self.a=float(data['a']);self.b=float(data['b'])
        self.alpha=np.asarray(data['alpha'],float);self.beta=np.asarray(data['beta'],float)
        self.transform=data.get('transform')
        self.direct_calcium=data.get('observable_calcium')=='physical C'
        self.gate_coordinate=data.get('gate_coordinate','identity')
        self.gate_power=float(data.get('gate_power',1.))
        if self.gate_coordinate not in ('identity','power') or not np.isfinite(self.gate_power) or self.gate_power<=0:raise ValueError('Unsupported hidden coordinate for NEST benchmark')
        if self.direct_calcium and self.transform:raise ValueError('Physical-C model cannot also have a calcium transform')
        if self.a<=0 or self.b<=0 or len(self.alpha)!=3+len(self.thresholds) or len(self.beta)!=len(self.alpha):raise ValueError('Invalid candidate coefficients')
        if not np.all(np.isfinite(self.thresholds)) or not np.all(np.diff(self.thresholds)>0) or np.any(self.thresholds<=0):raise ValueError('Invalid thresholds')
        if not np.isfinite(np.r_[self.a,self.b,self.alpha,self.beta]).all():raise ValueError('Non-finite model coefficients')
        if self.transform and any(k not in self.transform for k in ('p','e','n','k')):raise ValueError('Unknown coordinate family')
        self.z_thresholds=self.phi(self.thresholds)
        self.coefficients=self.region_coefficients()

    def phi(self,c):
        c=np.asarray(c,dtype=float)
        if np.any(c<=0):raise ValueError('Physical calcium must be positive')
        if self.direct_calcium:return c
        if not self.transform:return c/.5-1
        t=c/.5;p,e,n,k=[float(self.transform[q]) for q in ('p','e','n','k')]
        if n<=0:raise ValueError('Transform exponent n must be positive')
        w=np.exp(np.array([p,e,0.])-max(p,e,0.));w/=w.sum()
        return w[0]*np.expm1(n*np.log(t))/n+w[1]*(np.expm1(np.clip(k*(t-1),-700,700))/k if abs(k)>1e-7 else t-1)+w[2]*np.log(t)

    def inverse(self,z):
        z=np.asarray(z,dtype=float)
        if self.direct_calcium:
            if np.any(z<=0) or np.any(z>=CMAX):raise ValueError('Trajectory leaves physical calcium interval')
            return z
        if not self.transform:
            c=.5*(z+1)
            if np.any(c<=0) or np.any(c>=CMAX):raise ValueError('Trajectory leaves physical calcium interval')
            return c
        lo=np.full_like(z,1e-14);hi=np.full_like(z,CMAX)
        if np.any(z<=self.phi(lo)) or np.any(z>=self.phi(hi)):raise ValueError('Trajectory leaves physical calcium interval')
        for _ in range(56):
            mid=(lo+hi)/2;mask=self.phi(mid)<z;lo=np.where(mask,mid,lo);hi=np.where(mask,hi,mid)
        return (lo+hi)/2

    def region_coefficients(self):
        out=[]
        for j in range(len(self.thresholds)+1):
            n=self.alpha[1]+self.alpha[3:3+j].sum();q=self.beta[1]+self.beta[3:3+j].sum()
            nc=self.alpha[0]-self.alpha[3:3+j]@self.z_thresholds[:j]
            hc=self.beta[0]-self.beta[3:3+j]@self.z_thresholds[:j]
            out.append([[-self.a*nc,-self.a*n,self.a,-self.a*self.alpha[2]],
                        [self.b*hc,self.b*q,-self.b,self.b*self.beta[2]]])
        return np.asarray(out,dtype=float)

    def rhs(self,t,y,baseline,tau):
        z,h,I=y;region=np.searchsorted(self.z_thresholds,z,side='right')
        dz,dh=self.coefficients[region]@np.array([1,z,h,I])
        return np.array([dz,dh,(baseline-I)/tau])

class NativeFlow:
    def __init__(self,library=None):
        if library is None:
            build=Path(tempfile.mkdtemp(prefix='fm_pwa_native_'));library=build/'pwa_event.so'
            subprocess.run(['g++','-O3','-shared','-fPIC',str(ROOT/'native'/'pwa_event.cpp'),'-o',str(library)],check=True)
        self.lib=ctypes.CDLL(str(library));ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');iptr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        self.lib.pwla_simulate.argtypes=[ptr,ctypes.c_int,ptr,ptr,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ptr,ptr,ctypes.c_int,ptr,ctypes.c_int,iptr]
        self.lib.pwla_simulate.restype=ctypes.c_int

    def simulate(self,model,initial,baseline,tau,duration,jumps=(),output_dt=.05,scan_dt=.02):
        if tau<=0 or duration<=0 or output_dt<=0 or scan_dt<=0:raise ValueError('Time parameters must be positive')
        c,h,I=map(float,initial);initial=np.ascontiguousarray([float(model.phi(c)),h,I],dtype=float)
        events=sorted((float(t),float(d)) for t,d in jumps)
        ts=np.ascontiguousarray([p[0] for p in events],dtype=float);ds=np.ascontiguousarray([p[1] for p in events],dtype=float)
        capacity=int(np.ceil(duration/output_dt))+2;out=np.empty((capacity,3));counts=np.zeros(3,dtype=np.int32)
        n=self.lib.pwla_simulate(np.ascontiguousarray(model.z_thresholds),len(model.thresholds)+1,np.ascontiguousarray(model.coefficients.ravel()),initial,float(baseline),float(tau),float(duration),float(output_dt),float(scan_dt),ts,ds,len(events),out.ravel(),capacity,counts)
        if n<0:raise RuntimeError(f'Native flow failed: status={n}')
        out=out[:n].copy();out[:,0]=model.inverse(out[:,0]);t=np.minimum(np.arange(n)*output_dt,duration)
        if abs(t[-1]-duration)>1e-9:raise RuntimeError('Native flow did not reach final time')
        return t,out,dict(affine_flows=int(counts[0]),threshold_crossings=int(counts[1]),input_jumps=int(counts[2]))
