"""Fast numerical cycle counts, with adaptive native flow and root refinement."""
import ctypes,json,subprocess,os,threading
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from lr_reference import eqs,jac,CMAX
P=Path(__file__).resolve().parent
if not (P/'libflow.so').exists():subprocess.run(['g++','-O3','-shared','-fPIC',str(P/'flow.cpp'),'-o',str(P/'libflow.so')],check=True)
LIB=ctypes.CDLL(str(P/'libflow.so'));PTR=ctypes.POINTER(ctypes.c_double)
LIB.first_return.argtypes=[ctypes.c_int,PTR,PTR,ctypes.c_double,ctypes.c_int,*([ctypes.c_double]*6),PTR]
LIB.first_return.restype=ctypes.c_int
_LIB9=None
_LIB9_LOCK=threading.Lock()
def lib9():
    global _LIB9
    with _LIB9_LOCK:
        if _LIB9 is None:
            path=P/'libflow9.so';src=P/'flow.cpp'
            if not path.exists() or path.stat().st_mtime_ns<src.stat().st_mtime_ns:
                tmp=P/f'.libflow9_{os.getpid()}.so'
                subprocess.run(['g++','-O3','-shared','-fPIC','-DMAX_CUTS=9',str(src),'-o',str(tmp)],check=True)
                os.replace(tmp,path)
            _LIB9=ctypes.CDLL(str(path));_LIB9.first_return.argtypes=LIB.first_return.argtypes;_LIB9.first_return.restype=ctypes.c_int
        return _LIB9

def ret(I,s,section,co=None,th=None,bounds=None,back=False,tol=2e-9,fast=False):
    mode=(2 if fast else 1) if co is not None else 0;co=np.ascontiguousarray(np.zeros((2,11)) if co is None else co,dtype=float);th=np.ascontiguousarray(np.zeros(7) if th is None else th,dtype=float);out=np.zeros(7)
    lo,hi=(0,CMAX) if bounds is None else bounds
    if co.shape not in ((2,11),(2,13)) or th.shape not in ((7,),(9,)) or co.shape[1]!=len(th)+4:raise ValueError('Unsupported return-map coefficient layout')
    lib=LIB if len(th)==7 else lib9()
    status=lib.first_return(mode,co.ctypes.data_as(PTR),th.ctypes.data_as(PTR),float(I),-1 if back else 1,float(section),float(s),float(lo),float(hi),float(tol),3000.,out.ctypes.data_as(PTR))
    return status,out

def cycles(I,equilibria,co=None,th=None,bounds=None,n=28,tol=2e-9,seeds=()):
    found=[];timeouts=0;unresolved=0
    for eq in equilibria:
        section,h=eq['z'] if isinstance(eq,dict) else eq
        if not (0<h<1):continue
        span=1-h
        if span<1e-5:continue
        rr=np.unique(np.r_[np.geomspace(3e-5,span*.12,n//2),np.linspace(span*.12,span*.999,n//2),[s-h for s in seeds if 3e-5<s-h<span]])
        for back in [False,True]:
            cache={}
            def fun(r):
                nonlocal timeouts
                if r not in cache:cache[r]=ret(I,h+r,section,co,th,bounds,back,tol)
                status,v=cache[r]
                if status==-2:timeouts+=1
                return v[0]/r if status==1 else np.nan
            dd=[fun(r) for r in rr]
            for a,b,fa,fb in zip(rr[:-1],rr[1:],dd[:-1],dd[1:]):
                if not (np.isfinite(fa) and np.isfinite(fb) and fa*fb<0):continue
                if max(abs(fa),abs(fb))<1e-7:unresolved+=1;continue
                try:r=brentq(fun,a,b,xtol=1e-10)
                except ValueError:unresolved+=1;continue
                status,q=ret(I,h+r,section,co,th,bounds,back,tol/3)
                if status!=1 or abs(q[0])/r>3e-5 or q[4]-q[3]<1e-5 or abs(q[2])<1e-6:unresolved+=1;continue
                # Forward-stable cycles are found reliably by forward flow, unstable by reverse.
                stable=q[2]<0
                if stable==back:continue
                if any(stable==p['stable'] and abs(q[1]-p['T'])<5e-6*max(1,q[1]) and abs(q[3]-p['min'])<.002 and abs(q[4]-p['max'])<.002 for p in found):continue
                found.append(dict(I=float(I),s=float(h+r),section=float(section),T=float(q[1]),log_multiplier=float(q[2]),stable=bool(stable),min=float(q[3]),max=float(q[4]),hmin=float(q[5]),hmax=float(q[6]),residual=float(q[0]),backward=back))
    return found,dict(timeouts=timeouts,unresolved=unresolved)

def lr_state(I,n=36):
    ee=eqs(I);stable=sum(np.linalg.eigvals(jac(e,I)).real.max()<0 for e in ee);q,diag=cycles(I,ee,n=n)
    state=[int(stable),len(ee)-int(stable),sum(p['stable'] for p in q),sum(not p['stable'] for p in q)]
    return dict(I=float(I),state=state,equilibria=[e.tolist() for e in ee],cycles=q,diagnostics=diag)

if __name__=='__main__':
    import sys
    vals=list(map(float,sys.argv[1:])) or [0,.5,.52,.53,.55,.8,.9,1.,1.1,1.2,2.]
    for I in vals:print(json.dumps(lr_state(I)),flush=True)
