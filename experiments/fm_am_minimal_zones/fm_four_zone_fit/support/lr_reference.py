"""Li--Rinzel FM, K_ER=.05, De Pitta/legacy parameters used in this project."""
import numpy as np
from scipy.optimize import brentq
CMAX=2/1.185
def rhs(z,I):
    C,h=z;q=1.049*(I+.13)/(I+.9434)
    return np.array([(6*(I/(I+.13))**3*(C/(C+.08234))**3*h**3+.11)*(2-1.185*C)-.9*C*C/(C*C+.0025),.2*(q*(1-h)-C*h)])
def jac(z,I):
    C,h=z;q=1.049*(I+.13)/(I+.9434);m=I/(I+.13);n=C/(C+.08234)
    return np.array([[6*m**3*h**3*(3*n*n*.08234/(C+.08234)**2*(2-1.185*C)-1.185*n**3)-.13035-.0045*C/(C*C+.0025)**2,18*m**3*n**3*h*h*(2-1.185*C)],[-.2*h,-.2*(q+C)]])
def gate(C,I):
    q=1.049*(I+.13)/(I+.9434);return q/(q+C)
REST=brentq(lambda C:rhs([C,.8],0)[0],.001,.1,xtol=1e-15)
def input_at_C(C):
    if abs(C-REST)<1e-13:return 0.
    return brentq(lambda I:rhs([C,gate(C,I)],I)[0],0,20,xtol=1e-14)
def eqs(I):
    grid=np.r_[np.geomspace(1e-6,.1,500),np.linspace(.1,CMAX,700)];val=[rhs([c,gate(c,I)],I)[0] for c in grid];out=[]
    for a,b,va,vb in zip(grid[:-1],grid[1:],val[:-1],val[1:]):
        if va*vb<0:
            c=brentq(lambda c:rhs([c,gate(c,I)],I)[0],a,b,xtol=1e-14);out.append(np.array([c,gate(c,I)]))
    return out
def spectral_at_C(C):
    I=input_at_C(C);h=gate(C,I);J=jac([C,h],I);tr=np.trace(J);det=np.linalg.det(J)
    return np.array([tr,det,tr*tr-4*det])
def events():
    lo=REST+1e-6;hi=eqs(1.3)[-1][0];grid=np.linspace(lo,hi,2001);vals=np.array([spectral_at_C(c) for c in grid]);out=[]
    for col,kind in [(0,'trace_zero'),(1,'fold'),(2,'node_focus')]:
        for a,b,va,vb in zip(grid[:-1],grid[1:],vals[:-1,col],vals[1:,col]):
            if va*vb<0:
                c=brentq(lambda c:spectral_at_C(c)[col],a,b,xtol=1e-14);i=input_at_C(c);out.append(dict(kind=kind,C=float(c),I=float(i),h=float(gate(c,i))))
    return sorted(out,key=lambda q:q['C'])
