import sympy as s,numpy as np,sys,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import root,brentq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phenomenological_k_fit'))
from fit_k import setup
C,h,I,K=s.symbols('C h I K',real=True);q=s.Rational(1049,1000)*(I+s.Rational(13,100))/(I+s.Rational(9434,10000));F=s.Matrix([(6*(I/(I+s.Rational(13,100)))**3*(C/(C+s.Rational(8234,100000)))**3*h**3+s.Rational(11,100))*(2-s.Rational(1185,1000)*C)-s.Rational(9,10)*C**2/(C**2+K**2),s.Rational(1,5)*(q*(1-h)-C*h)])
x=[C,h];jac=s.lambdify((C,h,I,K),F.jacobian(x),'numpy'); Bs=[[[s.diff(F[i],x[j],x[k]) for k in range(2)] for j in range(2)] for i in range(2)]; Cs=[[[[s.diff(F[i],x[j],x[k],x[l]) for l in range(2)] for k in range(2)] for j in range(2)] for i in range(2)]; bf=s.lambdify((C,h,I,K),Bs,'numpy');cf=s.lambdify((C,h,I,K),Cs,'numpy')
def calc(k,branch=1):
 m=setup(k,'am');c=m.cuts[branch];i=m.events[branch];q0=1.049*(i+.13)/(i+.9434);hh=q0/(q0+c);A=np.array(jac(c,hh,i,k),float); B=np.array(bf(c,hh,i,k),float);D=np.array(cf(c,hh,i,k),float)
 ev,V=np.linalg.eig(A); ix=np.argmax(ev.imag); w=ev[ix].imag;z=V[:,ix];ep,P=np.linalg.eig(A.T);p=P[:,np.argmin(abs(ep+1j*w))];p=p/np.conj(np.vdot(p,z))
 b=lambda u,v:np.einsum('ijk,j,k->i',B,u,v); d=lambda u,v,t:np.einsum('ijkl,j,k,l->i',D,u,v,t)
 G=np.vdot(p,d(z,z,z.conj())-2*b(z,np.linalg.solve(A,b(z,z.conj())))+b(z.conj(),np.linalg.solve(2j*w*np.eye(2)-A,b(z,z))))
 return float(G.real/(2*w)),c,i,w
ks=np.linspace(.1,.11,101); rows=[]; lower_rows=[]
for k in ks:
 l,c,i,w=calc(k);rows.append(dict(K=float(k),l1=l,C=c,I=i,omega=w))
 l,c,i,w=calc(k,0);lower_rows.append(dict(K=float(k),l1=l,C=c,I=i,omega=w))
k=brentq(lambda x:calc(x)[0],.102,.103,xtol=1e-12);l,c,i,w=calc(k)
out=Path(__file__).resolve().parent/'hopf_results';out.mkdir(exist_ok=True)
(out/'hopf_lyapunov.json').write_text(json.dumps(dict(formula='Kuznetsov first Lyapunov coefficient with eigenvectors normalized to Euclidean norm, <p,q>=1; root invariant under positive rescaling',upper_hopf_zero=dict(K=k,C=c,I=i,omega=w,l1=l),rows=rows,lower_hopf_rows=lower_rows,note='l1=0 is a candidate generalized Hopf, not proof of nonzero second Lyapunov coefficient or fold connectivity.'),indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,data,title in zip(axes,(rows,lower_rows),('Upper Hopf','Lower Hopf')):
 ax.plot(ks,[r['l1'] for r in data]);ax.axhline(0,color='black',lw=.8);ax.set_xlabel('K_ER (µM)');ax.set_ylabel('First Lyapunov coefficient l1');ax.set_title(title);ax.grid(alpha=.3)
axes[0].axvline(k,color='red',linestyle='--',label=f'zero K={k:.7f}');axes[0].legend();fig.savefig(out/'hopf_lyapunov.png',dpi=160)
print('ROOT',k,calc(k),'RESULT',out,flush=True)
