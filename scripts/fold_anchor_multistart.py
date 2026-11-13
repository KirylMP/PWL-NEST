"""Fixed-input neutral-orbit search with free hidden coordinates at anchors.

For a nondegenerate candidate, separately verify return-map curvature and
input sensitivity, and compare with independent integration. A small residual
alone is not a fold proof.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares,brentq

ROOT=Path(__file__).resolve().parents[1]
R=(1,2,1,0);B=(0,1,1,0,0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=('AM','AFM','FM'));p.add_argument('--kind',required=True,choices=('log','square'));p.add_argument('--seed',type=int,default=0);p.add_argument('--nfev',type=int,default=1000);p.add_argument('--resume',type=Path);p.add_argument('--exact',action='store_true');p.add_argument('--fold-amplitude-weight',type=float,default=0.);args=p.parse_args()
 m=json.loads((ROOT/'results/coordinate_experiment'/f'{args.case}_{args.kind}.json').read_text())['model'];base=json.loads((ROOT/'paper/data/random_selected'/args.case/'candidate.json').read_text());cut=np.array(m['cuts_u']);u0=m['rest_state'][0];I1,I2=base['fixed_equilibrium_threshold_I'];If=base['reference_fold']['I'];c=np.array(m['coefficient_rows']);aa=c[0,2];bb=-c[1,2];assert aa>0 and bb>0
 # F_u=a(q-N), F_q=b(H-q), with N,H continuous linear splines.
 n0=-c[0,[0,1,3,4,5]]/aa;h0=c[1,[0,1,3,4,5]]/bb;diff=n0-h0
 K=np.array([[1,0,max(u0-cut[0],0)],[1,I1,0],[1,I2,cut[1]-cut[0]]]);Z=np.array([u0,cut[0],cut[1]])
 def unpack(x):
  n=np.r_[m['rest_state'][1]-x[0]*u0,x[:4]];d1,d2=x[4:6];dep=np.linalg.solve(K,-d1*Z);dif=np.array([dep[0],d1,dep[1],dep[2],d2]);h=n-dif;a=np.exp(x[6]);b=np.exp(x[7]);times=np.exp(x[8:]);return n,h,a,b,times
 initial=np.r_[n0[1:],diff[1],diff[4],np.log(aa),np.log(bb),np.log(base['numerical_fold']['times'])];scales=np.maximum(np.abs(initial[:6]),.25);rng=np.random.default_rng(args.seed);x0=initial.copy()
 if args.seed:x0[:6]+=rng.normal(0,.15,6)*scales;x0[6:8]+=rng.normal(0,.1,2);x0[8:]+=rng.normal(0,.3,4)
 if args.resume:
  old=json.loads(args.resume.read_text());n=np.array(old['n']);h=np.array(old['h']);x0=np.r_[n[1:],n[1]-h[1],n[4]-h[4],np.log(old['a']),np.log(old['b']),np.log(old['times'])];initial=x0.copy();scales=np.maximum(np.abs(initial[:6]),.25)
 def evaluate(x):
  n,h,a,b,times=unpack(x);beg=[];ends=[];mats=[];speed=[]
  for j,r in enumerate(R):
   nu=n[1]+sum(n[3:3+r]);hu=h[1]+sum(h[3:3+r]);nc=n[0]+n[2]*If-n[3:3+r]@cut[:r];hc=h[0]+h[2]*If-h[3:3+r]@cut[:r]
   A=np.array([[-a*nu,a,-a*nc],[b*hu,-b,b*hc],[0.,0.,0.]]);Q=expm(A*times[j]);den=Q[0,1]
   if abs(den)<1e-12:raise ValueError('singular')
   u=cut[B[j]];q=(cut[B[j+1]]-Q[0,0]*u-Q[0,2])/den;state=np.array([u,q,1.]);beg.append(q);ends.append((Q@state)[1]);mats.append(A);speed.append(float((A@state)[0]))
  T=[];D=[]
  for r in range(3):
   A=np.array([[-a*(n[1]+sum(n[3:3+r])),a],[b*(h[1]+sum(h[3:3+r])),-b]]);T.append(np.trace(A));D.append(np.linalg.det(A))
  T=np.array(T);D=np.array(D);disc=T*T-4*D
  return np.array(ends)-np.roll(beg,-1),float(T[list(R)]@times),np.array(speed),T,D,disc,mats,np.array(beg),times,n,h,a,b
 def fun(x):
  try:
   match,logm,sp,T,D,disc,mats,q,t,*_=evaluate(x)
   bad=np.maximum(0,-np.array([-T[0]-.003,D[0]-.0001,T[1]-.003,-disc[1]-.0001,-T[2]-.003,-disc[2]-.0001]));speedbad=np.maximum(0,.0001-sp*np.array([1,1,-1,-1]));period=unpack(x)[-1].sum()
   regularizer=1e-6 if args.exact else .001; period_weight=0. if args.exact else .005
   amplitude_residual=0.
   if args.fold_amplitude_weight:
    upper0=np.array([cut[1],q[1],1.]);v=lambda dt:float((mats[1]@expm(mats[1]*dt)@upper0)[0]);v0,v1=v(0),v(t[1]);tmax=brentq(v,0,t[1]) if v0*v1<0 else (0 if v0<0 else t[1]);umax=float((expm(mats[1]*tmax)@upper0)[0]);
    if args.kind=='log' and umax>100:raise ValueError('overflowing physical readout')
    cmax=np.exp(umax) if args.kind=='log' else umax**2;amplitude_residual=args.fold_amplitude_weight*(cmax-base['reference_fold']['C_max'])/.08
   return np.r_[200*match,100*logm,30*bad,50*speedbad,regularizer*(x[:6]-initial[:6])/scales,regularizer*(x[6:8]-initial[6:8]),period_weight*(period-base['reference_fold']['period']),amplitude_residual]
  except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):return np.full(4+1+6+4+6+2+1+1,1e5)
 lo=np.r_[np.full(6,-np.inf),np.log([.005,.001]),np.log(np.full(4,.00001))];hi=np.r_[np.full(6,np.inf),np.log([100.,20.]),np.log(np.full(4,50.))];x0=np.clip(x0,lo+1e-9,hi-1e-9)
 sol=least_squares(fun,x0,bounds=(lo,hi),max_nfev=args.nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10)
 match,logm,sp,T,D,disc,mats,q,t,n,h,a,b=evaluate(sol.x);residence=[]
 for j,r in enumerate(R):
  z=np.array([expm(mats[j]*s)@np.array([cut[B[j]],q[j],1.]) for s in np.linspace(0,t[j],301)]);lo=-np.inf if r==0 else cut[r-1];hi=np.inf if r==2 else cut[r];residence.append(bool(min(z[:,0])>=lo-1e-8 and max(z[:,0])<=hi+1e-8))
 spectral=bool(T[0]<0 and D[0]>0 and T[1]>0 and disc[1]<0 and T[2]<0 and disc[2]<0);directions=bool(np.min(sp*np.array([1,1,-1,-1]))>1e-7);near=bool(max(abs(match))<1e-7 and abs(logm)<1e-7 and spectral and directions and all(residence))
 out={'case':args.case,'kind':args.kind,'seed':args.seed,'I_fold':If,'n':n.tolist(),'h':h.tolist(),'a':a,'b':b,'cuts':cut.tolist(),'times':t.tolist(),'q_crossings':q.tolist(),'matching_max':float(max(abs(match))),'logM':logm,'speeds':sp.tolist(),'trace':T.tolist(),'determinant':D.tolist(),'discriminant':disc.tolist(),'sampled_residence':residence,'spectral_ok':spectral,'crossing_directions_ok':directions,'numerically_neutral_candidate':near,'nfev':sol.nfev,'message':sol.message,'cost':float(sol.cost),'scope':'sampled residence only; nondegeneracy and independent IVP not yet checked'}
 dest=ROOT/'results/fold_anchors'/args.case/args.kind;dest.mkdir(parents=True,exist_ok=True);(dest/f'seed_{args.seed}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('case','kind','seed','matching_max','logM','speeds','spectral_ok','sampled_residence','numerically_neutral_candidate','nfev')}),flush=True)
if __name__=='__main__':main()
