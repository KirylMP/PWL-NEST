"""Sample free parameters and solve one dependent coefficient plus fold times."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from optimize_fixed_fold_cycles import Field,flow,ROUTES

ROOT=Path(__file__).resolve().parents[1]
PIVOT_NAMES=('n_u','n_I','n_cut1','n_cut2','h_cut1_slope','h_cut2_slope','log_a','log_b')

def pivot_diagnostics(F,p,logs,eps=1e-5):
 """Test local fold solvability for each candidate dependent coefficient.

 The five fold equations are solved for one scaled field coefficient and four
 log flight times. Full rank of this 5x5 Jacobian is the local IFT condition.
 """
 def f(pp,tt):
  matching,logm=F.route(pp,F.Istar,tt,'both');return np.r_[matching,logm]
 Jt=np.column_stack([(f(p,logs+eps*np.eye(4)[i])-f(p,logs-eps*np.eye(4)[i]))/(2*eps) for i in range(4)])
 rows=[]
 for i,name in enumerate(PIVOT_NAMES):
  delta=np.eye(8)[i]*F.scale[i]*eps
  dp=(f(p+delta,logs)-f(p-delta,logs))/(2*eps)
  singular=np.linalg.svd(np.column_stack((dp,Jt)),compute_uv=False)
  ratio=float(singular[-1]/singular[0]) if singular[0]>0 else 0.
  rows.append({'index':i,'name':name,'sigma_min':float(singular[-1]),'sigma_max':float(singular[0]),'condition_number':float(singular[0]/singular[-1]) if singular[-1]>0 else None,'relative_sigma_min':ratio,'locally_solvable':bool(ratio>1e-8)})
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=('AM','AFM','FM'));p.add_argument('--kind',required=True,choices=('log','square','softplus'),default='softplus');p.add_argument('--seed',type=int,required=True);p.add_argument('--sigma',type=float,default=.03);p.add_argument('--maxfev',type=int,default=1200);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.sigma<=0:p.error('sigma must be positive')
 j=json.loads(a.reference.read_text());F=Field(a.case,a.kind,j);eps=1e-5
 def f(p,t):
  m,l=F.route(p,F.Istar,t,'both');return np.r_[m,l]
 J=np.column_stack([(f(F.p0,F.fold0+eps*np.eye(4)[i])-f(F.p0,F.fold0-eps*np.eye(4)[i]))/(2*eps) for i in range(4)])
 diagnostics=pivot_diagnostics(F,F.p0,F.fold0,eps)
 scores=[d['sigma_min'] for d in diagnostics]
 pivot=int(np.argmax(scores));free=[i for i in range(8) if i!=pivot];rng=np.random.default_rng(a.seed);p0=F.p0.copy();p0[free]+=rng.normal(0,a.sigma,len(free))*F.scale[free]
 def equations(z):
  pp=p0.copy();pp[pivot]=z[0]
  try:return f(pp,z[1:])
  except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):return np.full(5,1e6)
 sol=root(equations,np.r_[F.p0[pivot],F.fold0],tol=1e-10,options={'maxfev':a.maxfev});res=equations(sol.x)
 if max(abs(res))>1e-8:raise ValueError('dependent coefficient/fold root unresolved')
 if abs((sol.x[0]-F.p0[pivot])/F.scale[pivot])>.5 or max(abs(sol.x[1:]-F.fold0))>1.1:raise ValueError('fold branch jump')
 pp=p0.copy();pp[pivot]=sol.x[0];times=np.exp(sol.x[1:]);n,h,aa,bb=F.coeff(pp);mats=F.matrices(pp,F.Istar);T=np.array([np.trace(v[0]) for v in mats]);D=np.array([np.linalg.det(v[0]) for v in mats]);disc=T*T-4*D;_,lm,_,fm,states,speeds=F.route(pp,F.Istar,sol.x[1:],'both',True);residence=[];candidate_pivots=pivot_diagnostics(F,pp,sol.x[1:],eps)
 for i,r in enumerate(ROUTES['both'][0]):
  A,b=fm[r];u=flow(A,b,states[i],np.linspace(0,times[i],301))[:,0];lo=-np.inf if r==0 else F.cut[r-1];hi=np.inf if r==2 else F.cut[r];residence.append(bool(min(u)>=lo-1e-8 and max(u)<=hi+1e-8))
 spectral=bool(T[0]<0 and D[0]>0 and T[1]>0 and disc[1]<0 and T[2]<0 and disc[2]<0);directions=bool(min(speeds*np.array([1,1,-1,-1]))>1e-7)
 result={'case':a.case,'kind':a.kind,'seed':a.seed,'I_fold':F.Istar,'n':n.tolist(),'h':h.tolist(),'a':float(aa),'b':float(bb),'cuts':F.cut.tolist(),'times':times.tolist(),'matching_max':float(max(abs(res[:4]))),'logM':float(lm),'speeds':speeds.tolist(),'trace':T.tolist(),'determinant':D.tolist(),'discriminant':disc.tolist(),'sampled_residence':residence,'spectral_ok':spectral,'crossing_directions_ok':directions,'numerically_neutral_candidate':bool(spectral and directions and all(residence) and max(abs(res))<1e-8),'pivot_index':pivot,'pivot_name':PIVOT_NAMES[pivot],'pivot_diagnostics_at_reference':diagnostics,'pivot_diagnostics_at_candidate':candidate_pivots,'sigma':a.sigma,'source':str(a.reference),'scope':'implicit fixed-I fold solve; local pivot rank is finite-difference evidence, not global solvability'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'seed':a.seed,'matching':result['matching_max'],'candidate':result['numerically_neutral_candidate'],'pivot':pivot}),flush=True)
if __name__=='__main__':main()
