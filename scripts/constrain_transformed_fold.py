"""Exploratory joint correction of transformed fit and a fixed-input neutral orbit.

It does not claim a fold unless matching, transversality, residence and
nondegeneracy are all independently checked.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import expm,null_space
from scipy.optimize import least_squares
from fit_transformed_coordinates import features

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=['AM','AFM','FM']);p.add_argument('--kind',required=True,choices=['log','square']);p.add_argument('--nfev',type=int,default=400);a=p.parse_args()
 path=ROOT/'results/coordinate_experiment'/f'{a.case}_{a.kind}.json';d=json.loads(path.read_text());m=d['model'];base=json.loads((ROOT/'paper/data/random_selected'/a.case/'candidate.json').read_text());cuts=np.array(m['cuts_u']);anchors=m['anchors'];N=null_space(np.array([features(u,q,I,cuts) for I,u,q in anchors]));coeff=np.array(m['coefficient_rows']).T;z0=(N.T@coeff).ravel();foldI=base['reference_fold']['I'];reg=(1,2,1,0);bounds=(0,1,1,0,0)
 t0=np.array(base['numerical_fold']['times']);zscale=np.maximum(np.abs(z0),.1)
 def parts(w,full=False):
  z=w[:len(z0)];t=np.exp(w[len(z0):]);c=N@z.reshape(3,2); starts=[];ends=[];mats=[];tr=[];speeds=[]
  for j,r in enumerate(reg):
   intercept=c[0,:]+c[3,:]*foldI-np.sum(c[4:4+r,:]*cuts[:r,None],axis=0)
   A=np.zeros((3,3));A[:2,0]=c[1,:]+np.sum(c[4:4+r,:],axis=0);A[:2,1]=c[2,:];A[:2,2]=intercept
   Q=expm(A*t[j]);den=Q[0,1]
   if abs(den)<1e-13:raise ValueError('singular')
   u0=cuts[bounds[j]];u1=cuts[bounds[j+1]];q=(u1-Q[0,0]*u0-Q[0,2])/den
   state=np.array([u0,q,1.]);starts.append(q);ends.append((Q@state)[1]);mats.append(A);tr.append(np.trace(A[:2,:]));speeds.append(float((A@state)[0]))
  match=np.array(ends)-np.roll(starts,-1);logm=float(np.dot(tr,t));return (match,logm,t,c,mats,np.array(starts),speeds) if full else (match,logm)
 def residual(w):
  try:
   match,logm,t,c,mats,starts,speeds=parts(w,True)
   matrices=[]
   for j in range(3):matrices.append(np.column_stack((c[1,:]+np.sum(c[4:4+j,:],axis=0),c[2,:])))
   T=np.array([np.trace(M) for M in matrices]);D=np.array([np.linalg.det(M) for M in matrices]);disc=T*T-4*D
   violated=np.maximum(0,-np.array([-T[0]-.001,D[0]-.00001,T[1]-.001,-disc[1]-.00001,-T[2]-.001,-disc[2]-.00001]))
   speed_violation=np.maximum(0,.00001-np.array(speeds)*np.array([1,1,-1,-1]))
   return np.r_[1000*match,1000*logm,10*violated,10*speed_violation,.01*(w[:len(z0)]-z0)/zscale]
  except (ValueError,OverflowError,FloatingPointError):return np.full(4+1+6+4+len(z0),1e4)
 lo=np.r_[np.full(len(z0),-np.inf),np.log(np.full(4,1e-5))];hi=np.r_[np.full(len(z0),np.inf),np.log(np.full(4,120.))]
 sol=least_squares(residual,np.r_[z0,np.log(t0)],bounds=(lo,hi),max_nfev=a.nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
 match,logm,t,c,mats,q,speeds=parts(sol.x,True);checks=[]
 for j,r in enumerate(reg):
  states=np.array([expm(mats[j]*s)@np.array([cuts[bounds[j]],q[j],1.]) for s in np.linspace(0,t[j],201)]);lower=-np.inf if r==0 else cuts[r-1];upper=np.inf if r==2 else cuts[r];checks.append(bool(np.min(states[:,0])>=lower-1e-7 and np.max(states[:,0])<=upper+1e-7))
 out={'case':a.case,'kind':a.kind,'input_I':foldI,'times':t.tolist(),'matching_max':float(max(abs(match))),'logM':logm,'speeds':speeds,'sampled_residence':checks,'cost':float(sol.cost),'nfev':sol.nfev,'optimizer_status':sol.message,'candidate_coefficients':c.T.tolist(),'scope':'attempt only; nondegeneracy and independent integration not checked'}
 filename=ROOT/'results/coordinate_experiment'/f'{a.case}_{a.kind}_fold_attempt.json';filename.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
