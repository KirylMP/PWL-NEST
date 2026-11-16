"""Polish a fitted candidate's fixed-I neutral fold without refitting its field.

Only the selected dependent coefficient and the four fold flight times move.
All other free coefficients remain fixed. The original file is never changed.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from optimize_fixed_fold_cycles import Field
from run_transformed_server_batch import check_fold

def main():
 p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 src=a.candidate.resolve();j=json.loads(src.read_text());case=j['case'];kind=j['kind']
 if 'pivot_index' not in j:raise SystemExit('candidate has no pivot_index; cannot preserve the fitted parameterization')
 pivot=int(j['pivot_index']);F=Field(case,kind,j);p0=F.p0.copy();z0=np.r_[p0[pivot],F.fold0]
 def equations(z):
  pp=p0.copy();pp[pivot]=z[0]
  try:
   matching,logm=F.route(pp,F.Istar,z[1:],'both')
   return np.r_[matching,logm]
  except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):return np.full(5,1e6)
 before=equations(z0);sol=root(equations,z0,method='hybr',tol=1e-11,options={'xtol':1e-12,'maxfev':2000});after=equations(sol.x)
 if not np.isfinite(after).all() or max(abs(after))>1e-10:raise SystemExit(f'fold polish did not converge tightly: max residual {max(abs(after)):.3g}; solver: {sol.message}')
 parameter_shift=abs(sol.x[0]-z0[0])/F.scale[pivot];time_shift=float(max(abs(sol.x[1:]-z0[1:])))
 if parameter_shift>1e-3 or time_shift>1e-3:raise SystemExit(f'root correction is not a tiny polish (scaled coefficient shift={parameter_shift:.3g}, max log-time shift={time_shift:.3g})')
 pp=p0.copy();pp[pivot]=sol.x[0];times=np.exp(sol.x[1:]);n,h,aa,bb=F.coeff(pp);_,logm,_,mats,states,speeds=F.route(pp,F.Istar,sol.x[1:],'both',True)
 traces=np.array([np.trace(A) for A,b in mats]);dets=np.array([np.linalg.det(A) for A,b in mats]);disc=traces*traces-4*dets
 polished=dict(j);polished['pre_polish_cycles']=polished.pop('cycles',None);polished.update({'n':n.tolist(),'h':h.tolist(),'a':float(aa),'b':float(bb),'times':times.tolist(),'trace':traces.tolist(),'determinant':dets.tolist(),'discriminant':disc.tolist(),'speeds':speeds.tolist(),'matching_max':float(max(abs(after[:4]))),'logM':float(logm),'pre_polish':{'candidate_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'matching_max':float(j.get('matching_max',float('nan'))),'logM':float(j.get('logM',float('nan'))),'scaled_pivot_shift':float(parameter_shift),'max_log_time_shift':time_shift,'note':'training loss and cycle predictions in the source report refer to the unpolished fit'}})
 validation=check_fold(polished);polished['polish_validation']=validation
 dest=a.output.resolve() if a.output else src.with_name(src.stem+'_polished.json');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(polished,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'output':str(dest),'case':case,'kind':kind,'matching_max':polished['matching_max'],'logM':polished['logM'],'scaled_pivot_shift':parameter_shift,'max_log_time_shift':time_shift,'independent_validation':validation},indent=2))
if __name__=='__main__':main()
