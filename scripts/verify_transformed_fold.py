"""Independently evaluate a neutral-cycle candidate and its stable cycles."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from fit_transformed_coordinates import calc,features
from physical_fold_routes import Route

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--subset',choices=('all','train','holdout'),default='all');p.add_argument('--progress',action='store_true');a=p.parse_args();j=json.loads(a.candidate.read_text());base=json.loads((ROOT/'results/coordinate_experiment'/f"{j['case']}_{j['kind']}.json").read_text());model=base['model'];n=np.array(j['n']);h=np.array(j['h']);aa=j['a'];bb=j['b'];
 model['coefficient_rows']=[[-aa*n[0],-aa*n[1],aa,-aa*n[2],-aa*n[3],-aa*n[4]],[bb*h[0],bb*h[1],-bb,bb*h[2],bb*h[3],bb*h[4]]]
 model['fold_constraint']={'I':j['I_fold'],'matching_max':j['matching_max'],'logM':j['logM'],'status':'numerical candidate; see independent checks'}
 model['fit']='joint neutral-orbit matching from initial field fit; not yet optimized for cycle error'
 old={'physical_thresholds_C':j['cuts'],'alpha':j['n'],'beta':j['h'],'a':aa,'b':bb,'trace':j['trace']};route=Route(old,'both');times=np.array(j['times']);I=j['I_fold'];res,z,q,mats=route.match(times,I,True);period=float(sum(times));initial=np.array([z[0],q[0],0.]);out={'candidate':str(a.candidate),'I':I,'matching_max':float(max(abs(res))),'logM':float(np.array(j['trace'])[[1,2,1,0]]@times),'period':period,'IVP':[]}
 for step in (.03,.01,.003):
  def rhs(_,y):
   U,Q=y[:2];v=np.r_[1,U,I,np.maximum(U-np.array(j['cuts']),0.)];r=np.searchsorted(j['cuts'],U,side='right');return [aa*(Q-n@v),bb*(h@v-Q),j['trace'][r]]
  v=solve_ivp(rhs,(0,period),initial,method='DOP853',rtol=1e-11,atol=1e-13,max_step=step)
  out['IVP'].append({'max_step_s':step,'success':bool(v.success),'closure':float(np.max(np.abs(v.y[:2,-1]-initial[:2]))),'logM':float(v.y[2,-1])})
 eps=1e-5;f=lambda t:route.match(t,I);eye=np.eye(4);J=np.column_stack([(f(times+eps*e)-f(times-eps*e))/(2*eps) for e in eye]);U,S,V=np.linalg.svd(J);left=U[:,-1];right=V[-1];v=f(times);curv=[]
 for step in (1e-3,3e-4,1e-4):curv.append(float(left@(f(times+step*right)-2*v+f(times-step*right))/(step*step)))
 out['neutral_jacobian_singular_values']=S.tolist();out['matching_curvature_samples']=curv;out['matching_input_derivative']=float(left@(route.match(times,I+eps)-route.match(times,I-eps))/(2*eps));out['scope']='matching-equation fold diagnostic, not first-return derivatives; sampled route, finite precision and IVP, no global completeness'
 data=[{'I':r['I'],'set':r['set'],'period':r['reference']['period'],'states':[[v,0.] for v in r['target_C']]} for r in base['results'] if a.subset=='all' or r['set']==a.subset];out['verification_subset']=a.subset;out['cycles']=calc(model,data,progress=a.progress);out['candidate_model']=model
 suffix='_verification.json' if a.subset=='all' else f'_verification_{a.subset}.json';dest=a.candidate.with_name(a.candidate.stem+suffix);dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('cycles','candidate_model')},indent=2))
 for s in sorted({r['set'] for r in out['cycles']}):
  rows=[v for v in out['cycles'] if v['set']==s and v['prediction']];print(s,'cycles',len(rows),'MAE',*[float(np.mean([abs(v['prediction'][key]-v['reference'][key]) for v in rows])) if rows else None for key in ('Cmin','Cmax','period')])
if __name__=='__main__':main()
