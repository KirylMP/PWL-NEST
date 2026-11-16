"""Independently evaluate a neutral-cycle candidate and its stable cycles."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from fit_transformed_coordinates import calc,features
from physical_fold_routes import Route

ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--subset',choices=('all','train','holdout'),default='all');p.add_argument('--progress',action='store_true');a=p.parse_args()
 j=json.loads(a.candidate.read_text());base=json.loads((ROOT/'results/coordinate_experiment'/f"{j['case']}_{j['kind']}.json").read_text());model=base['model'];n=np.array(j['n']);h=np.array(j['h']);aa=j['a'];bb=j['b']
 if a.progress:print('loaded candidate; subset=',a.subset,flush=True)
 model['coefficient_rows']=[[-aa*n[0],-aa*n[1],aa,-aa*n[2],-aa*n[3],-aa*n[4]],[bb*h[0],bb*h[1],-bb,bb*h[2],bb*h[3],bb*h[4]]]
 model['fold_constraint']={'I':j['I_fold'],'matching_max':j['matching_max'],'logM':j['logM'],'status':'numerical candidate; see independent checks'}
 model['fit']='joint neutral-orbit matching from initial field fit; not yet optimized for cycle error'
 times=np.array(j['times']);I=j['I_fold'];period=float(sum(times))
 out={'candidate':str(a.candidate),'I':I,'period':period,'IVP':[],'verification_subset':a.subset}
 if a.subset=='all':
  if a.progress:print('starting independent fold matching and IVP audit',flush=True)
  old={'physical_thresholds_C':j['cuts'],'alpha':j['n'],'beta':j['h'],'a':aa,'b':bb,'trace':j['trace']};route=Route(old,'both');res,z,q,mats=route.match(times,I,True);initial=np.array([z[0],q[0],0.]);out.update({'matching_max':float(max(abs(res))),'logM':float(np.array(j['trace'])[[1,2,1,0]]@times)})
  for step in (.03,.01,.003):
   if a.progress:print('fold IVP audit max_step=',step,flush=True)
   def rhs(_,y):
    U,Q=y[:2];v=np.r_[1,U,I,np.maximum(U-np.array(j['cuts']),0.)];r=np.searchsorted(j['cuts'],U,side='right');return [aa*(Q-n@v),bb*(h@v-Q),j['trace'][r]]
   v=solve_ivp(rhs,(0,period),initial,method='DOP853',rtol=1e-11,atol=1e-13,max_step=step)
   out['IVP'].append({'max_step_s':step,'success':bool(v.success),'closure':float(np.max(np.abs(v.y[:2,-1]-initial[:2]))),'logM':float(v.y[2,-1])})
  if a.progress:print('fold IVP audit finished; calculating nondegeneracy diagnostics',flush=True)
  eps=1e-5;f=lambda t:route.match(t,I);eye=np.eye(4);J=np.column_stack([(f(times+eps*e)-f(times-eps*e))/(2*eps) for e in eye]);U,S,V=np.linalg.svd(J);left=U[:,-1];right=V[-1];v=f(times);curv=[]
  for step in (1e-3,3e-4,1e-4):curv.append(float(left@(f(times+step*right)-2*v+f(times-step*right))/(step*step)))
  out['neutral_jacobian_singular_values']=S.tolist();out['matching_curvature_samples']=curv;out['matching_input_derivative']=float(left@(route.match(times,I+eps)-route.match(times,I-eps))/(2*eps));out['scope']='matching-equation fold diagnostic, not first-return derivatives; sampled route, finite precision and IVP, no global completeness'
 else:
  out.update({'matching_max':j.get('matching_max'),'logM':j.get('logM'),'scope':'fold audit skipped in subset mode; use the batch fold_validation for the independent fold check'})
  if a.progress:print('subset mode: skipping duplicate fold IVP and nondegeneracy audit',flush=True)
 data=[{'I':r['I'],'set':r['set'],'period':r['reference']['period'],'states':[[v,0.] for v in r['target_C']]} for r in base['results'] if a.subset=='all' or r['set']==a.subset]
 if a.progress:print('starting cycle integrations:',len(data),flush=True)
 out['cycles']=calc(model,data,progress=a.progress);out['candidate_model']=model
 suffix='_verification.json' if a.subset=='all' else f'_verification_{a.subset}.json';dest=a.candidate.with_name(a.candidate.stem+suffix);dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('cycles','candidate_model')},indent=2))
 for s in sorted({r['set'] for r in out['cycles']}):
  rows=[v for v in out['cycles'] if v['set']==s and v['prediction']];print(s,'cycles',len(rows),'MAE',*[float(np.mean([abs(v['prediction'][key]-v['reference'][key]) for v in rows])) if rows else None for key in ('Cmin','Cmax','period')])
if __name__=='__main__':main()
