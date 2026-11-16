"""Collect and fit distinct fixed-fold transformed models with bounded parallelism.

All progress is written under --output and may be resumed. A numerically valid
fold does not imply that its seven training cycles or four holdouts are usable.
The counts refer to sampled, finite-tolerance validations, not proofs.
"""
import argparse,csv,hashlib,json,os,platform,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from physical_fold_routes import Route

ROOT=Path(__file__).resolve().parents[1]
REFERENCES={('AM','log'):6,('AM','square'):3,('AFM','log'):2,('AFM','square'):3,('FM','log'):7,('FM','square'):0}

def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def resolve_path(value):
 p=Path(value);return p if p.is_absolute() else ROOT/p
def portable(path):
 try:return str(path.resolve().relative_to(ROOT))
 except ValueError:return str(path.resolve())
def invoke(args,log,timeout,env):
 with log.open('w') as f:
  try:
   r=subprocess.run([sys.executable,'-u',*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=timeout);return r.returncode
  except subprocess.TimeoutExpired:f.write('\nTimed out.\n');return 124

def check_fold(j):
 original=json.loads((ROOT/'paper/data/random_selected'/j['case']/'candidate.json').read_text());coordinate=json.loads((ROOT/'results/coordinate_experiment'/f"{j['case']}_{j['kind']}.json").read_text())['model'];
 if abs(j['I_fold']-original['reference_fold']['I'])>1e-12 or np.max(np.abs(np.asarray(j['cuts'])-np.asarray(coordinate['cuts_u'])))>1e-12:raise ValueError('fixed fold input or thresholds changed')
 if j['a']<=0 or j['b']<=0:raise ValueError('nonpositive timescale coefficient')
 nn=np.asarray(j['n']);hh=np.asarray(j['h']);u0,q0=coordinate['rest_state'];i1,i2=original['fixed_equilibrium_threshold_I'];cuts=np.asarray(j['cuts'])
 def hinge(u,I):return np.r_[1.,u,I,np.maximum(u-cuts,0.)]
 x0=hinge(u0,0.);errors=[j['a']*(q0-nn@x0),j['b']*(hh@x0-q0),(nn-hh)@hinge(cuts[0],i1),(nn-hh)@hinge(cuts[1],i2)];anchor_error=float(max(abs(np.asarray(errors))))
 if anchor_error>1e-8:raise ValueError('fixed rest or threshold equilibrium anchor')
 d={'physical_thresholds_C':j['cuts'],'alpha':j['n'],'beta':j['h'],'a':j['a'],'b':j['b'],'trace':j['trace']};orb=Route(d,'both');t=np.asarray(j['times']);I=float(j['I_fold']);res,z,q,mats=orb.match(t,I,True);speed=[];U=[];regions=orb.regions
 if max(abs(res))>1e-8 or abs(np.asarray(j['trace'])[list(regions)]@t)>1e-8:raise ValueError('fold matching or logM tolerance')
 T=np.array(j['trace']);D=np.array(j['determinant']);disc=T*T-4*D
 if not (T[0]<0 and D[0]>0 and T[1]>0 and disc[1]<0 and T[2]<0 and disc[2]<0):raise ValueError('spectral conditions')
 for k,(A,r) in enumerate(zip(mats,regions)):
  state=np.r_[z[k],q[k],1.];speed.append(float((A@state)[0]));values=np.array([(expm(A*s)@state)[0] for s in np.linspace(0,t[k],301)]);lo=-np.inf if r==0 else j['cuts'][r-1];hi=np.inf if r==2 else j['cuts'][r]
  if min(values)<lo-1e-8 or max(values)>hi+1e-8:raise ValueError('sampled fold itinerary')
  U.extend(values.tolist())
 if min(np.asarray(speed)*[1,1,-1,-1])<=1e-7:raise ValueError('fold crossing direction or near grazing')
 n=np.array(j['n']);h=np.array(j['h']);cut=np.array(j['cuts']);a=j['a'];b=j['b']
 def rhs(_,y):
  u,v=y[:2];f=np.r_[1,u,I,np.maximum(u-cut,0)];r=np.searchsorted(cut,u,side='right');return [a*(v-n@f),b*(h@f-v),T[r]]
 initial=np.array([z[0],q[0],0.]);sol=solve_ivp(rhs,(0.,sum(t)),initial,method='DOP853',rtol=1e-10,atol=1e-12,max_step=.01)
 if not sol.success:raise ValueError('independent IVP failure')
 closure=float(max(abs(sol.y[:2,-1]-initial[:2])));ivplog=float(sol.y[2,-1])
 if closure>1e-6 or abs(ivplog)>1e-4:raise ValueError('independent IVP closure or logM')
 eps=1e-5;eye=np.eye(4);f=lambda x:orb.match(x,I);J=np.column_stack([(f(t+eps*v)-f(t-eps*v))/(2*eps) for v in eye]);L,S,V=np.linalg.svd(J);left=L[:,-1];right=V[-1];base=f(t);curv=[float(left@(f(t+s*right)-2*base+f(t-s*right))/(s*s)) for s in (1e-3,3e-4,1e-4)];di=float(left@(orb.match(t,I+eps)-orb.match(t,I-eps))/(2*eps))
 if not np.isfinite(curv+[di]).all() or min(map(abs,curv))<1e-10 or abs(di)<1e-8 or not(all(x>0 for x in curv) or all(x<0 for x in curv)):raise ValueError('nondegeneracy unresolved')
 U=np.array(U);C=np.exp(U) if j['kind']=='log' else U*U
 if not np.isfinite(C).all():raise ValueError('readout overflow')
 return {'anchor_max_residual':anchor_error,'matching_max':float(max(abs(res))),'logM':float(np.asarray(j['trace'])[list(regions)]@t),'ivp_closure':closure,'ivp_logM':ivplog,'crossing_speeds':speed,'minimum_u_sampled':float(min(U)),'Cmin_sampled':float(min(C)),'Cmax_sampled':float(max(C)),'curvature_samples':curv,'input_derivative':di,'singular_values':S.tolist(),'scope':'finite differences, sampled residence and IVP; not a proof of first exit'}

def signature(j):return np.r_[j['n'],j['h'],j['a'],j['b'],np.log(j['times'])]
def duplicate(sig,existing,tol):
 return any(float(np.linalg.norm((sig-other)/np.maximum(1.,np.maximum(abs(sig),abs(other)))))<tol for other in existing)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--valid-per-combo',type=int,default=32);ap.add_argument('--workers',type=int,default=16);ap.add_argument('--fold-nfev',type=int,default=1200);ap.add_argument('--fit-nfev',type=int,default=80);ap.add_argument('--max-attempts',type=int,default=5000);ap.add_argument('--sigma',type=float,default=.03);ap.add_argument('--dedup-tol',type=float,default=1e-5);ap.add_argument('--timeout',type=int,default=3600);ap.add_argument('--combos',nargs='+',default=[f'{a}:{b}' for a,b in REFERENCES]);ap.add_argument('--resume',action='store_true');ap.add_argument('--retry-failed',action='store_true');args=ap.parse_args()
 if min(args.valid_per_combo,args.workers,args.fold_nfev,args.fit_nfev,args.max_attempts,args.timeout)<1 or args.sigma<=0 or args.dedup_tol<0:ap.error('invalid numerical settings')
 if args.retry_failed and not args.resume:ap.error('--retry-failed requires --resume')
 combos=[]
 for item in args.combos:
  parts=item.split(':');pair=tuple(parts)
  if len(parts)!=2 or pair not in REFERENCES or pair in combos:ap.error('unknown/duplicate combo; use e.g. AM:square')
  combos.append(pair)
 out=args.output.resolve();manifest=out/'batch.json';config={'combos':[list(x) for x in combos],'valid_per_combo':args.valid_per_combo,'fold_nfev':args.fold_nfev,'fit_nfev':args.fit_nfev,'max_attempts':args.max_attempts,'sigma':args.sigma,'dedup_tol':args.dedup_tol,'timeout':args.timeout}
 if manifest.exists():
  old=json.loads(manifest.read_text())
  if not args.resume:raise SystemExit('Output exists; use --resume or a fresh directory.')
  if old['config']!=config:raise SystemExit('Resume parameters differ from batch manifest.')
 else:
  if out.exists() and any(out.iterdir()):raise SystemExit('Output directory is non-empty; choose a new directory.')
  out.mkdir(parents=True,exist_ok=True)
  rev=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True);atomic(manifest,{'config':config,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'git_head':rev.stdout.strip() if rev.returncode==0 else None,'python':platform.python_version(),'numpy':np.__version__,'workers_initial':args.workers,'selection':'first distinct candidates passing finite validations, not uniform over all valid fields'})
 env=os.environ.copy();env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(ROOT/'scripts')+os.pathsep+str(ROOT/'src')})
 accepted={c:[] for c in combos};sigs={c:[] for c in combos};nextid={c:0 for c in combos}
 for combo in combos:
  c,k=combo;group=out/'search'/f'{c}_{k}';group.mkdir(parents=True,exist_ok=True)
  previous=[]
  for attempt in sorted(group.glob('attempt_*')):
   try:nextid[combo]=max(nextid[combo],int(attempt.name.split('_')[-1])+1)
   except ValueError:continue
   status=attempt/'status.json'
   if not status.exists():continue
   item=json.loads(status.read_text());previous.append(item)
  for item in sorted(previous,key=lambda r:(r.get('accepted_order',10**12),r['seed'])):
   if item.get('fold_valid') and item.get('distinct') is not False and resolve_path(item['candidate']).exists() and len(accepted[combo])<args.valid_per_combo:
    vec=signature(json.loads(resolve_path(item['candidate']).read_text()))
    if not duplicate(vec,sigs[combo],args.dedup_tol):accepted[combo].append(item);sigs[combo].append(vec)
 def worker(combo,index):
  c,k=combo;attempt=out/'search'/f'{c}_{k}'/f'attempt_{index:06d}';attempt.mkdir(parents=True,exist_ok=True);state={'case':c,'kind':k,'seed':index,'fold_valid':False};source=ROOT/'results/fold_anchors'/c/k/f'seed_{REFERENCES[combo]}.json'
  try:
   if not source.is_file():raise FileNotFoundError(f'missing reference {source}')
   if index==0:
    candidate=source
   else:
    candidate=attempt/f'seed_{index}.json';sigma=args.sigma*([.1,1/3,1.,3.][(index-1)%4]);state['sample_sigma']=sigma
    params=[ROOT/'scripts/sample_transformed_fold.py','--case',c,'--kind',k,'--seed',index,'--sigma',sigma,'--maxfev',args.fold_nfev,'--reference',source,'--output',candidate]
    code=invoke(params,attempt/'search.log',args.timeout,env)
    if code or not candidate.exists():raise ValueError(f'implicit fold search unresolved or timed out ({code})')
    pre=json.loads(candidate.read_text())
    if not pre['numerically_neutral_candidate']:raise ValueError('implicit solution failed spectral/direction/itinerary checks')
   candidate=candidate.resolve();j=json.loads(candidate.read_text());state.update({'candidate':portable(candidate),'source_sha256':sha(source),'candidate_sha256':sha(candidate),'validation':check_fold(j),'fold_valid':True})
  except Exception as e:state['reason']=f'{type(e).__name__}: {e}'
  atomic(attempt/'status.json',state);return state

 print('PHASE 1: collecting fixed-fold candidates; workers:',args.workers,flush=True)
 with ThreadPoolExecutor(max_workers=args.workers) as pool:
  pending={}
  while True:
   while len(pending)<args.workers:
    launched=False
    for combo in combos:
     running=sum(c==combo for c in pending.values())
     if len(pending)<args.workers and len(accepted[combo])+running<args.valid_per_combo and nextid[combo]<args.max_attempts:
      i=nextid[combo];nextid[combo]+=1;pending[pool.submit(worker,combo,i)]=combo;launched=True
    if not launched:break
   if not pending:break
   completed,_=wait(pending,return_when=FIRST_COMPLETED)
   for future in completed:
    combo=pending.pop(future)
    try:item=future.result()
    except Exception as e:print('search worker failure',combo,repr(e),flush=True);continue
    if item.get('fold_valid'):
     vec=signature(json.loads(resolve_path(item['candidate']).read_text()))
     if duplicate(vec,sigs[combo],args.dedup_tol):item['distinct']=False
     else:item['distinct']=True;item['accepted_order']=len(accepted[combo]);accepted[combo].append(item);sigs[combo].append(vec)
     status=out/'search'/f'{combo[0]}_{combo[1]}'/f"attempt_{item['seed']:06d}"/'status.json';atomic(status,item)
    print('search',*combo,'attempt',item['seed'],'accepted',len(accepted[combo]),'/',args.valid_per_combo,'status',('accepted' if item.get('distinct') else item.get('reason','duplicate')),flush=True)
  
 atomic(out/'accepted.json',{'groups':{f'{c}_{k}':accepted[(c,k)] for c,k in combos}})
 print('PHASE 2: fit all collected candidates; workers:',args.workers,flush=True)
 def fit(item):
  c=item['case'];k=item['kind'];seed=item['seed'];directory=out/'fits'/f'{c}_{k}'/f'seed_{seed:06d}';directory.mkdir(parents=True,exist_ok=True);status_path=directory/'status.json'
  if args.resume and status_path.exists():
   previous=json.loads(status_path.read_text())
   if not args.retry_failed or previous.get('status')=='completed':return previous
  status={'case':c,'kind':k,'seed':seed,'input_candidate':item['candidate'],'input_sha256':item['candidate_sha256'],'status':'failed'}
  try:
   candidate_input=resolve_path(item['candidate'])
   if sha(candidate_input)!=item['candidate_sha256']:raise ValueError('input candidate hash mismatch')
   fitdir=directory/'fit';params=[ROOT/'scripts/optimize_fixed_fold_cycles.py','--case',c,'--kind',k,'--candidate',candidate_input,'--output',fitdir,'--max-nfev',args.fit_nfev];code=invoke(params,directory/'fit.log',args.timeout,env)
   if code or not (fitdir/'candidate.json').exists() or not (fitdir/'report.json').exists():raise ValueError(f'fit failed or timed out ({code})')
   candidate=fitdir/'candidate.json';v=check_fold(json.loads(candidate.read_text()));status.update({'fold_validation':v,'candidate_sha256':sha(candidate),'training_loss':json.loads((fitdir/'report.json').read_text())['best_loss'],'optimizer_message':json.loads((fitdir/'report.json').read_text())['message']})
   code=invoke([ROOT/'scripts/verify_transformed_fold.py',candidate],directory/'verify.log',args.timeout,env);ver=fitdir/'candidate_verification.json'
   if code or not ver.exists():raise ValueError(f'holdout verification failed or timed out ({code})')
   data=json.loads(ver.read_text());rows=[x for x in data['cycles'] if x['set']=='holdout'];found=[x for x in rows if x['prediction']];status.update({'holdout_total':len(rows),'holdout_cycles':len(found),'status':'completed' if len(found)==len(rows)==4 else 'holdout_unresolved'})
   for key in ('Cmin','Cmax','period'):status['holdout_MAE_'+key]=float(np.mean([abs(r['prediction'][key]-r['reference'][key]) for r in found])) if found else None
  except Exception as e:status['reason']=f'{type(e).__name__}: {e}'
  atomic(status_path,status);return status
 allitems=[i for c in combos for i in accepted[c]];results=[]
 with ThreadPoolExecutor(max_workers=args.workers) as pool:
  pending={pool.submit(fit,item):item for item in allitems}
  while pending:
   done,_=wait(pending,return_when=FIRST_COMPLETED)
   for f in done:
    item=pending.pop(f)
    try:r=f.result()
    except Exception as e:r={'case':item['case'],'kind':item['kind'],'seed':item['seed'],'status':'worker_error','reason':str(e)}
    results.append(r);print('fit',r['case'],r['kind'],r['seed'],r['status'],'loss',r.get('training_loss'),flush=True)
 fields=['case','kind','seed','status','training_loss','holdout_cycles','holdout_total','holdout_MAE_Cmin','holdout_MAE_Cmax','holdout_MAE_period','optimizer_message','reason']
 with (out/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(sorted(results,key=lambda r:(r['case'],r['kind'],r['seed'])))
 missing={f'{c}_{k}':args.valid_per_combo-len(accepted[(c,k)]) for c,k in combos if len(accepted[(c,k)])<args.valid_per_combo}
 atomic(out/'batch_result.json',{'attempted':{f'{c}_{k}':nextid[(c,k)] for c,k in combos},'accepted':{f'{c}_{k}':len(accepted[(c,k)]) for c,k in combos},'missing':missing,'fit_statuses':{s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})}})
 print('Batch finished; inspect',out/'summary.csv','missing fold quotas:',missing,flush=True)
if __name__=='__main__':main()
