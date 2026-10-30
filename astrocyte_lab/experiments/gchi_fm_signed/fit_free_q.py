"""FM fitting with exact anchor projections and hard zone-type constraints."""
import os,json,ctypes,time
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize
P=Path(__file__).resolve().parent;m=json.load(open(P/os.environ.get('FIT_SEED','seed.json')));refs=json.load(open(P/'reference_cycles.json'))['results'];geo=json.load(open(P/'geometry.json'));refA=np.array(geo['A_C_I_h']);refB=np.array(geo['B_C_I_h']);co0=np.array(m['coefficients']);cuts=np.ascontiguousarray(m['thresholds_C']);anc=m['anchors'];c=np.array([x['C'] for x in anc]);q=np.array([x['q'] for x in anc]);i=np.array([x['I'] for x in anc]);g=np.array([x['gamma'] for x in anc])
def U(g,scale):return np.arcsinh(np.asarray(g)/scale)
def projection(q,scale):
 A=np.column_stack([np.ones(len(c)),c,q,i,*[np.maximum(c-t,0) for t in cuts]]);return A,[null_space(A),null_space(A),null_space(np.column_stack([A,U(g,scale)]))]
initial_scale=float(m.get('input_scale',1e-4));_,Ns=projection(q,initial_scale);sizes=[n.shape[1] for n in Ns];ends=np.cumsum([5]+sizes);v0=np.r_[q,np.log(initial_scale),*[n.T@(co0[j,:7] if j<2 else co0[j]) for j,n in enumerate(Ns)]]
def decode(v):
 scale=np.exp(np.clip(v[4],np.log(1e-7),np.log(.1)));_,nn=projection(v[:4],scale);z=[n@v[ends[j]:ends[j+1]] for j,n in enumerate(nn)];return np.ascontiguousarray(np.vstack([np.r_[z[0],0],np.r_[z[1],0],z[2]])),scale
def matb(co,k):
 A=co[:,1:4].copy();b=co[:,0].copy()
 for j in range(k):A[:,0]+=co[:,4+j];b-=co[:,4+j]*cuts[j]
 return A,b
def valid(co):
 try:
  for k,n in enumerate([0,1,2,0]):
   M,_=matb(co,k);e=np.linalg.eigvals(M)
   if any(abs(e.real)<1e-5) or sum(e.real>0)!=n:return False
  return True
 except np.linalg.LinAlgError:return False
lib=ctypes.CDLL(str(P/'shape_checked.so'));ptr=ctypes.POINTER(ctypes.c_double);lib.cycle.argtypes=[ptr,ptr,ctypes.c_double,ptr,ctypes.c_double,ctypes.c_int,ptr];lib.cycle.restype=ctypes.c_int
train=[r for r in refs if r['mode']=='FM' and r['valid'] and r['gamma'] in [.022,.04,.1,.4,1.3,5.]];calls=0;best=float('inf');bestv=None;bestrep=[];start=time.monotonic();last_checkpoint=0.
def fun(v):
 global calls,best,bestv,bestrep,last_checkpoint
 calls+=1;co,scale=decode(v)
 if not valid(co):return 1e6
 try:
  x=np.linalg.solve(*[matb(co,1)[0],-matb(co,1)[1]]);y=np.linalg.solve(*[matb(co,2)[0],-matb(co,2)[1]])
 except np.linalg.LinAlgError:return 1e6
 # Soft zero-input saddle matching and ensure equilibria in corresponding zones.
 geom=sum(((x[j]-refA[k])/.1)**2 for j,k in [(0,0),(2,1)])+sum(((y[j]-refB[k])/.1)**2 for j,k in [(0,0),(2,1)])
 geom+=1000*(max(0,cuts[0]-x[0])**2+max(0,x[0]-cuts[1])**2+max(0,cuts[1]-y[0])**2+max(0,y[0]-cuts[2])**2)
 val=geom;rep=[]
 for r in train:
  z=np.ascontiguousarray(r['last_peak_state'],float);cc,ii=z[0],z[2]
  if abs(co[0,2])<1e-9:val+=30;rep.append({'gamma':r['gamma'],'detected':False});continue
  z[1]=-(co[0,0]+co[0,1]*cc+co[0,3]*ii+sum(co[0,4+j]*max(cc-cuts[j],0) for j in range(3)))/co[0,2]
  
  if r['gamma']==5.0:
   z=np.ascontiguousarray(json.load(open(P/'initial_scan.json'))[-1]['hits'][0]['initial'],float)
  out=np.zeros(5);ok=bool(lib.cycle(co.ctypes.data_as(ptr),cuts.ctypes.data_as(ptr),float(U(r['gamma'],scale)),z.ctypes.data_as(ptr),float(os.environ.get('FIT_DT','0.02')),int(os.environ.get('FIT_STEPS','60000')),out.ctypes.data_as(ptr)))
  if ok:
   tar=np.array([r['Cmin'],r['Cmax'],r['Imin'],r['Imax'],r['period']]);residual_scale=np.array([.12,.35,.3,.7,max(12,r['period'])]);err=(out-tar)/residual_scale;err[4]*=2;val+=float(err@err)/len(train)
  else:val+=60/len(train)
  rep.append({'gamma':r['gamma'],'detected':ok,'values':out.tolist() if ok else None})
 if any(r['detected'] for r in rep) and val<best:
  best=val;bestv=v.copy();bestrep=rep
  if time.monotonic()-last_checkpoint>=15 or not (P/'checkpoint.json').exists():
   last_checkpoint=time.monotonic()
   ck={'vector':bestv.tolist(),'loss':best,'evaluations':calls,'training':bestrep,'note':'optimization checkpoint; resume with FIT_RESUME=checkpoint.json'}
   tmp=P/'checkpoint.tmp';tmp.write_text(json.dumps(ck,indent=2)+'\n');tmp.replace(P/'checkpoint.json')
 if calls%500==0:print('calls',calls,'best',round(best,4),'detected',sum(r['detected'] for r in rep),'elapsed',round(time.monotonic()-start),flush=True)
 return val
if os.environ.get('FIT_RESUME'):
 ck=json.load(open(P/os.environ['FIT_RESUME']));v0=np.array(ck['vector'],float)
 if len(v0)!=ends[-1]:raise ValueError('checkpoint incompatible with current fit')
print('start',fun(v0),flush=True);rng=np.random.default_rng(int(os.environ.get('FIT_RANDOM_SEED','5552')))
for j in range(int(os.environ.get('FIT_RANDOM_STARTS','10000'))):
 v=(bestv if bestv is not None and j%3 else v0).copy();v+=rng.normal(size=len(v))*np.maximum(abs(v0),.05)*[.005,.02,.06,.2][j%4];fun(v)
print('best random',best,flush=True)
if bestv is None:(P/os.environ.get('FIT_STATUS','asinh_free_q_status.json')).write_text(json.dumps({'status':'no_cycles_found','evaluations':calls},indent=2)+'\n');raise SystemExit('No candidate with cycles')
sol=minimize(fun,bestv,method='Powell',options={'maxfev':int(os.environ.get('FIT_MAXFEV','5000')),'xtol':.001,'ftol':1e-4});co,scale=decode(bestv);es=[np.linalg.eigvals(matb(co,k)[0]) for k in range(4)];eq=[np.linalg.solve(matb(co,k)[0],-matb(co,k)[1]) for k in range(4)];A,_=projection(bestv[:4],scale);ar=A@co[:,:7].T+U(g,scale)[:,None]*co[:,7];anc=[dict(a,q=float(bestv[j])) for j,a in enumerate(anc)];result=m|{'description':'FM cycle fit with signed log input; exact rest and three transition anchors; hard spectral counts 0,1,2,0; q anchor values and asinh input scale optimized; finite cycle search','coefficients':co.tolist(),'anchors':anc,'input':'asinh(gamma/scale)','input_scale':float(scale),'zone_eigenvalues':[[{'real':float(z.real),'imag':float(z.imag)} for z in e] for e in es],'gamma_zero_affine_equilibria_by_zone':[x.tolist() for x in eq],'anchor_max_residual':float(np.max(abs(ar))),'training':bestrep,'loss':best,'evaluations':calls}
(P/os.environ.get('FIT_OUTPUT','asinh_free_q_refined.json')).write_text(json.dumps(result,indent=2)+'\n');print('saved',best,bestrep,flush=True)
