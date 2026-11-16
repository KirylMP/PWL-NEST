"""Optimize cycle extrema, period and shape while solving a fixed-I fold.

This is a numerical fixed-itinerary method. Finite-search failure is reported,
not evidence of nonexistence. Candidate output requires an independent audit.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root,least_squares

ROOT=Path(__file__).resolve().parents[1]
ROUTES={'both':((1,2,1,0),(0,1,1,0,0)), 'upper':((2,1),(1,1,1)), 'lower':((1,0),(0,0,0))}

def flow(A,b,y,t):
    """Analytic affine 2x2 flow, scalar or vector time."""
    t=np.asarray(t,dtype=float);tr=np.trace(A)/2.;D=A-tr*np.eye(2);d2=tr*tr-np.linalg.det(A)
    if d2>1e-14:
        d=np.sqrt(d2);ep=np.exp(np.clip((tr+d)*t,-745,700));em=np.exp(np.clip((tr-d)*t,-745,700));co=(ep+em)/2;si=(ep-em)/(2*d)
    elif d2 < -1e-14:
        d=np.sqrt(-d2);e=np.exp(np.clip(tr*t,-745,700));co=e*np.cos(d*t);si=e*np.sin(d*t)/d
    else:
        e=np.exp(np.clip(tr*t,-745,700));co=e;si=e*t
    eq=np.linalg.solve(A,-b);v=np.asarray(y)-eq;return eq+co[...,None]*v+si[...,None]*(D@v)

class Field:
 def __init__(self,case,kind,candidate):
  self.case=case;self.kind=kind;self.j=candidate;self.cut=np.array(candidate['cuts']);self.Istar=candidate['I_fold'];m=json.loads((ROOT/'results/coordinate_experiment'/f'{case}_{kind}.json').read_text())['model'];self.rest=np.array(m['rest_state']);self.Cref=float(m.get('C_ref_uM',.05));u0=self.rest[0];old=json.loads((ROOT/'paper/data/random_selected'/case/'candidate.json').read_text());self.I1,self.I2=old['fixed_equilibrium_threshold_I'];self.K=np.array([[1,0,max(u0-self.cut[0],0)],[1,self.I1,0],[1,self.I2,self.cut[1]-self.cut[0]]]);self.Z=np.array([u0,*self.cut]);self.u0=u0
  n=np.array(candidate['n']);h=np.array(candidate['h']);self.p0=np.r_[n[1:],n[1]-h[1],n[4]-h[4],np.log(candidate['a']),np.log(candidate['b'])];self.fold0=np.log(candidate['times']);self.scale=np.maximum(abs(self.p0),.25)
 def coeff(self,p):
  n=np.r_[self.rest[1]-p[0]*self.u0,p[:4]];d1,d2=p[4:6];v=np.linalg.solve(self.K,-d1*self.Z);h=n-np.array([v[0],d1,v[1],v[2],d2]);return n,h,np.exp(p[6]),np.exp(p[7])
 def matrices(self,p,I):
  n,h,a,b=self.coeff(p);out=[]
  for r in range(3):
   ni=n[0]+n[2]*I-n[3:3+r]@self.cut[:r];hi=h[0]+h[2]*I-h[3:3+r]@self.cut[:r]
   A=np.array([[-a*(n[1]+sum(n[3:3+r])),a],[b*(h[1]+sum(h[3:3+r])),-b]]);v=np.array([-a*ni,b*hi]);out.append((A,v))
  return out
 def f(self,p,I,y):
  mats=self.matrices(p,I);r=np.searchsorted(self.cut,y[0],side='right');A,b=mats[r];return A@y+b
 def route(self,p,I,logs,name,details=False):
  regions,bounds=ROUTES[name];T=np.exp(np.clip(logs,-15,5));mats=self.matrices(p,I);qs=[];ends=[];starts=[];speeds=[]
  for k,r in enumerate(regions):
   A,b=mats[r];u0=self.cut[bounds[k]];u1=self.cut[bounds[k+1]];y0=flow(A,b,[u0,0],T[k]);y1=flow(A,b,[u0,1],T[k]);den=y1[0]-y0[0]
   if abs(den)<1e-12:raise ValueError('singular flow')
   q=(u1-y0[0])/den;end=flow(A,b,[u0,q],T[k]);qs.append(q);ends.append(end[1]);starts.append([u0,q]);speeds.append((A@starts[-1]+b)[0])
  mismatch=np.array(ends)-np.roll(qs,-1);logM=sum(np.trace(mats[r][0])*T[k] for k,r in enumerate(regions))
  if not np.isfinite(mismatch).all() or not np.isfinite(logM):raise ValueError('nonfinite')
  if not details:return mismatch,logM
  return mismatch,logM,T,mats,np.array(starts),np.array(speeds)
 def inspect(self,p,I,logs,name,need_shape=True):
  residual,logM,T,mats,starts,speed=self.route(p,I,logs,name,True);regions,bounds=ROUTES[name]
  if np.max(abs(residual))>1e-6:raise ValueError('cycle matching')
  if logM>=-1e-7:raise ValueError('unstable cycle')
  segments=[];peaks=[];samples=[];clock=0.
  for k,r in enumerate(regions):
   A,b=mats[r];dt=np.linspace(0,T[k],41);u=flow(A,b,starts[k],dt)[:,0];lower=-np.inf if r==0 else self.cut[r-1];upper=np.inf if r==2 else self.cut[r]
   if min(u)<lower-1e-7 or max(u)>upper+1e-7:raise ValueError('itinerary')
   if (1 if k<len(T)/2 else -1)*speed[k]<=1e-8:raise ValueError('direction')
   segments.append((clock,clock+T[k],A,b,starts[k]));peaks.extend([(val,clock+dt0) for val,dt0 in zip(u,dt)]);samples.extend(u);clock+=T[k]
  if self.kind=='square' and min(samples)<=0:raise ValueError('square branch crosses zero')
  if max(samples)-min(samples)<1e-5:raise ValueError('trivial')
  def cv(u):
   if self.kind=='log': return np.exp(u)
   if self.kind=='square': return u*u
   return self.Cref*np.logaddexp(0.,u)
  peak=max(peaks,key=lambda a:a[0])[1];period=sum(T);grid=(peak+period*np.arange(64)/64)%period;vals=np.empty(64)
  for start,end,A,b,y in segments:
   mask=(grid>=start)&(grid<end)
   if np.any(mask):vals[mask]=cv(flow(A,b,y,grid[mask]-start)[:,0])
  dense=cv(np.asarray(samples));return {'I':float(I),'route':name,'times':T.tolist(),'Cmin':float(np.min(dense)),'Cmax':float(np.max(dense)),'period':float(period),'shape':vals.tolist(),'logM':float(logM),'umin':float(min(samples))}

def initial_cycles(field,rows):
 seeds=[]
 for row in rows:
  I=row['I'];mats=field.matrices(field.p0,I)
  def fun(t,y):
   A,b=mats[np.searchsorted(field.cut,y[0],side='right')];return A@y+b
  def make_event(bound,direction):
   def ev(t,y):return y[0]-bound
   ev.direction=direction;return ev
  events=[make_event(field.cut[0],1),make_event(field.cut[0],-1),make_event(field.cut[1],1),make_event(field.cut[1],-1)]
  sol=solve_ivp(fun,(0,600),field.rest,events=events,method='DOP853',rtol=1e-8,atol=1e-10,max_step=.2)
  if not sol.success:raise ValueError(f'initial simulation failed at {I}')
  evs=sorted([(float(t),i) for i,ts in enumerate(sol.t_events) for t in ts if t>250]);found=None
  patterns=[('both',[0,2,3,1,0]),('upper',[2,3,2]),('lower',[0,1,0])]
  for name,pat in patterns:
   for ix in range(len(evs)-len(pat),-1,-1):
    seq=evs[ix:ix+len(pat)]
    if [v[1] for v in seq]==pat:
     durations=np.diff([v[0] for v in seq])
     try:field.inspect(field.p0,I,np.log(durations),name);found={'I':I,'route':name,'logs':np.log(durations).tolist()};break
     except ValueError:pass
   if found:break
  if found is None:raise ValueError(f'no usable first-crossing itinerary for I={I}')
  seeds.append(found)
 return seeds

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',choices=['AM','AFM','FM'],required=True);p.add_argument('--kind',choices=['log','square','softplus'],required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-nfev',type=int,default=30);args=p.parse_args()
 j=json.loads(args.candidate.read_text());F=Field(args.case,args.kind,j);data=json.loads((ROOT/'results/coordinate_experiment'/f'{args.case}_{args.kind}.json').read_text())['results'];training=[r for r in data if r['set']=='train'];print('finding starting cycle routes',flush=True);seeds=initial_cycles(F,training);print('routes',[(x['route'],x['I']) for x in seeds],flush=True)
 # Choose a locally independent parameter eliminated by fold equations.
 def fold_res(pp,ft):
  match,lm=F.route(pp,F.Istar,ft,'both');return np.r_[match,lm]
 eps=1e-5;J=np.column_stack([(fold_res(F.p0,F.fold0+eps*np.eye(4)[i])-fold_res(F.p0,F.fold0-eps*np.eye(4)[i]))/(2*eps) for i in range(4)])
 options=[]
 for i in range(8):
  dp=np.eye(8)[i]*F.scale[i]*eps;v=(fold_res(F.p0+dp,F.fold0)-fold_res(F.p0-dp,F.fold0))/(2*eps);options.append(np.linalg.svd(np.column_stack((v,J)),compute_uv=False)[-1])
 pivot=int(np.argmax(options));free=np.array([i for i in range(8) if i!=pivot]);print('pivot',pivot,'jacobian min singular',options[pivot],flush=True)
 if options[pivot]<1e-10:raise ValueError('fold pivot numerically singular')
 baseline_fold=np.r_[F.p0[pivot],F.fold0];nres=7*(3+64);best=None;calls=0;failures={};args.output.mkdir(parents=True,exist_ok=True)
 def evaluate(x):
  nonlocal best,calls
  calls+=1;pp=F.p0.copy();pp[free]=F.p0[free]+x*F.scale[free]
  def ff(y):
   pp[pivot]=y[0]
   try:return fold_res(pp,y[1:])
   except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):return np.full(5,1e6)
  try:
   fsol=root(ff,baseline_fold,tol=1e-9);fold=ff(fsol.x)
   if max(abs(fold))>1e-7 or max(fsol.x[1:])>np.log(120) or min(fsol.x[1:])<np.log(1e-6):raise ValueError('fold_unresolved')
   if abs((fsol.x[0]-F.p0[pivot])/F.scale[pivot])>.5 or max(abs(fsol.x[1:]-F.fold0))>1.1:raise ValueError('fold_branch_jump')
   n,h,rate_a,rate_b=F.coeff(pp); mats=F.matrices(pp,F.Istar);T=np.array([np.trace(m[0]) for m in mats]);D=np.array([np.linalg.det(m[0]) for m in mats]);disc=T*T-4*D
   if not (T[0]<0 and D[0]>0 and T[1]>0 and disc[1]<-1e-6 and T[2]<0 and disc[2]<-1e-6):raise ValueError('spectral_violation')
   _,_,ft,fm,fs,sp=F.route(pp,F.Istar,fsol.x[1:],'both',True)
   if min(sp*np.array([1,1,-1,-1]))<1e-7:raise ValueError('fold_direction')
   for nreg,r in enumerate(ROUTES['both'][0]):
    A,b=fm[r];u=flow(A,b,fs[nreg],np.linspace(0,ft[nreg],41))[:,0];lo=-np.inf if r==0 else F.cut[r-1];hi=np.inf if r==2 else F.cut[r]
    if min(u)<lo-1e-7 or max(u)>hi+1e-7:raise ValueError('fold_itinerary')
   residual=[];predictions=[]
   for row,seed in zip(training,seeds):
    I=row['I'];name=seed['route'];guess=np.array(seed['logs']);z=root(lambda y:F.route(pp,I,y,name)[0],guess,tol=1e-8)
    if not np.isfinite(z.x).all() or max(abs(z.x-guess))>3:raise ValueError('cycle_unresolved')
    v=F.inspect(pp,I,z.x,name);target=row['reference'];shape=np.array(v['shape']);ref=np.asarray(row['target_C'])[::4]
    if np.ptp(shape)<1e-8 or np.ptp(ref)<1e-8:raise ValueError('shape_degenerate')
    residual.extend([(v['Cmin']-target['Cmin'])/.03,(v['Cmax']-target['Cmax'])/.08,(v['period']-target['period'])/5]);residual.extend(((shape-shape.min())/np.ptp(shape)-(ref-ref.min())/np.ptp(ref))/(.1*8));predictions.append(v)
   residual=np.array(residual);loss=float(residual@residual)
   if best is None or loss<best['loss']:
    best={'case':args.case,'kind':args.kind,'loss':loss,'n':n.tolist(),'h':h.tolist(),'a':float(rate_a),'b':float(rate_b),'cuts':F.cut.tolist(),'times':ft.tolist(),'I_fold':F.Istar,'matching_max':float(max(abs(fold[:4]))),'logM':float(fold[4]),'speeds':sp.tolist(),'trace':T.tolist(),'determinant':D.tolist(),'discriminant':disc.tolist(),'cycles':predictions,'pivot_index':pivot,'calls':calls,'scope':'numerical fixed-fold root, sampled route checks; requires independent validation'}
    (args.output/'candidate.json').write_text(json.dumps(best,indent=2,allow_nan=False)+'\n');print('call',calls,'loss',loss,flush=True)
   return residual
  except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as e:
   key=str(e);failures[key]=failures.get(key,0)+1;return np.full(nres,100.)+.00001*np.resize(x,nres)
 initial=evaluate(np.zeros(7));
 if best is None:raise RuntimeError(f'initial candidate not admissible: {failures}')
 opt=least_squares(evaluate,np.zeros(7),bounds=(-np.ones(7),np.ones(7)),diff_step=1e-3,max_nfev=args.max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-5,x_scale='jac')
 result={'case':args.case,'kind':args.kind,'calls':calls,'nfev':int(opt.nfev),'message':opt.message,'success':bool(opt.success),'best_loss':best['loss'],'failures':failures,'cycle_seeds':seeds,'source':str(args.candidate)}
 (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('finished',json.dumps(result),flush=True)
if __name__=='__main__':main()
