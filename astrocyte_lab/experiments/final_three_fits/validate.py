#!/usr/bin/env python3
"""Independent cycles and local bifurcation checks for a fitted candidate."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/high_k_family'))
from fit_high_k import configure
from fold import BoundaryFold

def main():
 p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--reference',type=Path,required=True);p.add_argument('--duration',type=float,default=450);p.add_argument('--offset',type=float,default=.001);a=p.parse_args();data=json.loads(a.candidate.read_text());ref=json.loads(a.reference.read_text());model,fixed,_=configure(data['K']);x=np.array(data['parameters']);m=model.unpack(x);results=[]
 def run(I,y0):
  def rhs(t,y):
   B=model.basis(y[0],I,m);return [m['a']*(y[1]-B@m['N']),m['b']*(B@m['H']-y[1])]
  def peak(t,y):return rhs(t,y)[0]
  peak.direction=-1;peak.terminal=False
  s=solve_ivp(rhs,(0,a.duration),y0,events=peak,method='DOP853',rtol=2e-9,atol=2e-11,max_step=.08,dense_output=True);ts=s.t_events[0];ts=ts[ts>a.duration*.5]
  if not s.success or len(ts)<3:return None
  t0,t1=ts[-2:];T=float(t1-t0);cs=s.sol(t0+T*np.arange(ref['phase_count'])/ref['phase_count'])[0]
  if max(cs)-min(cs)<.001 or abs((ts[-1]-ts[-2])-(ts[-2]-ts[-3]))>max(.3,.03*T):return None
  return dict(Cmin=float(min(cs)),Cmax=float(max(cs)),period=T,shape=cs.tolist())
 for q in ref['cycles']:
  out=run(q['I'],q['initial']);r=dict(I=q['I'],split=q['split'],LR=q,PWA=out)
  if out:r['error']=dict(Cmin=out['Cmin']-q['Cmin'],Cmax=out['Cmax']-q['Cmax'],period=out['period']-q['period'],shape_rmse=float(np.sqrt(np.mean((np.asarray(out['shape'])-q['shape'])**2))))
  results.append(r);print('VALIDATE I=',q['I'],'valid=',out is not None,flush=True)
 fold=BoundaryFold(model,data['fold_constraint']['I'],first=0,last=1);t=np.array(data['fold_constraint']['crossing_times']);n=len(t);eps=1e-5;eye=np.eye(n);J=np.column_stack([(fold.matching(x,t+eps*e)-fold.matching(x,t-eps*e))/(2*eps) for e in eye]);right=np.linalg.svd(J)[2][-1];branches=[]
 for sign in (-1,1):
  I=data['fold_constraint']['I']+sign*a.offset;fold.I=I;roots=[]
  for shift in [-.4,-.2,-.08,-.03,0,.03,.08,.2,.4]:
   start=np.clip(t+shift*right,1e-5,29.999)
   try:
    sol=least_squares(lambda v:fold.matching(x,v),start,bounds=(np.full(n,1e-6),np.full(n,30.)),max_nfev=400,xtol=1e-11,ftol=1e-11,gtol=1e-11)
    if max(abs(sol.fun))>1e-7 or any(np.linalg.norm(sol.x-r['times'])<.001 for r in roots):continue
    res,mm,z,h,M=fold.matching(x,sol.x,True);states=[];valid=True
    for j,(region,A) in enumerate(zip(fold.regions,M)):
     speed=(A@np.r_[z[j],h[j],1])[0];direction=1 if j<n//2 else -1
     vals=np.array([(expm(A*s)@np.r_[z[j],h[j],1])[:2] for s in np.linspace(0,sol.x[j],60)])
     lo=0 if region==0 else model.cuts[region-1];hi=model.CMAX if region==2 else model.cuts[region]
     valid &= bool(direction*speed>1e-8 and min(vals[:,0])>=lo-1e-6 and max(vals[:,0])<=hi+1e-6 and min(vals[:,1])>0 and max(vals[:,1])<1);states.extend(vals)
    if not valid:continue
    logm=float(sum(mm['T'][region]*dt for region,dt in zip(fold.regions,sol.x)))
    roots.append(dict(times=sol.x.tolist(),period=float(sum(sol.x)),log_multiplier=logm,stable=bool(logm<0),Cmin=float(min(q[0] for q in states)),Cmax=float(max(q[0] for q in states)),matching_error=float(max(abs(sol.fun)))))
   except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):continue
  branches.append(dict(I=I,cycles=roots));print('BRANCH I=',I,'cycles=',len(roots),flush=True)
 path=a.candidate.parent/'validation.json';path.write_text(json.dumps(dict(scope='Independent DOP853 finite stable trajectories plus local affine boundary shooting near fold; no global cycle completeness. Branch roots have sampled region checks, not independent IVP closure.',cycles=results,local_fold_branches=branches),indent=2)+'\n')
 fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');rs=[q for q in results if q['PWA'] is not None]
 for key in ('Cmin','Cmax'):
  axes[0].plot([q['I'] for q in results],[q['LR'][key] for q in results],'o-',label='LR '+key);axes[0].plot([q['I'] for q in rs],[q['PWA'][key] for q in rs],'x--',label='PWA '+key)
 for group in branches:
  for j,orb in enumerate(group['cycles']):
   axes[0].scatter([group['I']]*2,[orb['Cmin'],orb['Cmax']],facecolors='black' if orb['stable'] else 'none',edgecolors='black',marker='o',label=('PWA stable near fold' if orb['stable'] else 'PWA unstable near fold') if j==0 else None,zorder=5)
 axes[0].scatter([ref['fold']['I']]*2,[ref['fold']['C_min'],ref['fold']['C_max']],marker='D',c='purple',label='LR fold',zorder=5)
 fold.I=data['fold_constraint']['I'];_,_,z,h,matrices=fold.matching(x,t,True)
 fold_c=[float((expm(A*s)@np.r_[z[j],h[j],1])[0]) for j,A in enumerate(matrices) for s in np.linspace(0,t[j],200)]
 axes[0].scatter([fold.I]*2,[min(fold_c),max(fold_c)],marker='D',facecolors='none',edgecolors='black',label='PWA fold',zorder=5)
 axes[1].plot([q['I'] for q in results],[q['LR']['period'] for q in results],'o-',label='LR');axes[1].plot([q['I'] for q in rs],[q['PWA']['period'] for q in rs],'x--',label='PWA')
 if rs:
  q=rs[len(rs)//2];phase=np.arange(ref['phase_count'])/ref['phase_count'];axes[2].plot(phase,q['LR']['shape'],label='LR');axes[2].plot(phase,q['PWA']['shape'],label='PWA');axes[2].set_title(f'I={q["I"]:.3f}')
 for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=7)
 axes[0].set_ylabel('C (µM)');axes[1].set_ylabel('period (s)');axes[2].set_ylabel('C (µM)');axes[0].set_xlabel('I (µM)');axes[1].set_xlabel('I (µM)');axes[2].set_xlabel('normalized phase')
 fig.suptitle(f'K_ER={data["K"]:.3f}: independent trajectory checks');fig.savefig(a.candidate.parent/'comparison.png',dpi=150);plt.close(fig);print('SAVED',path,flush=True)
if __name__=='__main__':main()
