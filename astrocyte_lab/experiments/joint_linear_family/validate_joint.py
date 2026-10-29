#!/usr/bin/env python3
"""Independent DOP853 holdout validation at previously unseen K_ER and I."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/high_k_family'))
from fit_high_k import configure
from fit_k import lr_rhs

def cycle(rhs,initial,duration):
 def peak(t,y):return rhs(t,y)[0]
 peak.direction=-1;peak.terminal=False
 def trough(t,y):return rhs(t,y)[0]
 trough.direction=1;trough.terminal=False
 s=solve_ivp(rhs,(0,duration),initial,method='DOP853',events=[peak,trough],rtol=2e-9,atol=2e-11,max_step=.15)
 a=s.t_events[0]>.5*duration;b=s.t_events[1]>.5*duration
 if not s.success or sum(a)<3 or sum(b)<2:return None
 pp=s.t_events[0][a];p=s.y_events[0][a,0];q=s.y_events[1][b,0];d=np.diff(pp)
 if p[-1]-q[-1]<.001 or abs(d[-1]-d[-2])>max(.3,.03*d[-1]) or abs(p[-1]-p[-2])>.005:return None
 return dict(Cmin=float(q[-1]),Cmax=float(p[-1]),period=float(d[-1]),peaks=int(len(pp)))

def main():
 p=argparse.ArgumentParser();p.add_argument('family',type=Path);p.add_argument('--K',type=float,nargs='+',default=[.0525,.0775,.1025]);p.add_argument('--duration',type=float,default=400.);a=p.parse_args();data=json.loads(a.family.read_text());left=np.array(data['left']);right=np.array(data['right']);out=[]
 for k in a.K:
  if not data['K_range'][0]<k<data['K_range'][1]:raise ValueError('Held-out K must be inside training interval')
  model,fixed,_=configure(k)
  if not fixed:raise ValueError('Cannot enforce LR lower event at held-out K')
  u=(k-.045)/.065;x=np.r_[(1-u)*left+u*right,0.];m=model.unpack(x);q=model.basis(model.cuts,model.events,m)@m['N']
  for frac in (.25,.5,.75):
   I=float(model.events[0]+frac*(model.events[1]-model.events[0]));reference=None;initial=None
   for y0 in (model.rest.tolist(),[.7,.8]):
    reference=cycle(lambda t,y:lr_rhs(y[0],y[1],I,k),y0,a.duration)
    if reference is not None:initial=y0;break
   pred=None
   if reference:
    def rhs(t,y):
     B=model.basis(y[0],I,m);return [m['a']*(y[1]-B@m['N']),m['b']*(B@m['H']-y[1])]
    pred=cycle(rhs,initial,a.duration)
   entry=dict(K=k,I=I,initial=initial,LR=reference,PWA=pred,event_q=q.tolist(),status='ok' if reference and pred else ('lr_unresolved' if reference is None else 'pwa_unresolved'))
   if reference and pred:entry['error']={name:float(pred[name]-reference[name]) for name in ('Cmin','Cmax','period')}
   out.append(entry);print(f"K={k:.5f} I={I:.6f} {entry['status']} error={entry.get('error')}",flush=True)
 path=a.family.parent/'independent_holdout.json';path.write_text(json.dumps(dict(method='DOP853 independent unseen K and I, finite trajectories from same initial states; not a cycle completeness proof.',rows=out),indent=2)+'\n');print('SAVED',path,flush=True)
if __name__=='__main__':main()
