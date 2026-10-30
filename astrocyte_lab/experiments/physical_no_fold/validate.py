#!/usr/bin/env python3
"""Independent DOP853 stable-cycle validation; no fold analysis."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/mixed_three_fits'))
from physical_model import PhysicalModel as MixedModel,phi,inverse

def main():
 p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--duration',type=float,default=450);a=p.parse_args();d=json.loads(a.candidate.read_text());original=json.loads((ROOT/f'astrocyte_lab/experiments/no_fold_three_fits/seeds/K_{d["K"]:.3f}_validation.json').read_text());model=MixedModel(d['K']);raw=np.zeros(model.size);raw[:min(len(d['parameters']),model.size)]=d['parameters'][:model.size];m=model.unpack(raw);results=[]
 for old in original['cycles']:
  ref=old['LR'];I=ref['I'];initial=ref['initial']
  def rhs(t,y):return m['co']@np.r_[1.,y[0],y[1],I,np.maximum(y[0]-m['padded'],0.)]
  def peak(t,y):return rhs(t,y)[0]
  peak.direction=-1;peak.terminal=False
  s=solve_ivp(rhs,(0,a.duration),[float(phi(initial[0],m['shape'])),initial[1]],events=peak,method='DOP853',rtol=2e-9,atol=2e-11,max_step=.08,dense_output=True);ts=s.t_events[0];ts=ts[ts>a.duration*.5];out=None
  zmin=float(phi(1e-12,m['shape']));zmax=float(phi(model.CMAX,m['shape']))
  if s.success and len(ts)>=3 and np.isfinite(s.y).all() and np.min(s.y[0])>zmin and np.max(s.y[0])<zmax:
   period=float(ts[-1]-ts[-2]);states=s.sol(ts[-2]+period*np.arange(len(ref['shape']))/len(ref['shape']))
   try:
    C=inverse(states[0],m['shape'])
    if max(C)-min(C)>.001 and abs((ts[-1]-ts[-2])-(ts[-2]-ts[-3]))<max(.3,.03*period):out=dict(Cmin=float(min(C)),Cmax=float(max(C)),period=period,shape=C.tolist(),qmin=float(np.min(states[1])),qmax=float(np.max(states[1])))
   except ValueError:pass
  row=dict(I=I,split=old['split'],LR=ref,PWA=out)
  if out:row['error']=dict(Cmin=out['Cmin']-ref['Cmin'],Cmax=out['Cmax']-ref['Cmax'],period=out['period']-ref['period'],shape_rmse=float(np.sqrt(np.mean((np.asarray(out['shape'])-ref['shape'])**2))))
  results.append(row);print('VALIDATE I=',I,'valid=',out is not None,flush=True)
 path=a.candidate.parent/'validation.json';path.write_text(json.dumps(dict(scope='Independent DOP853 finite stable trajectories; hidden q unrestricted; no cycle count or fold analysis.',cycles=results),indent=2)+'\n')
 fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained');valid=[r for r in results if r['PWA']]
 for key in ('Cmin','Cmax'):
  ax[0].plot([r['I'] for r in results],[r['LR'][key] for r in results],'o-',label='LR '+key);ax[0].plot([r['I'] for r in valid],[r['PWA'][key] for r in valid],'x--',label='PWA '+key)
 ax[1].plot([r['I'] for r in results],[r['LR']['period'] for r in results],'o-',label='LR');ax[1].plot([r['I'] for r in valid],[r['PWA']['period'] for r in valid],'x--',label='PWA')
 if valid:
  r=valid[len(valid)//2];phase=np.arange(len(r['LR']['shape']))/len(r['LR']['shape']);ax[2].plot(phase,r['LR']['shape'],label='LR');ax[2].plot(phase,r['PWA']['shape'],label='PWA');ax[2].set_title(f'I={r["I"]:.3f}')
 for item in ax:item.grid(alpha=.25);item.legend(fontsize=7)
 ax[0].set_ylabel('C (µM)');ax[1].set_ylabel('period (s)');ax[2].set_ylabel('C (µM)');ax[0].set_xlabel('I (µM)');ax[1].set_xlabel('I (µM)');ax[2].set_xlabel('normalized phase');fig.suptitle(f'K_ER={d["K"]:.3f}: stable trajectories');fig.savefig(a.candidate.parent/'comparison.png',dpi=150)
 print('SAVED',path,flush=True)
if __name__=='__main__':main()
