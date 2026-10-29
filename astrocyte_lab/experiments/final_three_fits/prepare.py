#!/usr/bin/env python3
"""Extract phase-aligned LR stable waveforms from the project reference cases."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/high_k_family'))
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/phenomenological_k_fit'))
from fit_k import lr_rhs

def main():
 p=argparse.ArgumentParser();p.add_argument('--K',type=float,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--duration',type=float,default=600);p.add_argument('--phase-count',type=int,default=64);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 reference=json.loads((ROOT/'astrocyte_lab/experiments/joint_linear_family/reference.json').read_text())['models'];row=min(reference,key=lambda x:abs(x['K']-a.K))
 if abs(row['K']-a.K)>1e-8:raise ValueError('Missing reference K')
 foldrows=json.loads((ROOT/'astrocyte_lab/experiments/phenomenological_k_fit/lr_fold_targets.json').read_text())['rows'];fold=min(foldrows,key=lambda x:abs(x['K']-a.K))
 if abs(fold['K']-a.K)>1e-8 or fold['nfev']>=500:raise ValueError('Unresolved LR fold')
 results=[]
 for j,q in enumerate(row['reference_cycles']):
  I=q['I']
  def field(t,y):return lr_rhs(y[0],y[1],I,a.K)
  def peak(t,y):return field(t,y)[0]
  peak.direction=-1;peak.terminal=False
  sol=solve_ivp(field,(0,a.duration),q['initial'],events=peak,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.2,dense_output=True)
  ts=sol.t_events[0];ts=ts[ts>a.duration*.5]
  if not sol.success or len(ts)<3:raise ValueError(f'LR waveform unresolved I={I}')
  t0,t1=ts[-2:];period=float(t1-t0);phase=np.arange(a.phase_count)/a.phase_count;shape=sol.sol(t0+period*phase)[0]
  if max(shape)-min(shape)<.001 or abs(period-q['period'])>max(.5,.05*q['period']):raise ValueError(f'LR waveform not converged I={I}')
  results.append(dict(I=I,initial=q['initial'],period=period,Cmin=float(min(shape)),Cmax=float(max(shape)),shape=shape.tolist(),split='holdout' if j%3==1 else 'train'))
  print(f'REFERENCE K={a.K:.3f} {j+1}/{len(row["reference_cycles"])} I={I:.6f} T={period:.4f}',flush=True)
 (a.out/'reference.json').write_text(json.dumps(dict(K=a.K,fold=fold,phase_count=a.phase_count,shape_phase='C peak at phase zero, uniformly spaced normalized time',cycles=results),indent=2)+'\n')
 print('SAVED',a.out/'reference.json',flush=True)
if __name__=='__main__':main()
