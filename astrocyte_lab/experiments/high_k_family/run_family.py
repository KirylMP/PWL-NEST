#!/usr/bin/env python3
"""Run high-K independent fits; fit selected coefficient families afterward."""
import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]

def approx(t,y,form):
 if form=='linear':
  v=np.polyfit(t,y,1);return v,lambda x:np.polyval(v,x)
 if form=='quadratic':
  v=np.polyfit(t,y,2);return v,lambda x:np.polyval(v,x)
 if form=='reciprocal':
  solutions=[]
  for sign in (-1,1):
   def pred(v,x):return v[0]+v[1]/(x+sign*(1.02+np.exp(v[2])))
   sol=least_squares(lambda v:pred(v,t)-y,[float(np.mean(y)),float(np.ptp(y)+1e-5),1.],max_nfev=3000,bounds=([-1e4,-1e4,-4],[1e4,1e4,6]))
   solutions.append((np.linalg.norm(sol.fun),sign,sol.x))
  _,sign,v=min(solutions,key=lambda q:q[0]);return np.r_[v,sign],lambda x:v[0]+v[1]/(x+sign*(1.02+np.exp(v[2])))
 if form=='sigmoid':
  sols=[]
  for center in (-1,0,1):
   sol=least_squares(lambda v:v[0]+v[1]*expit(v[2]*(t-v[3]))-y,[np.mean(y),np.ptp(y)+1e-4,2,center],bounds=([-1e4,-1e4,.01,-10],[1e4,1e4,40,10]),max_nfev=4000);sols.append(sol)
  v=min(sols,key=lambda s:np.linalg.norm(s.fun)).x;return v,lambda x:v[0]+v[1]*expit(v[2]*(x-v[3]))
 raise ValueError(form)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--values',type=float,nargs='+',default=list(np.linspace(.01,.11,21)));p.add_argument('--workers',type=int,default=11);p.add_argument('--starts',type=int,default=4);p.add_argument('--max-nfev',type=int,default=1000);p.add_argument('--duration',type=float,default=400);p.add_argument('--dt',type=float,default=.02);p.add_argument('--input-count',type=int,default=7);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 engine=a.out/'trajectory_engine.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(ROOT/'astrocyte_lab/scripts/physical_trajectory_engine.cpp'),'-o',str(engine)],check=True)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(a.out/'mpl'),XDG_CACHE_HOME=str(a.out/'cache'))
 def job(k):
  d=a.out/f'K_{k:.5f}';d.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,'-u',str(HERE/'fit_high_k.py'),'--K',str(k),'--out',str(d),'--engine',str(engine.resolve()),'--starts',str(a.starts),'--max-nfev',str(a.max_nfev),'--duration',str(a.duration),'--dt',str(a.dt),'--input-count',str(a.input_count)]
  with (d/'fit.log').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
  f=d/'parameters.json';r=json.loads(f.read_text()) if f.exists() else dict(K=k,status='process_error',log=str(d/'fit.log'));print(f"K={k:.5f} status={r['status']} exit={p.returncode} log={d/'fit.log'}",flush=True);return r
 print(f'{len(a.values)} independent fits, workers={a.workers}, output={a.out}',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(job,a.values))
 good=sorted([r for r in rows if r['status']=='fit_candidate'],key=lambda r:r['K']); summary=dict(rows=rows,warning='Candidates checked against an independent solver on training inputs, but holdout accuracy and global cycle structure are unverified. No fold or unstable-cycle constraint. Functional fits are post hoc and do not preserve exact equilibrium events or stability between fitted K values.')
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 if len(good)<5:print('Fewer than 5 fit candidates; coefficient fitting skipped',flush=True);return
 names=[('alpha',0,'linear'),('alpha',1,'reciprocal'),('alpha',2,'sigmoid'),('alpha',3,'linear'),('alpha',4,'quadratic'),('beta',0,'linear'),('beta',1,'reciprocal'),('beta',2,'linear'),('beta',3,'reciprocal'),('beta',4,'quadratic')]
 fig,axs=plt.subplots(2,5,figsize=(18,7),layout='constrained');k=np.array([r['K'] for r in good]);t=(k-.06)/.05;grid=np.linspace(min(t),max(t),200); report=[]
 for ax,(key,index,form) in zip(axs.flat,names):
  y=np.array([r[key][index] for r in good]);coef,pred=approx(t,y,form);res=y-pred(t);loo=[]
  for j in range(len(t)):
   try:
    mask=np.arange(len(t))!=j;_,f=approx(t[mask],y[mask],form);loo.append(float(f(t[j])-y[j]))
   except Exception:loo.append(None)
  goodloo=[e for e in loo if e is not None and np.isfinite(e)]; report.append(dict(name=f'{key}[{index}]',family=form,coefficients=coef.tolist(),K_normalization='t=(K-0.06)/0.05',train_rmse=float(np.sqrt(np.mean(res**2))),max_abs_error=float(max(abs(res))),loo_rmse=float(np.sqrt(np.mean(np.square(goodloo)))) if goodloo else None,leave_one_out_errors=loo,quadratic_opens_down=bool(coef[0]<0) if form=='quadratic' else None))
  ax.plot(k,y,'o',label='models');ax.plot(.06+.05*grid,pred(grid),'-',label=form);ax.set_title(f'{key}[{index}]');ax.set_xlabel('K_ER (µM)');ax.grid(alpha=.25);ax.legend(fontsize=7)
 fig.suptitle('Coefficient approximations; finite training candidates');fig.savefig(a.out/'coefficient_fits.png',dpi=160)
 (a.out/'coefficient_fits.json').write_text(json.dumps(dict(values_K=k.tolist(),fits=report,formulas={'linear':'a*t+b','quadratic':'a*t^2+b*t+c','reciprocal':'a+b/(t+sign*(1.02+exp(c)))','sigmoid':'a+b*sigmoid(s*(t-center))'},warning=summary['warning']),indent=2)+'\n');print('SAVED',a.out,flush=True)
if __name__=='__main__':main()
