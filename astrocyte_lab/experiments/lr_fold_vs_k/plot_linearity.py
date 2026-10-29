#!/usr/bin/env python3
"""Plot continued fold amplitudes and residuals against a linear fit."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
upper=json.loads((P/'results_upper/fold_vs_k.json').read_text());lower_path=P/'results_lower_small/fold_vs_k.json'
lower=json.loads(lower_path.read_text())['rows'] if lower_path.exists() else []
rows=sorted([r for r in lower if r['K']<.05-1e-8]+upper['rows'],key=lambda r:r['K'])
k=np.array([r['K'] for r in rows]);amp=np.array([r['amplitude'] for r in rows])
fit=np.array([.05-1e-9<=r['K']<=.1+1e-9 and abs(r['K']*1000-round(r['K']*1000))<1e-7 for r in rows])
p=np.polyfit(k[fit],amp[fit],1);res=amp-np.polyval(p,k);r2=1-np.sum(res[fit]**2)/np.sum((amp[fit]-np.mean(amp[fit]))**2)
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
valid=k<=.10200001;axs[0].plot(k[valid],amp[valid],'.-',label='Continued cycles');axs[0].plot(k[~valid],amp[~valid],'x',ms=9,label='Near equilibrium: unresolved');axs[0].plot(k,np.polyval(p,k),'--',label='Linear fit: K=0.05–0.10');axs[0].set_ylabel('Fold amplitude (µM)');axs[0].legend()
axs[1].axhline(0,color='grey',lw=1);axs[1].plot(k[valid],res[valid],'.-');axs[1].plot(k[~valid],res[~valid],'x',ms=9);axs[1].set_ylabel('Amplitude − linear fit (µM)')
for ax in axs:ax.grid(alpha=.3);ax.set_xlabel('K_ER (µM)')
fig.suptitle(f'Li–Rinzel: fold amplitude and linear approximation (R²={r2:.4f})');fig.savefig(P/'results_upper'/'linearity.png',dpi=180)
print('fit slope/intercept',p,'r2',r2,'max residual on fit',max(abs(res[fit])),'rmse',np.sqrt(np.mean(res[fit]**2)))
print('lower rows',len(lower),'upper stop',upper['stop'])
