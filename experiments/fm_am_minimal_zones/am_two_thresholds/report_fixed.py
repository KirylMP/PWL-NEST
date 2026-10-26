"""Summarize the fixed-threshold AM candidate and compare with LR targets."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import fit_am as fit


def main():
    out = fit.P
    model = json.loads((out/'candidate.json').read_text())
    rows = json.loads((out/'candidate_branches.json').read_text())['rows']
    lr = np.genfromtxt(fit.ROOT/'LR_stable_cycles.csv', delimiter=',', names=True,
                       dtype=None, encoding='utf8')

    targets=[]
    for target in fit.TARGETS:
        ref=lr[np.argmin(abs(lr['I']-target['I']))]
        row=min(rows,key=lambda r:abs(r['I']-float(ref['I'])))
        stable=[c for c in row['cycles'] if c['stable']]
        cyc=(min(stable,key=lambda c:(c['Cmin']-float(ref['Cmin']))**2+
                                     (c['Cmax']-float(ref['Cmax']))**2)
             if stable else None)
        targets.append(dict(I=float(ref['I']),reference=dict(T=float(ref['T']),
            Cmin=float(ref['Cmin']),Cmax=float(ref['Cmax'])),cycle=cyc))
    matched=[q for q in targets if q['cycle'] is not None]
    m=fit.unpack(np.array(model['parameters'])); independent=[]
    for item in matched:
        q=item['cycle']; I=item['I']; y0=np.array([q['section'],q['s']])
        def rhs(t,y):
            return m['co']@np.r_[1,y[0],y[1],I,np.maximum(y[0]-m['padded'],0)]
        tiny=1e-6
        initial=solve_ivp(rhs,[0,tiny],y0,method='DOP853',rtol=1e-11,atol=1e-13).y[:,-1]
        def crossing(t,y):return y[0]-q['section']
        crossing.direction=1;crossing.terminal=True
        sol=solve_ivp(rhs,[tiny,1.2*q['T']],initial,events=crossing,
                      method='DOP853',rtol=2e-10,atol=1e-12,max_step=.05)
        returned=len(sol.t_events[0])>0
        independent.append(dict(I=I,returned=returned,
            period_error=float(sol.t_events[0][0]-q['T']) if returned else None,
            closure_error=float(sol.y_events[0][0,1]-q['s']) if returned else None))
    errors=np.array([[q['cycle']['Cmin']-q['reference']['Cmin'],
                      q['cycle']['Cmax']-q['reference']['Cmax'],
                      q['cycle']['T']-q['reference']['T'],
                      (q['cycle']['Cmax']-q['cycle']['Cmin'])-
                      (q['reference']['Cmax']-q['reference']['Cmin'])]
                     for q in matched])
    summary=dict(candidate='candidate.json',loss=model.get('loss'),
        physical_thresholds_C=model['physical_thresholds_C'],
        thresholds_optimized=model.get('thresholds_optimized'),
        equilibrium_threshold_I=model['equilibrium_threshold_I'],
        regions=3,training_orbits=len(fit.TARGETS),matched_training_orbits=len(matched),
        maximum_detected_cycles=max(len(r['cycles']) for r in rows),
        RMSE_Cmin_Cmax=float(np.sqrt(np.mean(errors[:,:2]**2))),
        RMSE_period=float(np.sqrt(np.mean(errors[:,2]**2))),
        RMSE_amplitude=float(np.sqrt(np.mean(errors[:,3]**2))),
        independent_DOP853_checks=independent,
        counts_method='finite I grid; cycle roots from native Poincare return map')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,
                         'grid.alpha':.18})
    fig,axs=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    for key in ('Cmin','Cmax'):
        axs[0].scatter(lr['I'],lr[key],s=9,color='#9366b5',alpha=.65,
                       label='Li–Rinzel' if key=='Cmin' else None)
    stable=[(r['I'],c) for r in rows for c in r['cycles'] if c['stable']]
    unstable=[(r['I'],c) for r in rows for c in r['cycles'] if not c['stable']]
    for key in ('Cmin','Cmax'):
        axs[0].scatter([i for i,c in stable],[c[key] for i,c in stable],s=11,
            color='#111111',label='AM candidate' if key=='Cmin' else None)
    axs[0].axhline(fit.FIXED_TC[0],color='#168aad',ls='--',lw=.8)
    axs[0].axhline(fit.FIXED_TC[1],color='#168aad',ls='--',lw=.8,
                   label='fixed C thresholds')
    axs[0].set(xlabel='I',ylabel='C (µM)',title='Cycle extrema')
    axs[0].legend(fontsize=8)
    for dataset,color,label,filled in [(lr,'#9366b5','Li–Rinzel',True),
                                       (stable,'#111111','AM stable',True),
                                       (unstable,'#d97706','AM unstable',False)]:
        if dataset is lr:
            axs[1].scatter(lr['I'],lr['T'],s=9,color=color,label=label)
            axs[2].scatter(lr['I'],lr['Cmax']-lr['Cmin'],s=9,color=color,label=label)
        else:
            axs[1].scatter([i for i,c in dataset],[c['T'] for i,c in dataset],s=11,
                facecolors=color if filled else 'none',edgecolors=color,label=label)
            axs[2].scatter([i for i,c in dataset],[c['Cmax']-c['Cmin'] for i,c in dataset],s=11,
                facecolors=color if filled else 'none',edgecolors=color,label=label)
    axs[1].set(xlabel='I',ylabel='Period (s)',title='Cycle periods')
    axs[2].set(xlabel='I',ylabel='Cmax − Cmin (µM)',title='Cycle amplitude')
    axs[1].legend(fontsize=8);axs[2].legend(fontsize=8)
    fig.suptitle('AM fixed-threshold candidate vs Li–Rinzel')
    fig.savefig(out/'comparison_fixed_thresholds.png',dpi=180)
    fig.savefig(out/'comparison_fixed_thresholds.pdf')
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
