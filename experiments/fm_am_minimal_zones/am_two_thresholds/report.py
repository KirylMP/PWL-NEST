"""Compare measured three-zone AM dynamics with source LR and affine fit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fit_am as f
ROOT=f.ROOT

def main():
    selected=sys.argv[1] if len(sys.argv)>1 else 'mixed'
    models={};metrics={}
    for kind in ['affine',selected]:
        f.KIND=kind;p=ROOT/kind;model=json.loads((p/'candidate.json').read_text());m=f.unpack(np.array(model['parameters']));rows=json.loads((p/'candidate_branches.json').read_text())['rows'];models[kind]=(m,rows)
        result=dict(loss=model.get('loss'),cuts_C=m['tc'].tolist(),equilibrium_threshold_I=f.events(m).tolist(),max_detected_cycles=max(len(r['cycles']) for r in rows),I_samples=len(rows),targets=[],independent_checks=[])
        matched=[]
        for q in f.TARGETS:
            row=min(rows,key=lambda r:abs(r['I']-q['I']));assert abs(row['I']-q['I'])<1e-8
            stable=[r for r in row['cycles'] if r['stable']]
            chosen=min(stable,key=lambda r:(r['Cmin']-q['Cmin'])**2+(r['Cmax']-q['Cmax'])**2) if stable else None
            result['targets'].append(dict(I=q['I'],reference=q,cycle=chosen,stable_count=len(stable)))
            if chosen:
                matched.append([chosen['Cmin']-q['Cmin'],chosen['Cmax']-q['Cmax'],chosen['T']-q['T'],(chosen['Cmax']-chosen['Cmin'])-(q['Cmax']-q['Cmin'])])
                I=q['I'];y0=[chosen['section'],chosen['s']]
                def rhs(t,y):return m['co']@np.r_[1,y[0],y[1],I,np.maximum(y[0]-m['padded'],0)]
                initial=solve_ivp(rhs,[0,1e-6],y0,method='DOP853',rtol=2e-11,atol=1e-13).y[:,-1]
                def event(t,y):return y[0]-chosen['section']
                event.direction=1;event.terminal=True
                sol=solve_ivp(rhs,[1e-6,chosen['T']*1.2],initial,events=event,method='DOP853',rtol=2e-10,atol=1e-12,max_step=.05)
                ok=len(sol.t_events[0])>0
                result['independent_checks'].append(dict(I=I,returned=ok,period_error=float(sol.t_events[0][0]-chosen['T']) if ok else None,closure_error=float(sol.y_events[0][0,1]-chosen['s']) if ok else None))
        err=np.array(matched);result['matched_targets']=len(matched);result['missing_targets']=len(f.TARGETS)-len(matched)
        result['RMSE_extrema']=float(np.sqrt(np.mean(err[:,:2]**2)));result['RMSE_period']=float(np.sqrt(np.mean(err[:,2]**2)));result['RMSE_amplitude']=float(np.sqrt(np.mean(err[:,3]**2)))
        result['count_transitions']=[]
        for a,b in zip(rows[:-1],rows[1:]):
            aa=[sum(q['stable'] for q in a['cycles']),sum(not q['stable'] for q in a['cycles'])];bb=[sum(q['stable'] for q in b['cycles']),sum(not q['stable'] for q in b['cycles'])]
            if aa!=bb:result['count_transitions'].append(dict(bracket=[a['I'],b['I']],before=aa,after=bb))
        stable=[(r['I'],q) for r in rows for q in r['cycles'] if q['stable']]
        if stable:
            peak=max(stable,key=lambda v:v[1]['Cmax']-v[1]['Cmin']);result['amplitude_peak']=dict(I=peak[0],amplitude=peak[1]['Cmax']-peak[1]['Cmin'])
        metrics[kind]=result
    # Compare the same common set and disclose unmatched source cycles separately.
    common=[]
    for k in range(len(f.TARGETS)):
        if all(metrics[v]['targets'][k]['cycle'] is not None for v in metrics):common.append(k)
    for kind in metrics:
        ae=[];ext=[]
        for k in common:
            q=metrics[kind]['targets'][k];r=q['reference'];c=q['cycle'];ae.append((c['Cmax']-c['Cmin'])-(r['Cmax']-r['Cmin']));ext.extend([c['Cmin']-r['Cmin'],c['Cmax']-r['Cmax']])
        metrics[kind]['common_matched_targets']=len(common);metrics[kind]['RMSE_amplitude_common']=float(np.sqrt(np.mean(np.array(ae)**2)));metrics[kind]['RMSE_extrema_common']=float(np.sqrt(np.mean(np.array(ext)**2)))
    lr=np.genfromtxt(ROOT/'LR_stable_cycles.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.17})
    fig,axs=plt.subplots(2,2,figsize=(12.5,8.4));colors={'affine':'#e99020',selected:'#111111'};labels={'affine':'Affine C fit',selected:'Mixed-coordinate fit'}
    for ax in [axs[0,0],axs[0,1],axs[1,0]]:
        ax.set_xlim(.34,.66);ax.set_xlabel(r'$I$ ($\mu$M)')
    axs[0,0].scatter(lr['I'],lr['Cmin'],s=8,color='#a855c7',alpha=.65);axs[0,0].scatter(lr['I'],lr['Cmax'],s=8,color='#a855c7',alpha=.65)
    axs[0,1].scatter(lr['I'],lr['Cmax']-lr['Cmin'],s=8,color='#a855c7',alpha=.65,label='Li–Rinzel')
    axs[1,0].scatter(lr['I'],lr['T'],s=8,color='#a855c7',alpha=.65,label='Li–Rinzel')
    for kind,(m,rows) in models.items():
        color=colors[kind]
        for st in [True,False]:
            qq=[(r['I'],q) for r in rows for q in r['cycles'] if q['stable']==st]
            for key in ['Cmin','Cmax']:axs[0,0].scatter([I for I,q in qq],[q[key] for I,q in qq],s=12,facecolors=color if st else 'none',edgecolors=color,lw=.6,alpha=.8)
            axs[0,1].scatter([I for I,q in qq],[q['Cmax']-q['Cmin'] for I,q in qq],s=12,facecolors=color if st else 'none',edgecolors=color,lw=.6,alpha=.8,label=labels[kind] if st else None)
            axs[1,0].scatter([I for I,q in qq],[q['T'] for I,q in qq],s=12,facecolors=color if st else 'none',edgecolors=color,lw=.6,alpha=.8)
        ee=[(r['I'],e) for r in rows for e in r['equilibria']]
        axs[0,0].plot([i for i,e in ee],[e['C'] for i,e in ee],color=color,lw=.8,alpha=.45)
        c=np.linspace(.07,.65,500);axs[1,1].plot(c,f.phi(c,m['shape']),color=color,lw=2,label=labels[kind])
        if kind==selected:
            for ct in m['tc']:axs[0,0].axhline(ct,color='#0ea5e9',ls='--',lw=.8,alpha=.6);axs[1,1].axvline(ct,color='#0ea5e9',ls='--',lw=.8,alpha=.6)
    axs[0,0].set(title='AM: extrema of periodic orbits',ylabel=r'$C$ ($\mu$M)',ylim=(.06,.56))
    axs[0,1].set(title='Physical amplitude: growth and decline',ylabel=r'$C_{max}-C_{min}$ ($\mu$M)',ylim=(0,.43));axs[0,1].legend(fontsize=9)
    axs[1,0].set(title='Periods',ylabel='Period (s)',ylim=(9,16))
    axs[1,1].set(title=r'Coordinate map $z=\Phi(C)$',xlabel=r'$C$ ($\mu$M)',ylabel=r'$z$',xlim=(.07,.65));axs[1,1].legend(fontsize=9)
    fig.suptitle('AM with two movable calcium thresholds: mixed vs affine coordinates',fontsize=14);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(ROOT/'AM_mixed_comparison.png',dpi=180);fig.savefig(ROOT/'AM_mixed_comparison.pdf');plt.close(fig)
    (ROOT/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps({k:{q:v for q,v in r.items() if q not in ['targets','independent_checks']} for k,r in metrics.items()},indent=2))

if __name__=='__main__':main()
