"""Plot and independently verify a selected four-region experiment."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fit_four as f
P=f.P

def main():
    name=sys.argv[1] if len(sys.argv)>1 else 'candidate'
    model=json.loads((P/(name+'.json')).read_text());x=np.array(model['parameters']);m=f.unpack(x)
    ds=json.loads((P/(name+'_branches.json')).read_text());rows=ds['rows'];lr=json.loads((P/'LR_cycles_reference.json').read_text())['rows']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16})
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    for I in np.linspace(.4,1.25,1200):
        for e in f.equilibrium(I,m):axs[0].plot(I,e['C'],'.',ms=1.2,color='#2563eb' if e['stable'] else '#dc2626')
    for rrs,col,sz in [(lr,'#a855c7',7),(rows,'#111111',15)]:
        for st in [True,False]:
            qq=[(r['I'],q) for r in rrs for q in r['cycles'] if q['stable']==st]
            for key in ['Cmin','Cmax']:
                axs[0].scatter([I for I,q in qq],[q[key] for I,q in qq],s=sz,facecolors=col if st else 'none',edgecolors=col,lw=.6,alpha=.7)
            axs[1].scatter([I for I,q in qq],[q['T'] for I,q in qq],s=sz,facecolors=col if st else 'none',edgecolors=col,lw=.6,alpha=.7)
    for j,c in enumerate(f.TC):axs[0].axhline(c,color='#0ea5e9',ls='--',lw=.8,alpha=.7)
    for st,col,label in [(True,'#2563eb','Stable'),(False,'#dc2626','Unstable')]:
        axs[2].step([r['I'] for r in rows],[sum(q['stable']==st for q in r['cycles']) for r in rows],where='mid',color=col,label=label)
    axs[0].set(xlim=(.4,1.25),ylim=(0,1.4),xlabel=r'$I$ ($\mu$M)',ylabel=r'$C$ ($\mu$M)',title='Four regions: equilibria and cycle extrema')
    axs[1].set(xlim=(.4,1.25),ylim=(0,90),xlabel=r'$I$ ($\mu$M)',ylabel='Period (s)',title='Cycle periods')
    axs[2].set(xlim=(.4,1.25),ylim=(-.1,2.1),yticks=[0,1,2],xlabel=r'$I$ ($\mu$M)',ylabel='Detected cycles',title='Independent numerical cycle scan');axs[2].legend()
    handles=[Line2D([],[],marker='o',color='#111111',ls='none',label='Four-region model'),Line2D([],[],marker='o',color='#a855c7',ls='none',label='Li–Rinzel'),Line2D([],[],marker='o',color='#111111',ls='none',markerfacecolor='none',label='Unstable cycle'),Line2D([],[],color='#0ea5e9',ls='--',label='Fixed C thresholds')]
    fig.legend(handles=handles,loc='lower center',ncol=4);fig.tight_layout(rect=(0,.09,1,1));fig.savefig(P/(name+'_comparison.png'),dpi=180);fig.savefig(P/(name+'_comparison.pdf'));plt.close(fig)
    metrics=dict(max_detected_cycles=max(len(r['cycles']) for r in rows),I_samples=len(rows),events=ds['events'],samples=[],independent_checks=[])
    for I in [.55,.65,.8,.9,1.,1.1]:
        r=min(rows,key=lambda r:abs(r['I']-I));src=min(lr,key=lambda r:abs(r['I']-I))
        metrics['samples'].append(dict(I=r['I'],model=r['cycles'],LR=src['cycles']))
        for q in r['cycles']:
            direct=-1 if q['backward'] else 1
            def rhs(t,y):return direct*(m['co']@np.r_[1,y[0],y[1],I,np.maximum(y[0]-m['padded'],0)])
            y0=[q['section'],q['s']];tiny=1e-6
            initial=solve_ivp(rhs,[0,tiny],y0,method='DOP853',rtol=1e-11,atol=1e-13).y[:,-1]
            def event(t,y):return y[0]-q['section']
            event.direction=direct;event.terminal=True
            sol=solve_ivp(rhs,[tiny,q['T']*1.2],initial,events=event,method='DOP853',rtol=2e-10,atol=1e-12,max_step=.08)
            ok=len(sol.t_events[0])>0
            metrics['independent_checks'].append(dict(I=I,stable=q['stable'],returned=ok,period_error=float(sol.t_events[0][0]-q['T']) if ok else None,closure_error=float(sol.y_events[0][0,1]-q['s']) if ok else None))
    # Count transitions bracketed by consecutive scan nodes.
    metrics['count_transitions']=[]
    for a,b in zip(rows[:-1],rows[1:]):
        sa=[sum(q['stable'] for q in a['cycles']),sum(not q['stable'] for q in a['cycles'])];sb=[sum(q['stable'] for q in b['cycles']),sum(not q['stable'] for q in b['cycles'])]
        if sa!=sb:metrics['count_transitions'].append(dict(bracket=[a['I'],b['I']],before=sa,after=sb))
    C,I,w,y,*_=f.DATA;bb=f.basis(C,I,m);geo=np.stack([bb@m['N'],bb@m['H']],axis=-1)-y
    metrics['nullcline_RMSE']=np.sqrt(np.sum((geo*w[:,:,None])**2,axis=(0,1))).tolist()
    (P/(name+'_metrics.json')).write_text(json.dumps(metrics,indent=2));print(json.dumps({k:v for k,v in metrics.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
