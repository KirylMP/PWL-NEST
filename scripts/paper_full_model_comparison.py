"""Period and bifurcation portraits for LR, fixed-fold, and no-fold models.

PWA stable cycles are finite forward scans; supplied validations provide a few
additional points near constrained folds. No cycle count completeness is claimed.
"""
import argparse,concurrent.futures,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_figure1 as lr
from paper_lower_onset import simulate,unpack,pwa_eq,inverse

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'paper/figures';DATA=OUT/'figure6_data'
CASES=('AM','FM','AFM');MODES=('fixed','no_fold')
XLIM={'AM':(0,.8),'AFM':(0,1.05),'FM':(.35,1.25)}
CLIM={'AM':(0,.72),'AFM':(0,1.05),'FM':(0,1.25)}
TLIM={'AM':(0,25),'AFM':(0,55),'FM':(0,75)}

def model(mode,case):return json.loads((DATA/mode/case/'parameters.json').read_text())

def compute(workers,duration,points):
    meta=json.loads((lr.DATA/'metadata.json').read_text());tasks=[];plan=[]
    for mode in MODES:
        for case in CASES:
            d=model(mode,case);hopf=d['fixed_equilibrium_threshold_I'][0]
            fold=next(e['I'] for e in meta['events'] if e['case']==case and e['kind']=='cycle_fold')
            inputs=np.linspace(max(XLIM[case][0],hopf-(.02 if case=='FM' else .015)),min(XLIM[case][1],fold+.03),points)
            for I in inputs:tasks.append((case,d,float(I),duration));plan.append((mode,case))
    print('forward integrations:',len(tasks),'workers:',workers,flush=True)
    output={mode:{case:[] for case in CASES} for mode in MODES}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for (mode,case),result in zip(plan,pool.map(simulate,tasks)):
            output[mode][case].append(result)
            if len(output[mode][case])%20==0:print(mode,case,'done',len(output[mode][case]),flush=True)
    data={'duration':duration,'points_per_model':points,'method':'DOP853 finite forward trajectories, peak-spacing convergence; no global branch completeness', 'models':output}
    (DATA/'forward_scan.json').write_text(json.dumps(data,indent=2)+'\n')
    for mode in MODES:
        for case in CASES:print(mode,case,'cycles found',sum(r['valid'] for r in output[mode][case]),'/',points,flush=True)

def fold_metrics(d):
    """Reconstruct stored M=1 orbit from four affine segments."""
    shape,cuts,N,H,a,b=unpack(d);I=d['fold_constraint']['I'];times=d['fold_constraint']['crossing_times'];boundaries=[0,1,1,0,0];regions=[1,2,1,0];states=[];mismatches=[];starts=[];ends=[]
    for j,r in enumerate(regions):
        ns=N[1]+sum(N[3:3+r]);hs=H[1]+sum(H[3:3+r]);ni=N[0]+N[2]*I-np.dot(N[3:3+r],cuts[:r]);hi=H[0]+H[2]*I-np.dot(H[3:3+r],cuts[:r])
        A=np.array([[-a*ns,a,-a*ni],[b*hs,-b,b*hi],[0.,0.,0.]])
        E=expm(A*times[j]);z0=cuts[boundaries[j]];z1=cuts[boundaries[j+1]];q0=(z1-E[0,0]*z0-E[0,2])/E[0,1]
        state=np.array([z0,q0,1.]);starts.append(q0);ends.append((E@state)[1]);states.extend((expm(A*t)@state)[0] for t in np.linspace(0,times[j],250))
    closure=max(abs(np.array(ends)-np.roll(starts,-1)));C=inverse(np.array(states),shape)
    return dict(I=float(I),period=float(sum(times)),Cmin=float(min(C)),Cmax=float(max(C)),matching_error=float(closure),status='stored local neutral orbit')

def build_data():
    data=json.loads((DATA/'forward_scan.json').read_text());models={mode:{case:model(mode,case) for case in CASES} for mode in MODES};rows=lr.load_rows();metadata=json.loads((lr.DATA/'metadata.json').read_text())
    markers={case:fold_metrics(models['fixed'][case]) for case in CASES};(DATA/'fixed_fold_orbits.json').write_text(json.dumps(markers,indent=2)+'\n')
    provenance={mode:{case:dict(sha256=hashlib.sha256((DATA/mode/case/'parameters.json').read_bytes()).hexdigest(),source='mixed_models_for_review.zip' if mode=='fixed' else 'free_hidden_for_review.zip',note='earlier FM candidate; final frozen FM remains server-only' if (mode,case)==('no_fold','FM') else '') for case in CASES} for mode in MODES}
    (DATA/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return data,models,rows,metadata,markers

def save(fig,name):
    for ext in ('pdf','svg','png'):fig.savefig(OUT/(name+'.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)

def make(kind,data,models,rows,metadata,markers):
    path=DATA/'fold_branch_search.json'
    fold_branches=json.loads(path.read_text())['found'] if path.exists() else {}
    continued_path=DATA/'continued_pwa_cycles.json'
    continued=json.loads(continued_path.read_text())['results'] if continued_path.exists() else []
    upper_path=DATA/'upper_pwa_cycles.json'
    upper=json.loads(upper_path.read_text())['results'] if upper_path.exists() else []
    fm_fold_path=DATA/'no_fold_FM_fold_search.json'
    fm_folds=json.loads(fm_fold_path.read_text())['results'] if fm_fold_path.exists() else []
    fold_source=json.loads((ROOT/'paper/data/two_threshold_models/sources/mixed_fold_audit.json').read_text())
    fig,axes=plt.subplots(3,3,figsize=(10,7.5),sharex='col',sharey='col');fig.subplots_adjust(left=.075,right=.99,top=.91,bottom=.12,wspace=.27,hspace=.25)
    for j,case in enumerate(CASES):
        ref_eq=np.load(lr.DATA/f'{case}_equilibria.npz');ref_rows=[r for r in rows if r['case']==case and r['displayed']]
        for i,mode in enumerate(('reference','fixed','no_fold')):
            ax=axes[i,j];ax.grid(alpha=.12);ax.set_xlim(*XLIM[case]);ax.set_ylim(*(TLIM if kind=='period' else CLIM)[case])
            if i==0:
                if kind=='bifurcation':
                    for stable,col,ls in [(True,'#245f9b','-'),(False,'#bf434a','--')]:ax.plot(ref_eq['I'],np.where(ref_eq['stable']==stable,ref_eq['C'],np.nan),color=col,ls=ls,lw=1.15)
                for stable in (True,False):
                    subset=[r for r in ref_rows if r['stable']==stable]
                    keys=('period_s',) if kind=='period' else ('C_min','C_max')
                    for key in keys:ax.scatter([r['I'] for r in subset],[r[key] for r in subset],s=5.5 if stable else 10,facecolors='#202020' if stable else 'white',edgecolors='#202020' if stable else '#a32839',linewidths=.7,zorder=3)
                event=next(e for e in metadata['events'] if e['case']==case and e['kind']=='cycle_fold')
                for val in ([event['period_s']] if kind=='period' else [event['C_min'],event['C_max']]):ax.scatter(event['I'],val,marker='D',s=16,color='#7354a4',zorder=4)
            else:
                d=models[mode][case];m=unpack(d)
                if kind=='bifurcation':
                    inputs=np.linspace(*XLIM[case],700);eqs=[pwa_eq(I,m) for I in inputs]
                    for stable,col,ls in [(True,'#245f9b','-'),(False,'#bf434a','--')]:ax.plot(inputs,[next((r[0] for r in q if r[1]==stable),np.nan) for q in eqs],color=col,ls=ls,lw=1.15)
                forward=[r for r in data['models'][mode][case] if r['valid']]
                keys=('period',) if kind=='period' else ('Cmin','Cmax')
                for key in keys:ax.scatter([r['I'] for r in forward],[r[key] for r in forward],s=9,color='#202020',zorder=3)
                validation=json.loads((DATA/mode/case/'validation.json').read_text())
                checked=[r for r in validation['cycles'] if r.get('PWA')]
                for key in keys:ax.scatter([r['I'] for r in checked],[r['PWA'][key] for r in checked],s=8,color='#202020',zorder=3)
                # Additional finite independently calculated results, including any found unstable roots.
                for branch in validation.get('local_fold_branches',[]):
                    for cycle in branch['cycles']:
                        vals=[cycle['period']] if kind=='period' else [cycle['Cmin'],cycle['Cmax']]
                        ax.scatter([branch['I']]*len(vals),vals,s=12 if cycle['stable'] else 16,facecolors='#202020' if cycle['stable'] else 'white',edgecolors='#202020' if cycle['stable'] else '#a32839',linewidths=.8,zorder=4)
                for cycle in fold_branches.get(f'{mode}_{case}',[]):
                    vals=[cycle['period']] if kind=='period' else [cycle['Cmin'],cycle['Cmax']]
                    ax.scatter([cycle['I']]*len(vals),vals,s=12 if cycle['stable'] else 16,facecolors='#202020' if cycle['stable'] else 'white',edgecolors='#202020' if cycle['stable'] else '#a32839',linewidths=.8,zorder=4)
                for branch in continued+upper:
                    if branch['mode']!=mode or branch['case']!=case:continue
                    for stable in (True,False):
                        subset=[r for r in branch['points'] if r['stable']==stable]
                        if not stable and len(subset)>40:subset=subset[::2]
                        for key in keys:
                            ax.scatter([r['I'] for r in subset],[r[key] for r in subset],s=10,facecolors='#202020' if stable else 'white',edgecolors='#202020' if stable else '#a32839',linewidths=.7,zorder=5)
                if mode=='fixed':
                    event=markers[case]
                    for val in ([event['period']] if kind=='period' else [event['Cmin'],event['Cmax']]):ax.scatter(event['I'],val,marker='D',s=16,color='#7354a4',zorder=7)
                elif case in ('AM','AFM'):
                    event=next(v for v in fold_source['results'] if v.get('model')==case.lower() and v.get('status')=='candidate_fold' and v.get('itinerary')=='both_thresholds')
                    for val in ([event['period']] if kind=='period' else [event['Cmin'],event['Cmax']]):ax.scatter(event['I'],val,marker='D',s=16,color='#7354a4',zorder=7)
                elif case=='FM':
                    candidates=[v for v in fm_folds if v.get('status')=='candidate_fold' and v.get('itinerary')=='both_thresholds']
                    if candidates:
                        event=candidates[0]
                        for val in ([event['period']] if kind=='period' else [event['Cmin'],event['Cmax']]):ax.scatter(event['I'],val,marker='D',s=16,color='#7354a4',zorder=7)
            if i==0:ax.set_title(case+rf' ($K_{{ER}}={lr.PARAMETERS["KER_"+case]:g}\,\mu$M)',fontsize=9)
            if j==0:ax.set_ylabel(('Li–Rinzel','PWA: fold fixed','PWA: no fold target')[i]+'\n'+('Period (s)' if kind=='period' else r'$C$ ($\mu$M)'),fontsize=8)
            if i==2:ax.set_xlabel(r'$I$ ($\mu$M)')
            if (mode,case)==('no_fold','FM'):ax.text(.98,.96,'earlier candidate',transform=ax.transAxes,ha='right',va='top',fontsize=6.5,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    legend=[Line2D([],[],marker='o',ls='',color='#202020',ms=4,label='stable cycle'),Line2D([],[],marker='o',ls='',color='#a32839',mfc='white',ms=4,label='unstable cycle'),Line2D([],[],marker='D',ls='',color='#7354a4',ms=4,label='cycle fold')]
    if kind=='bifurcation':legend=[Line2D([],[],color='#245f9b',label='stable equilibrium'),Line2D([],[],color='#bf434a',ls='--',label='unstable equilibrium')]+legend
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.025),ncol=len(legend),frameon=False,fontsize=7)
    fig.suptitle('Oscillation periods' if kind=='period' else 'Equilibria and cycle extrema',fontsize=11)
    save(fig,'figure5_period_comparison' if kind=='period' else 'figure6_bifurcation_comparison')

def main():
    p=argparse.ArgumentParser();p.add_argument('--plot-only',action='store_true');p.add_argument('--workers',type=int,default=6);p.add_argument('--duration',type=float,default=600);p.add_argument('--points',type=int,default=61);a=p.parse_args()
    if not a.plot_only:compute(a.workers,a.duration,a.points)
    v=build_data();make('period',*v);make('bifurcation',*v)
if __name__=='__main__':main()
