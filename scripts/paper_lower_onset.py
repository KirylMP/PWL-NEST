"""Finite lower-onset comparison of LR continuation and frozen mixed PWA models."""
import argparse,csv,hashlib,json,os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/paper_mpl')
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_figure1 as lr

OUT=lr.OUT;DATA=OUT/'figure5_data';CASES=('AM','AFM','FM')
WINDOW={'AM':.008,'AFM':.008,'FM':.004}
YLIM={'AM':(.10,.23),'AFM':(.02,.56),'FM':(.0,1.10)}

def phi(c,shape):
    p,e,n,k=[shape[v] for v in ('p','e','n','k')];t=np.asarray(c)/.5
    w=np.exp(np.array([p,e,0.])-max(p,e,0.));w/=w.sum()
    return w[0]*np.expm1(n*np.log(t))/n+w[1]*(np.expm1(k*(t-1))/k if abs(k)>1e-7 else t-1)+w[2]*np.log(t)

def inverse(z,shape):
    z=np.asarray(z);low=np.full(z.shape,np.log(1e-12));high=np.full(z.shape,np.log(2/1.185))
    if np.any(z<=phi(1e-12,shape)) or np.any(z>=phi(2/1.185,shape)):raise ValueError('outside physical C range')
    for _ in range(55):
        mid=(low+high)/2;mask=phi(np.exp(mid),shape)<z;low=np.where(mask,mid,low);high=np.where(mask,high,mid)
    return np.exp((low+high)/2)

def unpack(d):
    shape=d['transform'];cuts=phi(np.array(d['physical_thresholds_C']),shape)
    return shape,cuts,np.asarray(d['alpha']),np.asarray(d['beta']),d['a'],d['b']

def field(z,q,I,m):
    shape,cuts,N,H,a,b=m
    basis=np.array([1.,z,I,max(z-cuts[0],0.),max(z-cuts[1],0.)])
    return np.array([a*(q-basis@N),b*(basis@H-q)])

def pwa_eq(I,m):
    shape,cuts,N,H,a,b=m;ns=N[1]+np.r_[0.,np.cumsum(N[3:])];hs=H[1]+np.r_[0.,np.cumsum(H[3:])]
    ni=N[0]-np.r_[0.,np.cumsum(N[3:]*cuts)];hi=H[0]-np.r_[0.,np.cumsum(H[3:]*cuts)]
    roots=(hi-ni+(H[2]-N[2])*I)/(ns-hs);out=[]
    for j,z in enumerate(roots):
        low=-np.inf if j==0 else cuts[j-1];high=np.inf if j==2 else cuts[j]
        if low<=z<=high:
            try:c=float(inverse(z,shape))
            except ValueError:continue
            trace=-a*ns[j]-b;det=a*b*(ns[j]-hs[j]);out.append((c,trace<0 and det>0,det<0))
    return out

def simulate(task):
    case,d,I,duration=task;m=unpack(d);shape,cuts,N,H,a,b=m;initial={'AM':.35,'AFM':.7,'FM':1.0}[case]
    z0=float(phi(initial,shape));basis=np.array([1.,z0,I,max(z0-cuts[0],0.),max(z0-cuts[1],0.)]);q0=float(basis@N)
    def rhs(t,y):return field(y[0],y[1],I,m)
    def peak(t,y):return rhs(t,y)[0]
    peak.direction=-1
    zlow=float(phi(1e-12,shape));zhigh=float(phi(2/1.185,shape))
    def domain_low(t,y):return y[0]-zlow
    def domain_high(t,y):return zhigh-y[0]
    domain_low.terminal=True;domain_high.terminal=True
    try:
        sol=solve_ivp(rhs,(0,duration),[z0,q0],method='DOP853',rtol=3e-8,atol=3e-10,max_step=.25,events=(peak,domain_low,domain_high),dense_output=True)
        ts=sol.t_events[0];ts=ts[ts>duration*.5]
        if not sol.success or sol.t[-1]<duration or len(ts)<3:return dict(I=I,valid=False,reason='insufficient_peaks_or_domain')
        T=ts[-1]-ts[-2];prev=ts[-2]-ts[-3];coords=sol.sol(np.linspace(ts[-2],ts[-1],700));C=inverse(coords[0],shape)
        amplitude=float(np.ptp(C));change=float(abs(T-prev))
        if amplitude<1e-4 or change>max(.1,.01*T):return dict(I=I,valid=False,reason='nonconverged_or_small')
        return dict(I=I,valid=True,Cmin=float(min(C)),Cmax=float(max(C)),period=float(T),period_change=change,scope='finite forward DOP853, converged peak spacing; only attracting trajectories')
    except (ValueError,OverflowError,FloatingPointError):return dict(I=I,valid=False,reason='numerical_error')

def compute(workers,duration):
    models={c:json.loads((DATA/'models'/f'{c}.json').read_text()) for c in CASES}
    tasks=[(c,models[c],float(I),duration) for c in CASES for I in models[c]['fixed_equilibrium_threshold_I'][0]+np.linspace(-WINDOW[c],WINDOW[c],41)]
    with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(simulate,tasks))
    output={c:results[j*41:(j+1)*41] for j,c in enumerate(CASES)}
    (DATA/'pwa_forward_scan.json').write_text(json.dumps({'duration':duration,'results':output},indent=2)+'\n')
    for c in CASES:print(c,'resolved forward cycles:',sum(v['valid'] for v in output[c]),'/ 41',flush=True)

def compute_lr_local_seeds():
    metadata=json.loads((lr.DATA/'metadata.json').read_text());out={}
    for case in CASES:
        hopf=min((e for e in metadata['events'] if e['case']==case and e['kind']=='trace_zero'),key=lambda e:e['I'])
        rows=[]
        for amplitude in np.geomspace(.0001,.025,22):
            try:
                solution=lr.seed_cycle(case,hopf,float(amplitude));row,_=lr.cycle_metrics(case,solution)
                if row['displayed']:rows.append(row)
            except (ValueError,RuntimeError,OverflowError,FloatingPointError) as exc:
                print(case,'unresolved amplitude',amplitude,str(exc)[:120],flush=True)
        out[case]=rows;print(case,'resolved local LR seeds:',len(rows),flush=True)
    (DATA/'lr_local_seeds.json').write_text(json.dumps(out,indent=2)+'\n')

def plot():
    models={c:json.loads((DATA/'models'/f'{c}.json').read_text()) for c in CASES};scan=json.loads((DATA/'pwa_forward_scan.json').read_text())['results']
    meta=json.loads((lr.DATA/'metadata.json').read_text());rows=lr.load_rows()
    local=json.loads((DATA/'lr_local_seeds.json').read_text())
    search_path=DATA/'pwa_both_threshold_search.json'
    searches=json.loads(search_path.read_text())['found'] if search_path.exists() else {}
    fig,axes=plt.subplots(2,3,figsize=(11,6.8),sharex='col',sharey='col');fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.19,wspace=.28,hspace=.16)
    for col,case in enumerate(CASES):
        d=models[case];cross=d['fixed_equilibrium_threshold_I'][0];ax=axes[0,col];eq=np.load(lr.DATA/f'{case}_equilibria.npz')
        for stable,color,style in [(True,'#2166a1','-'),(False,'#bf3e45','--')]:
            ax.plot(eq['I']-cross,np.where(eq['stable']==stable,eq['C'],np.nan),color=color,ls=style,lw=1.4)
        selected=[r for r in rows if r['case']==case and r['displayed'] and abs(r['I']-cross)<WINDOW[case]]+local[case]
        for stable in (True,False):
            branch=[r for r in selected if r['stable']==stable]
            for key in ('C_min','C_max'):
                ax.scatter([r['I']-cross for r in branch],[r[key] for r in branch],s=9,facecolor='#282828' if stable else 'white',edgecolor='#282828',lw=.7,zorder=4)
        ax.set_title(f'{case}: Li–Rinzel',fontsize=9)
        ax=axes[1,col];m=unpack(d)
        grid=cross+np.linspace(-WINDOW[case],WINDOW[case],500);eqs=[pwa_eq(I,m) for I in grid]
        for stable,color,style in [(True,'#2166a1','-'),(False,'#bf3e45','--')]:
            vals=[next((q[0] for q in values if q[1]==stable),np.nan) for values in eqs]
            ax.plot(grid-cross,vals,color=color,ls=style,lw=1.4)
        branch=[r for r in scan[case] if r['valid']]
        for key in ('Cmin','Cmax'):ax.scatter([r['I']-cross for r in branch],[r[key] for r in branch],s=12,c='#282828',zorder=4)
        for stable in (True,False):
            branch=[r for r in searches.get(case,[]) if r['stable']==stable]
            for key in ('Cmin','Cmax'):
                ax.scatter([r['I']-cross for r in branch],[r[key] for r in branch],s=15,facecolor='#282828' if stable else 'white',edgecolor='#282828',lw=.7,zorder=4)
        if case=='AM':
            audit=json.loads((DATA/'AM_audit/lower_hlb.json').read_text())['models']['candidate']['cycles']
            for r in audit:
                delta=r['delta_I'];zthr=m[1][0];zmin=float(phi(r['Cmin'],m[0]));zmax=float(phi(r['Cmax'],m[0]));ds=np.linspace(0,min(WINDOW[case],.004),80)
                for extremum in (zmin,zmax):
                    cc=inverse(zthr+(extremum-zthr)*ds/delta,m[0]);ax.plot(ds,cc,c='#262626',lw=1.)
        ax.set_xlim(-WINDOW[case],WINDOW[case]);ax.set_xlabel(r'$I-I_{\rm lower}$ ($\mu$M)')
        ax.set_title(case+' PWA'+(' (earlier candidate)' if case=='FM' else ''),fontsize=9)
        for axis in axes[:,col]:axis.axvline(0,color='.5',ls=':',lw=.8);axis.set_ylim(*YLIM[case]);axis.grid(alpha=.16)
    for axis in axes[:,0]:axis.set_ylabel(r'Calcium $C$ ($\mu$M)')
    fig.suptitle('Lower transition: LR and fitted two-threshold PWA',fontsize=12)
    fig.legend(handles=[Line2D([],[],color='#2166a1',label='stable equilibrium'),Line2D([],[],color='#bf3e45',ls='--',label='unstable equilibrium'),Line2D([],[],color='#282828',marker='o',ls='',ms=4,label='stable cycle'),Line2D([],[],marker='o',mfc='white',mec='#282828',ls='',ms=4,label='unstable cycle')],loc='lower center',bbox_to_anchor=(.5,.05),ncol=4,frameon=False,fontsize=8)
    fig.savefig(OUT/'figure5_lower_onset.pdf',bbox_inches='tight');fig.savefig(OUT/'figure5_lower_onset.svg',bbox_inches='tight');fig.savefig(OUT/'figure5_lower_onset.png',dpi=300,bbox_inches='tight');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plot-only',action='store_true');p.add_argument('--workers',type=int,default=6);p.add_argument('--duration',type=float,default=600);a=p.parse_args()
    if not a.plot_only:compute(a.workers,a.duration);compute_lr_local_seeds()
    plot()
if __name__=='__main__':main()
