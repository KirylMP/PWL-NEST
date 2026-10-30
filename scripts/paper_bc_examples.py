"""Generate phase portraits and stable large cycles for two concrete PWA systems."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/paper_mpl')
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from paper_mechanism_figures import OUT, save

THRESHOLDS=(-1.,1.)
CASES={'BC1':(1.,-.45,1.),'BC2':(3.,-.45,1.)}

def F(x,slopes):
    left,middle,right=slopes
    return middle*x+(left-middle)*np.minimum(x+1.,0.)+(right-middle)*np.maximum(x-1.,0.)

def flow(x,y,slopes):
    return y-F(x,slopes),-x

def trace(x,slopes):
    return -np.where(x < -1,slopes[0],np.where(x > 1,slopes[2],slopes[1]))

def run_case(name,slopes):
    def rhs(t,u):
        x,y=u[:2];dx,dy=flow(x,y,slopes)
        return np.array([dx,dy,trace(x,slopes)])
    def peak(t,u):return flow(u[0],u[1],slopes)[0]
    peak.direction=-1
    sol=solve_ivp(rhs,(0,500),[.25,0.,0.],method='DOP853',rtol=1e-9,atol=1e-11,
                  max_step=.06,events=peak,dense_output=True)
    if not sol.success or len(sol.t_events[0])<4:raise RuntimeError(f'{name}: no resolved oscillation')
    events=sol.t_events[0]; states=sol.y_events[0]; t0,t1=events[-2:];period=t1-t0
    closure=np.max(np.abs(states[-1,:2]-states[-2,:2]))
    period_change=abs(period-(events[-2]-events[-3]))
    if closure>1e-5 or period_change>1e-5:raise RuntimeError(f'{name}: orbit has not converged: closure={closure}, period change={period_change}')
    phase=np.linspace(t0,t1,2500);u=sol.sol(phase);x,y=u[:2]
    if not (min(x)<-1 and max(x)>1):raise RuntimeError(f'{name}: orbit does not cross both thresholds')
    log_m=float(u[2,-1]-u[2,0])
    if log_m>=0:raise RuntimeError(f'{name}: converged cycle not classified stable')
    return dict(name=name,slopes=list(slopes),thresholds=list(THRESHOLDS),
                traces=[-s for s in slopes],determinants=[1.,1.,1.],
                discriminants=[s*s-4 for s in slopes],period=float(period),
                closure_error=float(closure),period_change=float(period_change),
                log_floquet_multiplier=log_m,x_min=float(min(x)),x_max=float(max(x)),
                y_min=float(min(y)),y_max=float(max(y)),coordinate_note='dimensionless constructed PWA example, not fitted LR or physical calcium'),x,y

def main():
    cases=[run_case(name,slopes) for name,slopes in CASES.items()]
    maxx=max(max(abs(info['x_min']),abs(info['x_max'])) for info,_,_ in cases)*1.20
    maxy=max(max(abs(info['y_min']),abs(info['y_max'])) for info,_,_ in cases)*1.25
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.5));fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.84,wspace=.20)
    for ax,(info,x,y) in zip(axs,cases):
        slopes=info['slopes'];ax.axvspan(-maxx,-1,color='#eaf2f7',zorder=-5);ax.axvspan(-1,1,color='#fff3e9',zorder=-5);ax.axvspan(1,maxx,color='#eaf2f7',zorder=-5)
        for threshold in THRESHOLDS:ax.axvline(threshold,color='#159db5',lw=1,ls='--',zorder=1)
        X,Y=np.meshgrid(np.linspace(-maxx*.94,maxx*.94,23),np.linspace(-maxy*.94,maxy*.94,19));U,V=flow(X,Y,slopes)
        norm=np.hypot(U/maxx,V/maxy);norm=np.maximum(norm,1e-14)
        ax.quiver(X,Y,U/norm,V/norm,color='.65',alpha=.75,angles='xy',scale_units='xy',scale=30,width=.0028,zorder=0)
        grid=np.linspace(-maxx,maxx,1400);ax.plot(grid,F(grid,slopes),color='#da8523',lw=1.,label=r'$\dot z=0$');ax.axvline(0,color='#359367',lw=1.,label=r'$\dot y=0$')
        ax.plot(x,y,color='#171717',lw=1.7,zorder=3)
        for fraction in (.12,.38,.68,.88):
            k=int(fraction*(len(x)-12));ax.annotate('',xy=(x[k+10],y[k+10]),xytext=(x[k],y[k]),arrowprops=dict(arrowstyle='-|>',color='#171717',lw=1,mutation_scale=10))
        ax.scatter([0],[0],color='#b83c50',s=20,zorder=4)
        ax.set(xlim=(-maxx,maxx),ylim=(-maxy,maxy),xlabel='$z$ (dimensionless)',ylabel='$y$ (dimensionless)')
        ax.set_title(info['name']+(': lower focus' if info['name']=='BC1' else ': lower node'),fontsize=10)
        ax.text(.03,.97,'stable focus' if info['name']=='BC1' else 'stable node',transform=ax.transAxes,va='top',fontsize=8)
        ax.text(.5,.97,'unstable focus',transform=ax.transAxes,ha='center',va='top',fontsize=8)
        ax.text(.97,.97,'stable focus',transform=ax.transAxes,ha='right',va='top',fontsize=8)
        ax.text(.03,.04,rf"$T={info['period']:.2f}$, $M={np.exp(info['log_floquet_multiplier']):.3f}$",transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
    fig.suptitle('Continuous three-zone PWA examples: large stable cycles',fontsize=11)
    fig.legend(handles=[Line2D([],[],color='#171717',lw=1.7,label='stable cycle'),Line2D([],[],color='#da8523',label=r'$\dot z=0$'),Line2D([],[],color='#359367',label=r'$\dot y=0$'),Line2D([],[],color='#159db5',ls='--',label='thresholds'),Line2D([],[],marker='o',ls='',color='#b83c50',label='equilibrium')],loc='lower center',bbox_to_anchor=(.5,.04),ncol=5,frameon=False,fontsize=8)
    save(fig,'figure4_large_cycles')
    folder=OUT/'figure4_data';folder.mkdir(exist_ok=True)
    for info,x,y in cases:np.savez_compressed(folder/(info['name']+'_cycle.npz'),z=x,y=y)
    (folder/'parameters.json').write_text(json.dumps(dict(system='zdot = y-F(z); ydot=-z; F continuous piecewise linear with F(0)=0',method='DOP853, attracting forward trajectory, last two peak events, divergence integral',cases=[v[0] for v in cases]),indent=2)+'\n')
    print(json.dumps([v[0] for v in cases],indent=2))

if __name__=='__main__':main()
