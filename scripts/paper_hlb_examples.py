"""Compute two PWA phase portraits with stable HLB1 and HLB2 cycles."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/paper_mpl')
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from paper_mechanism_figures import OUT, save

DELTA=.12
CASES={'HLB1':(1.,-.45),'HLB2':(3.,-.45)}

def F(z,slopes):
    left,right=slopes
    return np.where(np.asarray(z)<0,left*np.asarray(z),right*np.asarray(z))

def flow(z,y,slopes,delta):
    return y-F(z,slopes),delta-z

def run_case(name,slopes):
    def rhs(t,u):
        z,y=u[:2];dz,dy=flow(z,y,slopes,DELTA)
        return np.array([dz,dy,-(slopes[0] if z<0 else slopes[1])])
    def peak(t,u):return flow(u[0],u[1],slopes,DELTA)[0]
    peak.direction=-1
    sol=solve_ivp(rhs,(0,500),[DELTA+.2,0.,0.],method='DOP853',rtol=1e-9,atol=1e-11,max_step=.04,events=peak,dense_output=True)
    if not sol.success or len(sol.t_events[0])<4:raise RuntimeError(f'{name}: no resolved cycle')
    times=sol.t_events[0];states=sol.y_events[0];t0,t1=times[-2:];period=t1-t0
    closure=float(np.max(np.abs(states[-1,:2]-states[-2,:2])))
    period_change=float(abs(period-(times[-2]-times[-3])))
    if closure>1e-5 or period_change>1e-5:raise RuntimeError(f'{name}: unresolved cycle: closure={closure}, period change={period_change}')
    u=sol.sol(np.linspace(t0,t1,2500));z,y=u[:2];log_m=float(u[2,-1]-u[2,0])
    if not min(z)<0<max(z) or log_m>=0:raise RuntimeError(f'{name}: expected stable cycle crossing threshold')
    return dict(name=name,delta=DELTA,threshold=0.,slopes=list(slopes),traces=[-s for s in slopes],determinants=[1.,1.],discriminants=[s*s-4 for s in slopes],equilibrium=[DELTA,float(F(DELTA,slopes))],period=float(period),closure_error=closure,period_change=period_change,log_floquet_multiplier=log_m,z_min=float(min(z)),z_max=float(max(z)),y_min=float(min(y)),y_max=float(max(y)),coordinate_note='dimensionless constructed PWA example, not fitted LR or physical calcium'),z,y

def main():
    cases=[run_case(name,slopes) for name,slopes in CASES.items()]
    zlo=min(v[0]['z_min'] for v in cases);zhi=max(v[0]['z_max'] for v in cases)
    ymin=min(v[0]['y_min'] for v in cases);ymax=max(v[0]['y_max'] for v in cases)
    zl,zh=zlo-.25*(zhi-zlo),zhi+.25*(zhi-zlo);yl,yh=ymin-.38*(ymax-ymin),ymax+.38*(ymax-ymin)
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.5));fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.84,wspace=.20)
    for ax,(info,z,y) in zip(axs,cases):
        slopes=info['slopes'];ax.axvspan(zl,0,color='#eaf2f7',zorder=-5);ax.axvspan(0,zh,color='#fff3e9',zorder=-5)
        ax.axvline(0,color='#159db5',lw=1,ls='--',zorder=1)
        Z,Y=np.meshgrid(np.linspace(zl,zh,23),np.linspace(yl,yh,19));U,V=flow(Z,Y,slopes,DELTA)
        norm=np.maximum(np.hypot(U/(zh-zl),V/(yh-yl)),1e-14)
        ax.quiver(Z,Y,U/norm,V/norm,color='.65',alpha=.75,angles='xy',scale_units='xy',scale=60,width=.0028,zorder=0)
        grid=np.linspace(zl,zh,1000);ax.plot(grid,F(grid,slopes),color='#da8523',lw=1.);ax.axvline(DELTA,color='#359367',lw=1.)
        ax.plot(z,y,color='#171717',lw=1.7,zorder=3)
        for fraction in (.12,.38,.68,.88):
            k=int(fraction*(len(z)-12));ax.annotate('',xy=(z[k+10],y[k+10]),xytext=(z[k],y[k]),arrowprops=dict(arrowstyle='-|>',color='#171717',lw=1,mutation_scale=10))
        ax.scatter(*info['equilibrium'],color='#b83c50',s=20,zorder=4)
        ax.set(xlim=(zl,zh),ylim=(yl,yh),xlabel='$z$ (dimensionless)',ylabel='$y$ (dimensionless)')
        ax.set_title(info['name']+(': focus / focus' if info['name']=='HLB1' else ': node / focus'),fontsize=10)
        ax.text(.03,.97,'stable focus' if info['name']=='HLB1' else 'stable node',transform=ax.transAxes,va='top',fontsize=8)
        ax.text(.97,.97,'unstable focus',transform=ax.transAxes,ha='right',va='top',fontsize=8)
        ax.text(.03,.04,rf"$T={info['period']:.2f}$, $M={np.exp(info['log_floquet_multiplier']):.3f}$",transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
    fig.suptitle('Continuous two-zone PWA examples: local stable cycles',fontsize=11)
    fig.legend(handles=[Line2D([],[],color='#171717',lw=1.7,label='stable cycle'),Line2D([],[],color='#da8523',label=r'$\dot z=0$'),Line2D([],[],color='#359367',label=r'$\dot y=0$'),Line2D([],[],color='#159db5',ls='--',label='threshold'),Line2D([],[],marker='o',ls='',color='#b83c50',label='equilibrium')],loc='lower center',bbox_to_anchor=(.5,.04),ncol=5,frameon=False,fontsize=8)
    save(fig,'figure3_hlb')
    folder=OUT/'figure3_data';folder.mkdir(exist_ok=True)
    for info,z,y in cases:np.savez_compressed(folder/(info['name']+'_cycle.npz'),z=z,y=y)
    (folder/'parameters.json').write_text(json.dumps(dict(system='zdot = y-F(z); ydot=delta-z; F(z)=s_left*z for z<0, s_right*z otherwise',method='DOP853, attracting forward trajectory, last two peak events, divergence integral',cases=[v[0] for v in cases]),indent=2)+'\n')
    print(json.dumps([v[0] for v in cases],indent=2))

if __name__=='__main__':main()
