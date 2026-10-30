"""LR equilibrium spectra and conceptual illustrations for two-threshold PWA."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Patch
import paper_figure1 as lr

OUT=lr.OUT
COL={'stable node':'#205e99','stable focus':'#3293b3','saddle':'#b17d24','unstable node':'#d26c38','unstable focus':'#b83c50'}
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none'})

def save(fig,name):
    for ext in ('pdf','svg','png'):fig.savefig(OUT/(name+'.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)

def spectrum():
    fig,(ax,detail)=plt.subplots(1,2,figsize=(10.5,6.5))
    fig.subplots_adjust(left=.085,right=.98,top=.88,bottom=.23,wspace=.31)
    axes=(ax,detail)
    casecolors={'AM':'#24639b','AFM':'#cf7831','FM':'#7354a4'}
    fills={'stable node':'#dce9f5','stable focus':'#cce8e6','saddle':'#f3e4bf','unstable node':'#f9dfc9','unstable focus':'#f4d3d8'}
    def regions(axis,xlim,ylim):
        t=np.linspace(*xlim,1200);d=np.linspace(*ylim,600);T,D=np.meshgrid(t,d)
        types=np.where(D<0,2,np.where(T<0,np.where(D>T*T/4,1,0),np.where(D>T*T/4,4,3)))
        from matplotlib.colors import ListedColormap
        axis.pcolormesh(t,d,types,cmap=ListedColormap(list(fills.values())),shading='nearest',rasterized=True,zorder=0)
        axis.axhline(0,color='.35',lw=.8,zorder=1);axis.axvline(0,color='.45',lw=.7,ls='--',zorder=1)
        axis.plot(t,t*t/4,color='.38',lw=.8,ls=':',zorder=1)
        axis.set_xlim(*xlim);axis.set_ylim(*ylim);axis.grid(alpha=.12)
    regions(ax,(-12.4,4.2),(-.15,.78));regions(detail,(-1.3,1.5),(-.13,.69))
    for axis,title in zip(axes,('a   Full equilibrium branches','b   Transition region')):
        axis.set(xlabel=r'Trace $T=\operatorname{tr}J$ (s$^{-1}$)',ylabel=r'Determinant $D=\det J$ (s$^{-2}$)',title=title)
        axis.set_box_aspect(1)
    fig.suptitle('Li–Rinzel equilibrium spectra in the trace–determinant plane',fontsize=10)
    records=[]
    for case in ('AM','AFM','FM'):
        eq=np.load(lr.DATA/f'{case}_equilibria.npz');I,C,h=eq['I'],eq['C'],eq['h']
        eig=np.linalg.eigvals(np.moveaxis(lr.jacobian(case,[C,h],I),-1,0).astype(complex))
        order=np.lexsort((eig.real,eig.imag),axis=1)
        eig=np.take_along_axis(eig,order,axis=1)
        det=eig[:,0]*eig[:,1];tr=(eig[:,0]+eig[:,1]).real;disc=tr**2-4*det.real
        typ=np.where(det.real<0,'saddle',np.where(tr<0,np.where(disc<0,'stable focus','stable node'),np.where(disc<0,'unstable focus','unstable node')))
        for axis in axes:
            axis.plot(tr,det.real,color=casecolors[case],lw=1.7,zorder=3)
            # The equilibrium branch is parameterized by C, whose direction can
            # reverse in I at a fold.  Orient each arrow by local dI/dC.
            for fraction in (.16,.32,.50,.70,.88):
                k=int(fraction*(len(C)-15));m=min(k+12,len(C)-1)
                if abs(I[m]-I[k])<1e-6:continue
                start,end=(k,m) if I[m]>I[k] else (m,k)
                axis.annotate('',xy=(tr[end],det.real[end]),xytext=(tr[start],det.real[start]),
                    arrowprops=dict(arrowstyle='-|>',color=casecolors[case],lw=1,mutation_scale=9),zorder=5)
            axis.plot(tr[0],det.real[0],'o',color=casecolors[case],ms=3,zorder=4)
        for i in range(len(C)):
            records.append(dict(case=case,I=I[i],C=C[i],h=h[i],trace=tr[i],determinant=det.real[i],lambda1_real=eig[i,0].real,lambda1_imag=eig[i,0].imag,lambda2_real=eig[i,1].real,lambda2_imag=eig[i,1].imag,type=typ[i]))
    handles=[Line2D([],[],color=casecolors[c],lw=2,label=rf'{c}, $K_{{ER}}={lr.PARAMETERS["KER_"+c]:g}\,\mu$M') for c in casecolors]
    ax.legend(handles=handles,loc='upper left',fontsize=7,framealpha=.9,title='LR regime')
    fig.legend(handles=[Patch(facecolor=c,edgecolor='.6',label=k) for k,c in fills.items()],loc='lower center',bbox_to_anchor=(.5,.06),ncol=5,frameon=False,fontsize=8)
    fig.text(.5,.025,'Arrows indicate increasing I; dots mark I ≈ 0.  Dashed: T = 0; dotted: D = T² / 4.',ha='center',fontsize=8)
    save(fig,'figure2_lr_spectrum')
    with (OUT/'figure2_lr_spectrum.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def common(ax,thresholds):
    ax.set(xlim=(-1.9,1.9),ylim=(-1.05,1.05),xlabel='Switching coordinate $z$',ylabel='Hidden coordinate $q$')
    for x in thresholds:ax.axvline(x,color='#22a3c2',lw=1,ls='--')
    ax.set_xticks([]);ax.set_yticks([]);ax.spines[['top','right']].set_visible(False)

def arrow_ellipse(ax,cx,rx,ry,linestyle='-',color='#222'):
    ax.add_patch(Ellipse((cx,0),2*rx,2*ry,fill=False,color=color,lw=1.7,ls=linestyle))
    t=1.0;p=lambda x:(cx+rx*np.cos(x),ry*np.sin(x))
    ax.add_patch(FancyArrowPatch(p(t),p(t+.22),arrowstyle='-|>',mutation_scale=10,color=color,lw=1))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--spectrum-only',action='store_true');args=parser.parse_args()
    spectrum()
    if not args.spectrum_only:
        from paper_hlb_examples import main as generate_hlb
        from paper_bc_examples import main as generate_large_cycles
        generate_hlb()
        generate_large_cycles()
