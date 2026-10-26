#!/usr/bin/env python3
"""Track all physical Li--Rinzel equilibrium zeros over (I, K_ER)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CMAX = 2.0/(1.0+0.185)


def gate(C, I, p):
    q = p['d_2']*(I+p['d_1'])/(I+p['d_3'])
    return q/(q+C) if q+C else 0.0


def calcium_field(C, h, I, K, p):
    receptor = I/(I+p['d_1']) if I+p['d_1'] else 0.0
    open_fraction = C/(C+p['d_5']) if C+p['d_5'] else 0.0
    return ((p['r_C']*receptor**3*open_fraction**3*h**3+p['r_L'])
            *(p['C_0']-(1+p['c_1'])*C)-p['v_ER']*C*C/(C*C+K*K))


def jacobian(C, h, I, K, p):
    q = p['d_2']*(I+p['d_1'])/(I+p['d_3'])
    m = I/(I+p['d_1'])
    n = C/(C+p['d_5'])
    dcc = (p['r_C']*m**3*h**3*(
        3*n*n*p['d_5']/(C+p['d_5'])**2*(p['C_0']-(1+p['c_1'])*C)
        -(1+p['c_1'])*n**3)
        -2*p['v_ER']*C*K*K/(C*C+K*K)**2)
    dch = 3*p['r_C']*m**3*n**3*h*h*(p['C_0']-(1+p['c_1'])*C)
    return np.array([[dcc, dch], [-p['a_2']*h, -p['a_2']*(q+C)]])


def classify(J):
    trace = float(np.trace(J)); determinant = float(np.linalg.det(J))
    discriminant = trace*trace-4*determinant
    if determinant < -1e-10:
        kind = 'saddle'
    elif determinant <= 1e-10 or abs(trace) <= 1e-10:
        kind = 'nonhyperbolic'
    elif trace < 0:
        kind = 'stable_node' if discriminant >= 0 else 'stable_focus'
    else:
        kind = 'unstable_node' if discriminant >= 0 else 'unstable_focus'
    return trace, determinant, discriminant, kind


def equilibria_at(I, K, p, c_grid):
    values = np.asarray([calcium_field(float(C), gate(float(C), I, p), I, K, p)
                         for C in c_grid])
    roots = []
    for j in range(len(c_grid)-1):
        left, right = float(c_grid[j]), float(c_grid[j+1])
        vl, vr = float(values[j]), float(values[j+1])
        if vl == 0:
            root = left
        elif vl*vr < 0:
            root = float(brentq(lambda C: calcium_field(C, gate(C, I, p), I, K, p),
                                left, right, xtol=2e-13))
        else:
            continue
        if not roots or abs(root-roots[-1]['C']) > 2e-8:
            h = gate(root, I, p)
            trace, det, disc, kind = classify(jacobian(root, h, I, K, p))
            roots.append(dict(I=float(I), K_ER=float(K), C=float(root), h=float(h),
                              trace=trace, determinant=det, discriminant=float(disc),
                              type=kind))
    if abs(float(values[-1])) < 1e-11:
        root = float(c_grid[-1]);h=gate(root,I,p)
        tr,det,disc,kind=classify(jacobian(root,h,I,K,p))
        if not roots or abs(root-roots[-1]['C'])>2e-8:
            roots.append(dict(I=float(I),K_ER=float(K),C=root,h=float(h),trace=tr,
                              determinant=det,discriminant=float(disc),type=kind))
    return roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path,
        default=ROOT/'results'/'equilibrium_zeros_K_sweep')
    parser.add_argument('--k-min', type=float, default=.025)
    parser.add_argument('--k-max', type=float, default=.2)
    parser.add_argument('--n-k', type=int, default=36)
    parser.add_argument('--i-min', type=float, default=0.)
    parser.add_argument('--i-max', type=float, default=2.)
    parser.add_argument('--n-i', type=int, default=241)
    parser.add_argument('--n-c', type=int, default=1601)
    args = parser.parse_args()
    case_data=json.loads((ROOT.parent/'fm_am_reduction'/'config'/'cases.json').read_text())
    p=case_data['shared_parameters']
    k_values=np.unique(np.r_[np.linspace(args.k_min,args.k_max,args.n_k),.05,.1])
    i_values=np.linspace(args.i_min,args.i_max,args.n_i)
    # Geometric spacing resolves the low-calcium roots; linear spacing covers
    # the upper physical interval without leaving gaps near the saturation edge.
    c_grid=np.unique(np.r_[np.geomspace(1e-9,.03,max(100,args.n_c//3)),
                           np.linspace(.03,CMAX-1e-9,args.n_c)])
    rows=[];counts=np.zeros((len(k_values),len(i_values)),dtype=int)
    roots_grid=[[[] for _ in i_values] for _ in k_values]
    for ki,K in enumerate(k_values):
        for ii,I in enumerate(i_values):
            roots=equilibria_at(float(I),float(K),p,c_grid)
            counts[ki,ii]=len(roots);rows.extend(roots);roots_grid[ki][ii]=roots
        print(f'K_ER={K:.6f}: equilibria {int(counts[ki].sum())} across {len(i_values)} I values',flush=True)

    args.out_dir.mkdir(parents=True,exist_ok=True)
    csv_path=args.out_dir/'equilibrium_zeros.csv'
    fields=['K_ER','I','C','h','trace','determinant','discriminant','type']
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
    representative=np.unique(np.r_[.025,.05,.075,.1,.125,.15,.2])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    colors=plt.cm.viridis(np.linspace(.08,.94,len(representative)))
    for color,K in zip(colors,representative):
        ix=int(np.argmin(abs(k_values-K)))
        # Roots are sorted by C at each I; NaNs separate branches when a root
        # is absent. This view is descriptive, not a branch continuation.
        max_roots=int(counts[ix].max())
        for branch in range(max_roots):
            vals=[]
            for ii,I in enumerate(i_values):
                q=sorted(roots_grid[ix][ii],key=lambda z:z['C'])
                vals.append(q[branch]['C'] if branch<len(q) else np.nan)
            axes[0].plot(i_values,vals,color=color,lw=1.3,label=f'K={k_values[ix]:.3g}' if branch==0 else None)
    axes[0].axvline(.05,color='black',ls=':',lw=.8);axes[0].axvline(.1,color='black',ls='--',lw=.8)
    axes[0].set(xlabel='I',ylabel='Equilibrium C* (µM)',title='LR equilibrium zeros vs I at K_ER slices')
    axes[0].legend(fontsize=7,ncol=2);axes[0].grid(alpha=.2)
    im=axes[1].imshow(counts,origin='lower',aspect='auto',interpolation='nearest',
        extent=[i_values[0],i_values[-1],k_values[0],k_values[-1]],cmap='viridis',
        vmin=.5,vmax=3.5)
    axes[1].set(xlabel='I',ylabel='K_ER',title='Number of physical equilibria')
    fig.colorbar(im,ax=axes[1],label='root count')
    selected_I=[.35,.5,.65,.8,1.0,1.2]
    for I in selected_I:
        nearest=int(np.argmin(abs(i_values-I)))
        vals=[]
        for ki,K in enumerate(k_values):
            matching=roots_grid[ki][nearest]
            vals.append(sorted(q['C'] for q in matching))
        for branch in range(max(map(len,vals),default=0)):
            axes[2].plot(k_values,[v[branch] if branch<len(v) else np.nan for v in vals],
                         marker='.',ms=2,lw=1,label=f'I={i_values[nearest]:.2f}' if branch==0 else None)
    axes[2].axvline(.05,color='black',ls=':',lw=.8,label='FM K_ER')
    axes[2].axvline(.1,color='black',ls='--',lw=.8,label='AM K_ER')
    axes[2].set(xlabel='K_ER',ylabel='Equilibrium C* (µM)',title='Zero branches as K_ER varies')
    axes[2].legend(fontsize=7,ncol=2);axes[2].grid(alpha=.2)
    fig.savefig(args.out_dir/'equilibrium_zeros_vs_K.png',dpi=180)
    fig.savefig(args.out_dir/'equilibrium_zeros_vs_K.pdf');plt.close(fig)
    summary=dict(K_range=[float(k_values[0]),float(k_values[-1])],K_values=len(k_values),
        I_range=[float(i_values[0]),float(i_values[-1])],I_values=len(i_values),
        max_physical_root_count=int(counts.max()),
        root_count_transition_cells=int(np.sum(np.diff(counts,axis=0)!=0)+np.sum(np.diff(counts,axis=1)!=0)),
        roots=len(rows),FM_reference_K=.05,AM_reference_K=.1,
        note='Finite-grid LR equilibrium enumeration by scalar C-nullcline roots; verify folds with refinement.')
    (args.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
