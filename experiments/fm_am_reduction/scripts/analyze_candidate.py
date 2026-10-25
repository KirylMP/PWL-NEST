#!/usr/bin/env python3
"""Independent finite-grid equilibrium and observed-orbit diagnostics.

Does not enumerate repelling cycles and does not certify bifurcations.
"""
import argparse
import os
import tempfile
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(),"pwl-fm-cache"))
os.makedirs(os.environ["XDG_CACHE_HOME"],exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(),"pwl-fm-matplotlib"))
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_stage import phi, hinge_design, model_roots
from prepare_reference_data import lr_field, roots_for_ip3
from plan import read_json


def vector(model):
    tr = model['transform']
    return np.r_[model['alpha'], model['beta'], np.log(model['a']), np.log(model['b']),
                 tr['p'], tr['e'], np.log(tr['n']), tr['k']]


def orbit(rhs, initial, duration, bounds, inverse=None):
    evaluations = 0
    def bounded(t, y):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 200000:
            raise RuntimeError('RHS evaluation budget exceeded')
        return rhs(t, y)
    def exit_box(t, y):
        return min(y[0]-bounds[0], bounds[1]-y[0], y[1], 1-y[1])
    exit_box.terminal = True
    exit_box.direction = -1
    try:
        sol = solve_ivp(bounded, [0,duration], initial, method='DOP853',
                        rtol=1e-8, atol=1e-10, max_step=.5,
                        t_eval=np.arange(0,duration+.01,.2), events=exit_box)
        if not sol.success or sol.t[-1] < duration-.21:
            return dict(status='failed_or_domain_exit', message=sol.message)
        c = inverse(sol.y[0]) if inverse else sol.y[0]
        mask = sol.t >= duration/2
        c, t = c[mask], sol.t[mask]
        peaks = find_peaks(c, prominence=1e-4)[0]
        result = dict(Cmin=float(c.min()), Cmax=float(c.max()), status='unresolved', period=None)
        if np.ptp(c) < 1e-5:
            result['status'] = 'rest'
        elif len(peaks) >= 5:
            periods = np.diff(t[peaks[-5:]])
            first, last = c[:len(c)//2], c[len(c)//2:]
            settled = abs(np.ptp(first)-np.ptp(last)) < max(1e-5,.02*np.ptp(c))
            if np.std(periods)/np.mean(periods) < .02 and settled:
                result.update(status='observed_periodic_orbit', period=float(np.mean(periods)))
        return result
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        return dict(status='integration_failed', message=str(exc))


def scan_point(task):
    ip3, model, p, duration = task
    cmax = p['C_0']/(1+p['c_1'])
    tr = model['transform']; eta = vector(model)[-4:]; cr = tr['C_ref']
    ct = np.asarray(model['physical_thresholds_C']); zt = phi(ct,eta,cr)
    alpha, beta = np.asarray(model['alpha']), np.asarray(model['beta'])
    ns = alpha[1]+np.r_[0.,np.cumsum(alpha[3:])]
    hs = beta[1]+np.r_[0.,np.cumsum(beta[3:])]
    eq = []
    for c,h in model_roots(vector(model),ct,ip3,cr,1e-7,cmax-1e-7):
        regions = [int(np.searchsorted(ct,c))]
        if np.min(abs(ct-c)) < 1e-7:
            k = int(np.argmin(abs(ct-c))); regions = [k,k+1]
        sides = []
        for r in regions:
            trace = -model['a']*ns[r]-model['b']
            det = model['a']*model['b']*(ns[r]-hs[r])
            sides.append(dict(trace=float(trace),determinant=float(det),discriminant=float(trace**2-4*det)))
        eq.append(dict(C=c,h=h,I=ip3,physical=bool(0<h<1),on_threshold=len(regions)>1, sides=sides))
    lr_eq = roots_for_ip3(ip3,p,cmax,3200)
    # Independently sampled, shifted I grid. No validation data enters selection.
    cg = np.unique(np.r_[np.geomspace(1e-7,.03,1500),np.linspace(.03,cmax-1e-7,15000)])
    zg = phi(cg,eta,cr)
    inverse = lambda z: np.interp(z,zg,cg)
    ni = alpha[0]+alpha[2]*ip3-np.r_[0., np.cumsum(alpha[3:]*zt)]
    hi = beta[0]+beta[2]*ip3-np.r_[0., np.cumsum(beta[3:]*zt)]
    def pwa_rhs(t,y):
        r = np.searchsorted(zt,y[0],side='right')
        z = y[0]
        hn = ni[r]+ns[r]*z
        hh = hi[r]+hs[r]*z
        return [model['a']*(y[1]-hn), model['b']*(hh-y[1])]
    # Multiple initial states expose some coexistence; they cannot find repelling cycles.
    traces = {'LR':[], 'PWA':[]}
    for c,h in [(.04,.8),(.35,.25),(min(1.,.8*cmax),.65)]:
        traces['LR'].append(orbit(lambda t,y: lr_field(y[0],y[1],ip3,p),[c,h],duration,[1e-7,cmax-1e-7]))
        traces['PWA'].append(orbit(pwa_rhs,[float(phi(np.array([c]),eta,cr)[0]),h],duration,[zg[0],zg[-1]],inverse))
    return dict(I=ip3,LR_equilibria=lr_eq,PWA_equilibria=eq,orbits=traces)


def stability(q):
    if q.get('on_threshold'):
        return 'threshold'
    if not q.get('physical',True):
        return 'outside'
    side = q.get('sides',[q])[0]
    if abs(side['determinant'])<1e-9 or (side['determinant']>0 and abs(side['trace'])<1e-8):
        return 'degenerate'
    return 'saddle' if side['determinant']<0 else ('stable' if side['trace']<0 else 'unstable')


def analyze(run_dir, workers=1, points=81, duration=1200.):
    out = run_dir/'analysis'; out.mkdir(exist_ok=True)
    model = read_json(run_dir/'optimizer'/'selected.json')['candidate']
    p = read_json(run_dir/'run.json')['parameters']
    # Midpoints avoid copying the training I grid.
    levels = (np.arange(points)+.5)*2/points
    tasks = [(float(i),model,p,duration) for i in levels]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(scan_point,tasks):
            rows.append(row)
            with (out/'scan.json').open('w') as f:
                json.dump(rows,f,indent=2,allow_nan=False)
            print(f"analysis I={row['I']:.5f} ({len(rows)}/{points})",flush=True)
    markers={'stable':'o','unstable':'x','saddle':'^','degenerate':'s','threshold':'D','outside':'+'}
    fig, axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey="row")
    for col,key in enumerate(['LR','PWA']):
        for kind,marker in markers.items():
            values=[(q['I'],q['C']) for r in rows for q in r[key+'_equilibria'] if stability(q)==kind]
            if values:
                axes[0,col].scatter(*np.array(values).T,s=10,marker=marker,label=kind)
        for r in rows:
            for q in r['orbits'][key]:
                if q['status']=='observed_periodic_orbit':
                    axes[0,col].plot([r['I']]*2,[q['Cmin'],q['Cmax']],'.',color='tab:red',ms=3)
                    axes[1,col].plot(r['I'],q['period'],'.',color='tab:red',ms=3)
        axes[0,col].set_title(key+': equilibria and observed cycle extrema')
        axes[0,col].plot([],[],'.',color='tab:red',label='observed cycle extrema')
        axes[0,col].set_ylabel('C (uM)'); axes[0,col].legend(fontsize=8)
        bad=[r['I'] for r in rows if any(q['status'] not in ['rest','observed_periodic_orbit'] for q in r['orbits'][key])]
        if bad:
            axes[1,col].scatter(bad,[.03]*len(bad),marker='x',color='gray',s=10,
                transform=axes[1,col].get_xaxis_transform(),label='unresolved/failed orbit')
            axes[1,col].legend(fontsize=8)
        axes[1,col].set_ylabel('Observed period (s)'); axes[1,col].set_xlabel('I (uM)')
    fig.suptitle('Finite-grid bifurcation diagnostics; repelling cycles not enumerated')
    fig.tight_layout(); fig.savefig(out/'bifurcation.png',dpi=180); fig.savefig(out/'bifurcation.pdf')
    for ax in axes.flat:ax.set_xlim(.3,1.2)
    fig.savefig(out/'bifurcation_zoom.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for key,color in [('LR','tab:blue'),('PWA','tab:orange')]:
        for row in rows:
            for q in row[key+'_equilibria']:
                for side in q.get('sides',[q]):
                    for ax,field in zip(axes,['trace','determinant','discriminant']):
                        ax.plot(q['I'],side[field],'.',color=color,ms=2)
    for key,color in [('LR','tab:blue'),('PWA','tab:orange')]:axes[0].plot([],[],'.',color=color,label=key)
    axes[0].legend()
    for ax,title in zip(axes,['trace','determinant','discriminant']):
        ax.axhline(0,color='k',lw=.5);ax.set_title(title);ax.set_xlabel('I')
    fig.tight_layout();fig.savefig(out/'spectra.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    c=np.linspace(.03,1.15,700);tr=model['transform'];eta=vector(model)[-4:]
    for ax,ip3 in zip(axes.flat,[.4,.6,.9,1.2]):
        q=p['d_2']*(ip3+p['d_1'])/(ip3+p['d_3']); h=q/(q+c)
        rad=(p['v_ER']*c*c/((c*c+p['K_ER']**2)*(p['C_0']-(1+p['c_1'])*c))-p['r_L'])/(p['r_C']*(ip3/(ip3+p['d_1']))**3*(c/(c+p['d_5']))**3)
        X=hinge_design(phi(c,eta,tr['C_ref']),np.full_like(c,ip3),phi(np.asarray(model['physical_thresholds_C']),eta,tr['C_ref']))
        ax.plot(c,np.cbrt(rad),label='LR N');ax.plot(c,h,label='LR H')
        ax.plot(c,X@model['alpha'],'--',label='PWA N');ax.plot(c,X@model['beta'],'--',label='PWA H')
        ax.set_title(f'I={ip3}');ax.set_ylim(-.3,1.3);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'nullclines.png',dpi=180);plt.close(fig)
    comparisons=[]; ext_errors=[]; period_errors=[]
    for row in rows:
        cycles={}
        for key in ['LR','PWA']:
            cycles[key]=[]
            for q in row['orbits'][key]:
                if q['status']!='observed_periodic_orbit':continue
                if not any(abs(q['Cmin']-r['Cmin'])<.001 and abs(q['Cmax']-r['Cmax'])<.001
                           and abs(q['period']/r['period']-1)<.02 for r in cycles[key]):
                    cycles[key].append(q)
        nlr,npwa=len(cycles['LR']),len(cycles['PWA'])
        if nlr and npwa:
            costs=np.array([[((a['Cmin']-b['Cmin'])/.05)**2+((a['Cmax']-b['Cmax'])/.05)**2
                             +((a['period']-b['period'])/5)**2 for b in cycles['PWA']] for a in cycles['LR']])
            ai,bi=linear_sum_assignment(costs)
            for a,b in zip(ai,bi):
                ca,cb=cycles['LR'][a],cycles['PWA'][b]
                ext_errors.extend([cb['Cmin']-ca['Cmin'],cb['Cmax']-ca['Cmax']])
                period_errors.append(cb['period']-ca['period'])
        comparisons.append(dict(I=row['I'],LR_observed_periodic_orbits=nlr,PWA_observed_periodic_orbits=npwa,
            complete_orbit_classification=all(q['status'] in ['rest','observed_periodic_orbit'] for key in ['LR','PWA'] for q in row['orbits'][key])))
    (out/'observed_cycle_comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    failures={key:sum(q['status'] not in ['rest','observed_periodic_orbit'] for r in rows for q in r['orbits'][key]) for key in ['LR','PWA']}
    summary=dict(scope='independent finite-grid diagnostics, no complete cycle enumeration',
        points=points,duration_s=duration,initial_states_per_I=3,unresolved_or_failed_orbits=failures,
        equilibrium_count_mismatch_points=sum(sum(stability(q)!='outside' for q in r['PWA_equilibria'])!=len(r['LR_equilibria']) for r in rows),
        candidate_selected_using_validation=False,repelling_cycles_searched=False,
        observed_cycle_count_mismatch_points=sum(q['LR_observed_periodic_orbits']!=q['PWA_observed_periodic_orbits'] for q in comparisons if q['complete_orbit_classification']),
        inconclusive_cycle_comparison_points=sum(not q['complete_orbit_classification'] for q in comparisons),
        matched_observed_cycles=len(period_errors),
        extrema_RMSE_uM=float(np.sqrt(np.mean(np.square(ext_errors)))) if ext_errors else None,
        period_RMSE_s=float(np.sqrt(np.mean(np.square(period_errors)))) if period_errors else None,
        transition_locations='only grid intervals; no bifurcation certification')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'orbits.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['I','model','initial_index','status','Cmin','Cmax','period','message']);writer.writeheader()
        for r in rows:
            for key in ['LR','PWA']:
                for j,q in enumerate(r['orbits'][key]):writer.writerow(dict(I=r['I'],model=key,initial_index=j,**q))
    trial_records=[]
    for path in sorted((run_dir/'optimizer'/'trials').glob('trial_*/result.json')):
        q=read_json(path)
        trial_records.append(dict(trial=q['trial'],loss=q['parts']['precycle_total'],
                                  success=q['success'],nfev=q['nfev'],optimality=q['optimality']))
    if trial_records:
        fig,ax=plt.subplots(figsize=(9,4))
        for success,color in [(True,'tab:blue'),(False,'tab:orange')]:
            group=[q for q in trial_records if q['success']==success]
            if group:ax.scatter([q['trial']+1 for q in group],[q['loss'] for q in group],
                                label='optimizer converged' if success else 'budget/termination',color=color)
        ax.set_xlabel('Restart');ax.set_ylabel('Pre-cycle training loss');ax.set_yscale('log');ax.legend()
        fig.tight_layout();fig.savefig(out/'trial_losses.png',dpi=180);plt.close(fig)
        (out/'trial_scores.json').write_text(json.dumps(trial_records,indent=2)+'\n')
    print('Analysis saved:',out,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--points',type=int,default=81);parser.add_argument('--duration',type=float,default=1200.)
    args=parser.parse_args();analyze(args.run_dir,args.workers,args.points,args.duration)
