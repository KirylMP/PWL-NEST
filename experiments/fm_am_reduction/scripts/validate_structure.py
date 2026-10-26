"""Independent multi-section root enumeration and neighboring-I continuation.

Finite grids are evidence, not a proof of global cycle completeness. Missing
regular return intervals remain unresolved rather than being counted as zero.
"""
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from fit_stage import model_roots,NonisolatedEquilibria
from prepare_reference_data import roots_for_ip3
from poincare import Section,LRSystem,PWASystem,isolate_cycles
from analyze_candidate import vector,stability


def same_orbit(a,b):
    """Compare closed orbits, including common oriented section intersections."""
    if a['kind']!=b['kind']:return False
    if abs(a['period']/b['period']-1)>=.003:return False
    if max(abs(a['Cmin']-b['Cmin']),abs(a['Cmax']-b['Cmax']))>=.001:return False
    common=[]
    for c,values in a.get('orbit_crossings',{}).items():
        other=b.get('orbit_crossings',{}).get(c,[])
        if values and other:
            if len(values)!=len(other) or max(abs(np.asarray(values)-other))>2e-7:return False
            common.append(c)
    # No merge based solely on nearby periods and amplitudes.
    return bool(common)


def _validate_point(task):
    ip3,model,p,nodes,solver,seeds=task;cmax=p['C_0']/(1+p['c_1'])
    source=roots_for_ip3(ip3,p,cmax,5000);output=dict(I=ip3,LR_equilibria=source)
    theta=vector(model);ct=np.asarray(model['physical_thresholds_C']);cr=model['transform']['C_ref']
    systems={'LR':LRSystem(p,ip3),'PWA':PWASystem(theta,ct,ip3,cr,cmax)}
    degenerate=None
    try:raw=model_roots(theta,ct,ip3,cr,1e-8,cmax-1e-8)
    except NonisolatedEquilibria as exc:raw=[];degenerate=str(exc)
    eq=[]
    for c,h in raw:
        regions=[int(np.searchsorted(ct,c,side='right'))]
        close=np.flatnonzero(abs(ct-c)<1e-7)
        if len(close):regions=[int(close[0]),int(close[0])+1]
        sides=[];system=systems['PWA']
        for r in regions:
            tr=-system.a*system.ns[r]-system.b;det=system.a*system.b*(system.ns[r]-system.hs[r])
            sides.append(dict(trace=float(tr),determinant=float(det),discriminant=float(tr*tr-4*det)))
        eq.append(dict(C=c,h=h,I=ip3,physical=bool(0<c<cmax and 0<h<1),on_threshold=len(regions)>1,sides=sides))
    output['PWA_equilibria']=eq;output['nonisolated_equilibria']=degenerate
    # Small candidate cycles may not intersect the LR equilibrium section.
    sections=sorted(set([.06,.2,.5,.9]+[q['C'] for q in source+eq
                                      if .02<q['C']<cmax-.02 and 0<q['h']<1]))
    for key,system in systems.items():
        cycles=[];scans=[];regular=0;unresolved=[]
        capture=tuple(np.array([system.coordinate(q['C']),q['h']]) for q in output[key+'_equilibria']
                      if 0<q['C']<cmax and 0<q['h']<1 and stability(q)=='stable')
        reverse_capture=tuple(np.array([system.coordinate(q['C']),q['h']]) for q in output[key+'_equilibria']
                      if 0<q['C']<cmax and 0<q['h']<1
                      and np.min(np.linalg.eigvals(system.jac([system.coordinate(q['C']),q['h']])).real)>1e-8)
        section_values=sorted(set(sections+[q['section_C'] for q in seeds.get(key,[])]))
        for csec in section_values:
            h0=system.nullcline(csec)
            if h0 is None or not np.isfinite(h0) or h0>=1-1e-5:
                unresolved.append(dict(C=csec,status='no_physical_launch_interval'));continue
            h0=max(0.,h0);section=Section(csec,h0)
            guesses=[]
            for q in seeds.get(key,[]):
                if abs(q['section_C']-csec)<1e-7:
                    u=(q['s']-h0)/(1-h0)
                    guesses.extend([u-.01,u,u+.01])
            grid=np.unique(np.r_[nodes,[u for u in guesses if 1e-5<u<.999]])
            scan=isolate_cycles(system,section,grid,{**solver,'capture_points':capture,'witness_sections':section_values},
                                reverse_capture_points=reverse_capture)
            regular+=scan['regular_edges']
            scans.append(dict(C=csec,h0=h0,**scan))
            for orbit in scan['cycles']:
                duplicate=next((q for q in cycles if same_orbit(orbit,q)),None)
                if duplicate is None:
                    orbit['observations']=[dict(section_C=csec,s=orbit['s'],time_direction=orbit['time_direction'])]
                    cycles.append(orbit)
                else:duplicate['observations'].append(dict(section_C=csec,s=orbit['s'],time_direction=orbit['time_direction']))
        # A uniformly signed PWA divergence excludes cycles in the physical rectangle.
        bendixson=key=='PWA' and (np.all(-system.a*system.ns-system.b<-1e-8) or np.all(-system.a*system.ns-system.b>1e-8))
        status='enumerated_on_grid' if regular else ('excluded_by_divergence' if bendixson else 'unresolved_no_regular_returns')
        output[key]=dict(cycles=cycles,sections=scans,regular_edges=regular,status=status,
                        skipped_sections=unresolved,divergence_exclusion=bool(bendixson))
    return output


def validate(run_dir,workers=1,points=81,nodes=33,max_time=600.,output_dir=None):
    from plan import read_json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if points<1 or nodes<5 or workers<1 or max_time<=0:raise ValueError('positive budgets and at least five section nodes required')
    run=read_json(run_dir/'run.json');model=read_json(run_dir/'optimizer'/'selected.json')['candidate']
    out=Path(output_dir) if output_dir is not None else run_dir/'analysis'/'structure'
    out.mkdir(parents=True,exist_ok=output_dir is None)
    levels=(np.arange(points)+.5)*2/points
    unodes=np.unique(np.r_[np.geomspace(.0002,.08,max(4,nodes//3)),np.linspace(.08,.995,nodes-max(4,nodes//3)+1)])
    # Tighter tolerances and denser section grids than the training maps.
    solver=dict(max_time=max_time,rtol=3e-10,atol=3e-12,max_step=.25,max_evaluations=200000)
    def tasks(seed_rows=None):
        for j,i in enumerate(levels):
            seeds={key:[] for key in ['LR','PWA']}
            if seed_rows:
                for k in [j-1,j+1]:
                    if 0<=k<len(seed_rows):
                        for key in seeds:seeds[key].extend(seed_rows[k][key]['cycles'])
            yield (float(i),model,run['parameters'],unodes.tolist(),solver,seeds)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=[]
        for row in pool.map(_validate_point,tasks()):
            rows.append(row)
            (out/'scan_initial.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
            print(f"Structure scan {len(rows)}/{points}, I={row['I']:.6f}",flush=True)
        refined=[]
        for row in pool.map(_validate_point,tasks(rows)):
            refined.append(row)
            (out/'scan_continued.json').write_text(json.dumps(refined,indent=2,allow_nan=False))
            print(f"Neighbor-I continuation {len(refined)}/{points}",flush=True)
    rows=refined;comparisons=[];ext=[];period=[];missing=extra=0
    for row in rows:
        signatures={};known=True
        for key in ['LR','PWA']:
            eq=row[key+'_equilibria'];cycles=row[key]['cycles']
            stable=sum(stability(q)=='stable' for q in eq)
            unstable=sum(stability(q) in ['unstable','saddle'] for q in eq)
            ns=sum(q['kind']=='stable' for q in cycles);nu=sum(q['kind']=='unstable' for q in cycles)
            signatures[key]=[stable,unstable,ns,nu]
            known &= row[key]['status']!='unresolved_no_regular_returns'
            known &= not any(stability(q) in ['threshold','degenerate'] for q in eq)
            known &= not any(q['kind']=='degenerate' for q in cycles)
            known &= not any(s['suspect_intervals'] for s in row[key]['sections'])
        known &= row['nonisolated_equilibria'] is None
        comparisons.append(dict(I=row['I'],signatures=signatures,regular=bool(known),
            squared_error=int(np.sum((np.asarray(signatures['LR'])-signatures['PWA'])**2)) if known else None))
        for kind in ['stable','unstable']:
            a=[q for q in row['LR']['cycles'] if q['kind']==kind];b=[q for q in row['PWA']['cycles'] if q['kind']==kind]
            if a and b:
                costs=np.array([[((x['Cmin']-y['Cmin'])/.05)**2+((x['Cmax']-y['Cmax'])/.05)**2+((x['period']-y['period'])/5)**2 for y in b] for x in a])
                ai,bi=linear_sum_assignment(costs)
                for i,j in zip(ai,bi):
                    ext.extend([b[j]['Cmin']-a[i]['Cmin'],b[j]['Cmax']-a[i]['Cmax']]);period.append(b[j]['period']-a[i]['period'])
                missing+=len(a)-len(ai);extra+=len(b)-len(bi)
            else:missing+=len(a);extra+=len(b)
    regular=[q['squared_error'] for q in comparisons if q['regular']]
    transitions=[]
    for a,b in zip(comparisons[:-1],comparisons[1:]):
        for key in ['LR','PWA']:
            if a['regular'] and b['regular'] and a['signatures'][key]!=b['signatures'][key]:
                transitions.append(dict(model=key,I_interval=[a['I'],b['I']],left=a['signatures'][key],right=b['signatures'][key]))
    summary=dict(scope='finite-grid multi-section first-return roots with neighbor-I predictor/corrector seeds',
        validation_version='bidirectional_closed_orbits_v4',
        validation_code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        candidate_sha256=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest(),
        points=points,section_nodes=len(unodes),solver=solver,repelling_cycles_searched=True,
        regular_comparison_points=len(regular),unresolved_or_degenerate_points=points-len(regular),
        L_str_val_on_regular_subset=float(np.mean(regular)) if regular else None,
        L_str_val_full_grid=float(np.mean(regular)) if len(regular)==points else None,
        matched_cycles=len(period),missing_observed_cycles=missing,extra_observed_cycles=extra,
        extrema_RMSE_uM=float(np.sqrt(np.mean(np.square(ext)))) if ext else None,
        period_RMSE_s=float(np.sqrt(np.mean(np.square(period)))) if period else None,
        transition_intervals=transitions,global_completeness_proven=False,
        deduplication=dict(period_relative=.003,extrema_absolute_uM=.001,common_section_h_absolute=2e-7,
                           same_stability_required=True,closure_checked=True,unstable_shooting='backward_time'),
        candidate_selected_using_validation=False,
        nonphysical_equilibrium_points=sum(any(not q.get('physical',True) for q in r['PWA_equilibria']) for r in rows))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (out/'structural_signatures.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey='row')
    for col,key in enumerate(['LR','PWA']):
        for kind,marker in [('stable','o'),('unstable','x'),('saddle','^')]:
            vals=[(r['I'],q['C']) for r in rows for q in r[key+'_equilibria'] if stability(q)==kind]
            if vals:axes[0,col].scatter(*np.array(vals).T,s=6,marker=marker,label='equilibrium '+kind)
        for kind,color,marker in [('stable','tab:red','.'),('unstable','tab:purple','x'),('degenerate','gray','s')]:
            for r in rows:
                for q in r[key]['cycles']:
                    if q['kind']!=kind:continue
                    axes[0,col].plot([r['I']]*2,[q['Cmin'],q['Cmax']],marker,color=color,ms=3)
                    axes[1,col].plot(r['I'],q['period'],marker,color=color,ms=3)
            axes[0,col].plot([],[],marker,color=color,label='cycle '+kind)
        bad=[r['I'] for r in rows if r[key]['status']=='unresolved_no_regular_returns']
        axes[1,col].scatter(bad,[.03]*len(bad),marker='x',color='gray',s=8,transform=axes[1,col].get_xaxis_transform())
        axes[0,col].set_title(key);axes[0,col].set_ylabel('C (uM)');axes[0,col].legend(fontsize=7)
        axes[1,col].set_xlabel('I (uM)');axes[1,col].set_ylabel('Period (s)')
    axes[0,0].set_ylim(0,run['parameters']['C_0']/(1+run['parameters']['c_1']))
    fig.suptitle('First-return root scan: stable and unstable cycles; gray marks unresolved domains')
    fig.tight_layout();fig.savefig(out/'bifurcation.png',dpi=180);fig.savefig(out/'bifurcation.pdf')
    for ax in axes.flat:ax.set_xlim(.3,1.2)
    fig.savefig(out/'bifurcation_zoom.png',dpi=180);plt.close(fig)
    print('Independent structural report:',out,flush=True)
    return summary

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16);parser.add_argument('--points',type=int,default=81)
    parser.add_argument('--nodes',type=int,default=33);parser.add_argument('--max-time',type=float,default=600)
    parser.add_argument('--output-dir',type=Path,help='fresh report directory; preserve previous validation')
    a=parser.parse_args();validate(a.run_dir,a.workers,a.points,a.nodes,a.max_time,a.output_dir)
