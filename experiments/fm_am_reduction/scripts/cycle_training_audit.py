"""Common collocation grids and explicit cycle-coverage/resolution checks."""
import json
import hashlib
from pathlib import Path
import numpy as np
from fit_stage import model_roots,hinge_design,phi
from prepare_reference_data import roots_for_ip3
from full_cycle_fit import reference_section,settings
from poincare import Section,LRSystem,PWASystem,return_table,smooth_counts,isolate_cycles
from moving_cycle_fit import anchor_section
from analyze_candidate import vector
from validate_structure import same_orbit


def nested_nodes(nodes,minimum):
    nodes=np.unique(np.asarray(nodes,dtype=float))
    while len(nodes)<minimum:nodes=np.unique(np.r_[nodes,(nodes[:-1]+nodes[1:])/2])
    return nodes


def collocation_levels(levels,max_gap=.05,warm_model=None):
    levels=sorted(set(map(float,levels)));coarse=list(levels)
    for a,b in zip(levels[:-1],levels[1:]):
        intervals=max(1,int(np.ceil((b-a)/max_gap-1e-10)))
        coarse.extend(np.linspace(a,b,intervals+1).tolist())
    events=[]
    if warm_model is not None:
        # Bracket equilibrium threshold events, avoiding a nonsmooth event node.
        x=vector(warm_model);n=len(warm_model['alpha']);d=x[:n]-x[n:2*n]
        zt=phi(np.asarray(warm_model['physical_thresholds_C']),x[-4:],warm_model['transform']['C_ref'])
        if abs(d[2])>1e-10:
            intercept=hinge_design(zt,np.zeros(len(zt)),zt)@d
            for event in -intercept/d[2]:
                if not levels[0]<event<levels[-1]:continue
                events.append(float(event))
                for offset in (-max_gap/10,max_gap/10):
                    if levels[0]<event+offset<levels[-1]:coarse.append(float(event+offset))
    coarse=np.unique(np.round(coarse,12))
    middle=[]
    for a,b in zip(coarse[:-1],coarse[1:]):
        mid=(a+b)/2
        if any(abs(mid-event)<1e-8 for event in events):middle.extend([a+.4*(b-a),a+.6*(b-a)])
        else:middle.append(mid)
    fine=np.unique(np.round(np.r_[coarse,middle],12))
    return coarse.tolist(),fine.tolist()


def _source_profile(ip3,p,nodes,solver):
    system=LRSystem(p,ip3);sec=reference_section(ip3,p)
    used_nodes=nodes;used_solver=solver
    roots=roots_for_ip3(ip3,p,p['C_0']/(1+p['c_1']),5000);stable=[];reverse=[]
    for q in roots:
        y=np.array([q['C'],q['h']]);eigenvalues=np.linalg.eigvals(system.jac(y)).real
        if max(eigenvalues)<-1e-8:stable.append(y)
        if min(eigenvalues)>1e-8:reverse.append(y)
    scan=isolate_cycles(system,sec,nodes,{**solver,'capture_points':tuple(stable)},reverse_capture_points=tuple(reverse))
    if scan['suspect_intervals'] or scan['regular_edges']==0:
        refined=nested_nodes(nodes,min(257,2*len(nodes)-1))
        used_nodes=refined;used_solver={**solver,'rtol':3e-10,'atol':3e-12,'max_step':.25}
        scan=isolate_cycles(system,sec,used_nodes,{**used_solver,'capture_points':tuple(stable)},reverse_capture_points=tuple(reverse))
    cycles=scan['cycles']
    bounds=[max(q['Cmin'] for q in cycles),min(q['Cmax'] for q in cycles)] if cycles else None
    return dict(I=ip3,anchor_C=sec.C,cycles=cycles,coverage_C_interval=bounds,
                nodes_used=np.asarray(used_nodes).tolist(),solver_used=used_solver,
                records=scan['records'],inverse_records=scan['inverse_records'],
                capture_points=[y.tolist() for y in stable],reverse_capture_points=[y.tolist() for y in reverse],
                regular_edges=scan['regular_edges'],suspect_intervals=scan['suspect_intervals'],
                resolved_near_zero_intervals=scan['resolved_near_zero_intervals'])


def prepare_source_inventory(run_dir,config,allow_settings_change=False):
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    p=json.loads((run_dir/'run.json').read_text())['parameters'];cfg=config['cycle_fit']
    path=run_dir/'reference'/'source_cycle_inventory.json'
    settings_signature=hashlib.sha256(json.dumps(dict(algorithm='v5_capture_inventory_2',p=p,solver=settings(config),
        nodes=cfg['profile_nodes'],base_u_nodes=cfg['u_nodes']),sort_keys=True).encode()).hexdigest()
    signature=hashlib.sha256(json.dumps(dict(settings=settings_signature,I=cfg['I_nodes']),sort_keys=True).encode()).hexdigest()
    cached={}
    if path.exists():
        data=json.loads(path.read_text())
        if data['signature']==signature:return data
        if data.get('settings_signature')!=settings_signature:
            if not allow_settings_change:raise ValueError('source cycle inventory belongs to different settings')
        else:cached={r['I']:r for r in data['rows']}
        archive=run_dir/'reference'/'inventory_history';archive.mkdir(exist_ok=True)
        (archive/f"{data['signature']}.json").write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    nodes=nested_nodes(cfg['u_nodes'],cfg['profile_nodes']).tolist()
    workers=cfg.get('reference_workers',1)
    task=partial(_source_profile,p=p,nodes=nodes,solver=settings(config))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        missing=[i for i in cfg['I_nodes'] if i not in cached]
        for row in pool.map(task,missing):
            cached[row['I']]=row
            print(f"Source cycle inventory I={row['I']:.6f}: {len(row['cycles'])} observed cycles",flush=True)
    data=dict(signature=signature,settings_signature=settings_signature,rows=[cached[i] for i in cfg['I_nodes']],
        scope='training source data; finite-grid roots, not a completeness proof')
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    return data


def calibrate_source_epsilon(run_dir,config):
    """A shared final epsilon must resolve the observed LR training counts."""
    cfg=config['cycle_fit'];schedule=config['cycle_smoothing']['epsilon_schedule']
    requested=schedule[-1];history=[];resolved=False
    for grid_round in range(2):
        inventory=prepare_source_inventory(run_dir,config,allow_settings_change=grid_round>0)
        epsilon=requested
        for attempt in range(6):
            errors=[]
            for row in inventory['rows']:
                count,_,edges=smooth_counts(row['records'],epsilon,config['cycle_smoothing']['xi'])
                half,_,_=smooth_counts(row['records'],epsilon/2,config['cycle_smoothing']['xi'])
                observed=np.array([sum(q['kind']=='stable' for q in row['cycles']),sum(q['kind']=='unstable' for q in row['cycles'])])
                error=max(float(np.max(abs(count-observed))),float(np.max(abs(count-half))))
                errors.append(dict(I=row['I'],error=error,regular_edges=edges,unresolved=bool(row['suspect_intervals']),
                    counts=count.tolist(),half_epsilon_counts=half.tolist(),observed_counts=observed.tolist(),
                    suspect_intervals=row['suspect_intervals'],nodes_used=len(row['records'])))
            resolved=all(q['error']<=cfg['count_tolerance'] and q['regular_edges']>=2 and not q['unresolved'] for q in errors)
            profile_unresolved=any(q['unresolved'] or q['regular_edges']<2 for q in errors)
            history.append(dict(epsilon=epsilon,profile_nodes=cfg['profile_nodes'],resolved=resolved,
                profile_unresolved=profile_unresolved,errors=errors))
            if resolved:break
            # Epsilon cannot repair an unresolved orbit/domain scan. Refine the
            # profile rather than repeatedly reducing unrelated smoothing.
            if profile_unresolved:break
            if attempt<5:epsilon/=3
        if resolved:break
        if grid_round==0:cfg['profile_nodes']=2*cfg['profile_nodes']-1
    if resolved and epsilon<schedule[-1]:schedule.append(epsilon)
    report=dict(requested_final_epsilon=requested,actual_final_epsilon=epsilon,resolved=resolved,history=history)
    (run_dir/'epsilon_calibration.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not resolved:
        bad=[q for q in errors if q['error']>cfg['count_tolerance'] or q['unresolved'] or q['regular_edges']<2]
        detail='; '.join(f"I={q['I']:.12g}: " +
            (','.join(sorted(set(s['message'] for s in q['suspect_intervals']))) if q['unresolved']
             else f"regular_edges={q['regular_edges']}, counter_error={q['error']:.6g}") for q in bad)
        raise ValueError(f'LR training calibration unresolved: {detail}; inspect epsilon_calibration.json')
    print(f'Shared final epsilon calibrated: {epsilon:g}; source profile nodes={cfg["profile_nodes"]}',flush=True)
    return report


def _warm_profile(task):
    model,p,ip3,nodes,solver=task;x=vector(model);cmax=p['C_0']/(1+p['c_1'])
    system=PWASystem(x,model['physical_thresholds_C'],ip3,model['transform']['C_ref'],cmax)
    try:
        sec,info=anchor_section(system,LRSystem(p,ip3),reference_section(ip3,p).C)
        profile=candidate_profiles(system,sec,nodes,solver)
        return dict(I=ip3,counts=[sum(q['kind']=='stable' for q in profile['cycles']),
                                sum(q['kind']=='unstable' for q in profile['cycles'])],
                    section=info,cycles=profile['cycles'],
                    unresolved=bool(profile['suspect_intervals']) or not(profile['regular_edges'] or profile['no_equilibrium_certificate']))
    except ValueError as exc:return dict(I=ip3,counts=None,unresolved=True,error=str(exc))


def refine_training_levels(run_dir,config,warm_model=None,rounds=1):
    """Adapt I to observed source/incumbent cycle transitions before fitting.

    All starts subsequently share this frozen grid. Independent validation is
    not consulted. Source inventories are extended with only the new I values.
    """
    from concurrent.futures import ProcessPoolExecutor
    cfg=config['cycle_fit'];p=json.loads((run_dir/'run.json').read_text())['parameters']
    nodes=nested_nodes(cfg['u_nodes'],cfg['profile_nodes']).tolist();warm_cache={};history=[]
    probe=list(cfg['coarse_I_nodes'])
    for iteration in range(rounds):
        inventory=prepare_source_inventory(run_dir,config);source={r['I']:r for r in inventory['rows']}
        if warm_model is not None:
            tasks=[(warm_model,p,i,nodes,settings(config)) for i in probe if i not in warm_cache]
            with ProcessPoolExecutor(max_workers=cfg.get('reference_workers',1)) as pool:
                for row in pool.map(_warm_profile,tasks):
                    warm_cache[row['I']]=row
                    print(f"Training-grid probe I={row['I']:.6f}: incumbent counts={row['counts']} unresolved={row['unresolved']}",flush=True)
        add=[];brackets=[]
        for a,b in zip(probe[:-1],probe[1:]):
            signature=lambda i:tuple(sum(q['kind']==kind for q in source[i]['cycles']) for kind in ['stable','unstable'])
            changes=signature(a)!=signature(b)
            if warm_model is not None:
                left,right=warm_cache[a],warm_cache[b]
                changes |= left['counts']!=right['counts'] or left['unresolved'] or right['unresolved']
            if changes:
                add.extend([a+.25*(b-a),a+.75*(b-a)])
                brackets.append([a,b])
        updated=sorted(set(cfg['I_nodes']+np.round(add,12).tolist()))
        history.append(dict(round=iteration,transition_brackets=brackets,before=cfg['I_nodes'],after=updated,
                            incumbent_profiles=[warm_cache[i] for i in probe] if warm_model is not None else None))
        cfg['I_nodes']=updated
        # The next probe can reuse all existing profiles and bisect observed brackets.
        probe=sorted(set(probe+np.round(add,12).tolist()))
        prepare_source_inventory(run_dir,config)
    (run_dir/'training_grid_history.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
    return history


def candidate_profiles(system,sec,nodes,solver):
    profile_solver={**solver,'rtol':3e-10,'atol':3e-12,'max_step':.25}
    roots=model_roots(system.theta,system.thresholds,system.ip3,system.c_ref,system.cmin,system.cmax)
    physical=[q for q in roots if 0<q[0]<system.cmax and 0<q[1]<1]
    sections=sorted(set([sec.C]+[q[0] for q in physical]));cycles=[];suspects=[];regular=0
    stable=[];reverse=[]
    for c,h in physical:
        y=np.array([system.coordinate(c),h]);eig=np.linalg.eigvals(system.jac(y)).real
        if max(eig)<-1e-8:stable.append(y)
        if min(eig)>1e-8:reverse.append(y)
    for c in sections:
        h0=max(0.,system.nullcline(c))
        if h0>=1:continue
        scan=isolate_cycles(system,Section(c,h0),nodes,{**profile_solver,
            'witness_sections':sections,'capture_points':tuple(stable)},reverse_capture_points=tuple(reverse))
        regular+=scan['regular_edges'];suspects.extend(scan['suspect_intervals'])
        for q in scan['cycles']:
            if not any(same_orbit(q,other) for other in cycles):cycles.append(q)
    # Index theory excludes a closed orbit in the physical rectangle if it
    # contains no equilibrium. This certificate is separate from failed maps.
    return dict(cycles=cycles,suspect_intervals=suspects,regular_edges=regular,
                no_equilibrium_certificate=not physical,physical_equilibria=len(physical),
                solver_used=profile_solver,u_nodes_used=np.asarray(nodes).tolist(),sections_C=sections)


def audit_candidate(run_dir,config,model,*,profiles=True,label=None,checkpoint_dir=None,progress=None):
    """Value-only common-grid score; unknown returns/coverage are never zero."""
    from fit_stage import MixedObjective
    geometry=MixedObjective(run_dir,config)
    if len(model['physical_thresholds_C'])!=len(geometry.thresholds) or not np.allclose(model['physical_thresholds_C'],geometry.thresholds,rtol=0,atol=1e-12):
        raise ValueError('audit candidate must use the run threshold partition')
    x=vector(model);parts=geometry.parts(x);parts.pop('precycle_total')
    cfg=config['cycle_fit'];epsilon=config['cycle_smoothing']['epsilon_schedule'][-1]
    xi=config['cycle_smoothing']['xi'];tau=config['cycle_smoothing']['softplus_tau']
    nodes=nested_nodes(cfg['u_nodes'],cfg['selection_nodes']);fine=nested_nodes(nodes,2*len(nodes)-1)
    p=geometry.run['parameters'];cmax=p['C_0']/(1+p['c_1']);rows=[];cyc=extra=0.
    inventory=json.loads((run_dir/'reference'/'source_cycle_inventory.json').read_text())
    source_profiles={r['I']:r for r in inventory['rows']}
    if progress is not None:progress.start_work('audit_I_points',len(cfg['I_nodes']))
    for ip3 in cfg['I_nodes']:
        if progress is not None:progress.advance(len(rows),I=ip3,activity='returns_and_cycle_profiles')
        system=PWASystem(x,geometry.thresholds,ip3,geometry.c_ref,cmax);source=LRSystem(p,ip3)
        try:
            sec,info=anchor_section(system,source,reference_section(ip3,p).C)
            tables={key:return_table(s,sec,fine,**settings(config)) for key,s in [('PWA',system),('LR',source)]}
            # Nested grids use exactly identical records at the common nodes.
            indices=[int(np.argmin(abs(fine-u))) for u in nodes]
            counts={};resolution=True;domains=True
            for key,table in tables.items():
                coarse=[table[j] for j in indices]
                a,_,ea=smooth_counts(coarse,epsilon,xi);b,_,eb=smooth_counts(table,epsilon,xi)
                c,_,_=smooth_counts(table,epsilon/2,xi)
                valid=min(ea,eb)>=2;domains &= valid
                convergence=max(float(np.max(abs(a-b))),float(np.max(abs(b-c))))
                integer_error=float(np.max(abs(c-np.round(c))))
                resolution &= valid and max(convergence,integer_error)<=cfg['count_tolerance']
                counts[key]=dict(common=a.tolist() if valid else None,refined=b.tolist() if valid else None,
                    half_epsilon=c.tolist() if valid else None,regular_edges=ea,
                    convergence_error=convergence,integer_error=integer_error)
            profile=candidate_profiles(system,sec,nodes,settings(config)) if profiles else None
            source_profile=source_profiles[ip3]
            src_covered=all(q['Cmin']+1e-7<sec.C<q['Cmax']-1e-7 for q in source_profile['cycles'])
            cand_covered=profile is not None and all(q['Cmin']+1e-7<sec.C<q['Cmax']-1e-7 for q in profile['cycles'])
            coverage=src_covered and cand_covered and not source_profile['suspect_intervals'] and source_profile['regular_edges']>0
            if profile is not None:
                observed=np.array([sum(q['kind']=='stable' for q in profile['cycles']),sum(q['kind']=='unstable' for q in profile['cycles'])])
                if counts['PWA']['half_epsilon'] is not None:
                    resolution &= np.max(abs(observed-counts['PWA']['half_epsilon']))<=cfg['count_tolerance']
                coverage &= not profile['suspect_intervals'] and (profile['regular_edges']>0 or profile['no_equilibrium_certificate'])
            if counts['LR']['half_epsilon'] is not None:
                observed=np.array([sum(q['kind']=='stable' for q in source_profile['cycles']),sum(q['kind']=='unstable' for q in source_profile['cycles'])])
                resolution &= np.max(abs(observed-counts['LR']['half_epsilon']))<=cfg['count_tolerance']
            if domains:
                a=np.array(counts['PWA']['common']);b=np.array(counts['LR']['common'])
                cyc+=config['loss_weights']['cycles']/len(cfg['I_nodes'])*float(np.sum((a-b)**2))
                sp=lambda z:tau*np.logaddexp(0,z/tau)
                extra+=config['loss_weights']['extra_cycles']*float((sp(a.sum()-2)-sp(b.sum()-2))**2)
            else:cyc+=2e6;extra+=1e6
            rows.append(dict(I=ip3,section=info,counts=counts,domain_admissible=bool(domains),
                resolution_admissible=bool(resolution),coverage_admissible=bool(coverage),candidate_profile=profile))
        except ValueError as exc:
            cyc+=2e6;extra+=1e6
            rows.append(dict(I=ip3,error=str(exc),domain_admissible=False,resolution_admissible=False,coverage_admissible=False))
        if label is not None:
            row=rows[-1]
            print(f"{label}: training audit {len(rows)}/{len(cfg['I_nodes'])}, I={ip3:.6f}, coverage={row['coverage_admissible']} resolution={row['resolution_admissible']}",flush=True)
        if progress is not None:progress.advance(len(rows),I=ip3)
        if checkpoint_dir is not None:
            target=checkpoint_dir/'training_audit_progress.json';temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(completed=len(rows),total=len(cfg['I_nodes']),rows=rows),indent=2,allow_nan=False)+'\n')
            temporary.replace(target)
    parts.update(cycles_weighted=cyc,extra_cycles_weighted=extra,return_domain_barrier=0.,
        full_total=parts['nullclines']+parts['realized_spectra_weighted']+parts['equilibria_weighted']+cyc+extra,
        epsilon=epsilon,return_domain_admissible=all(q['domain_admissible'] for q in rows),
        cycle_resolution_admissible=all(q['resolution_admissible'] for q in rows),
        cycle_coverage_admissible=all(q['coverage_admissible'] for q in rows))
    return dict(parts=parts,rows=rows,common_nodes=len(nodes),refined_nodes=len(fine),profiles_checked=profiles,
                counter_solver=settings(config),count_tolerance=cfg['count_tolerance'],
                candidate_selected_using_independent_validation=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((args.run_dir/'protocol.json').read_text())
    if config.get('cycle_fit',{}).get('section_policy')!='moving_equilibrium':parser.error('this command audits v5 moving-section experiments')
    path=args.candidate or args.run_dir/'optimizer'/'selected.json'
    if not path.exists() and args.candidate is None:path=args.run_dir/'optimizer'/'best_so_far.json'
    loaded=json.loads(path.read_text());model=loaded.get('candidate',loaded)
    if args.out.exists():parser.error('--out must be a fresh directory')
    args.out.mkdir(parents=True)
    report=audit_candidate(args.run_dir,config,model,label='audit',checkpoint_dir=args.out)
    (args.out/'training_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['parts'],indent=2),flush=True)
