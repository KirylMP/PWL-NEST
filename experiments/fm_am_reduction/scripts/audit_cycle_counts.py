#!/usr/bin/env python3
"""Compare training-section smoothed counts with independent closed orbits.

Read-only with respect to fitted experiments. Defaults reconstruct the v3 FM
configuration; pass the experiment's protocol.json to reproduce custom grids.
"""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
from analyze_candidate import vector
from fit_stage import model_roots
from full_cycle_fit import DEFAULTS,reference_section,settings
from plan import PACKAGE,read_json
from prepare_reference_data import roots_for_ip3
from poincare import LRSystem,PWASystem,return_table,smooth_counts
from validate_structure import _validate_point


def counter_report(table,reference,config):
    xi=config['cycle_smoothing']['xi'];tau=config['cycle_smoothing']['softplus_tau']
    result=[]
    for epsilon in config['cycle_smoothing']['epsilon_schedule']:
        count,_,edges=smooth_counts(table,epsilon,xi)
        truth,_,refedges=smooth_counts(reference,epsilon,xi)
        required=max(2,int(np.ceil(config['cycle_fit']['min_regular_edge_fraction']*refedges)))
        valid=refedges>=2 and edges>=required
        sp=lambda v:tau*np.logaddexp(0,v/tau)
        result.append(dict(epsilon=epsilon,counts=count.tolist() if valid else None,
            partial_variation=count.tolist(),reference_counts=truth.tolist() if refedges>=2 else None,
            regular_edges=edges,reference_edges=refedges,required_edges=required,admissible=valid,
            cycle_squared_error=float(np.sum((count-truth)**2)) if valid else None,
            extra_squared_error=float((sp(count.sum()-2)-sp(truth.sum()-2))**2) if valid else None))
    return dict(statuses=dict(Counter(q['status'] for q in table)),by_epsilon=result)


def aggregate_reports(rows,config):
    reports=[]
    for label in sorted(set(q['label'] for q in rows)):
        selected=[q for q in rows if q['label']==label]
        for scope,levels in [('original_training_I',config['cycle_fit']['I_nodes']),
                             ('audit_I_grid',sorted(q['I'] for q in selected))]:
            group=[q for q in selected if q['I'] in levels]
            missing=[i for i in levels if i not in [q['I'] for q in group]]
            for grid in ['sparse','dense']:
                for index,epsilon in enumerate(config['cycle_smoothing']['epsilon_schedule']):
                    parts=[q[grid]['by_epsilon'][index] for q in group]
                    valid=not missing and bool(parts) and all(q['admissible'] for q in parts)
                    reports.append(dict(label=label,scope=scope,grid=grid,epsilon=epsilon,I_nodes=levels,
                        missing_I_nodes=missing,all_domains_admissible=valid,
                        cycles_weighted=config['loss_weights']['cycles']*float(np.mean([q['cycle_squared_error'] for q in parts])) if valid else None,
                        extra_cycles_weighted=config['loss_weights']['extra_cycles']*sum(q['extra_squared_error'] for q in parts) if valid else None))
    return reports


def audit_point(task):
    label,model,p,ip3,config,nodes,structural,structure_nodes,max_time=task
    sec=reference_section(ip3,p);cmax=p['C_0']/(1+p['c_1'])
    systems={'LR':LRSystem(p,ip3),'PWA':PWASystem(vector(model),model['physical_thresholds_C'],ip3,model['transform']['C_ref'],cmax)}
    sparse=config['cycle_fit']['u_nodes'];split=max(3,nodes//3)
    dense=np.unique(np.r_[np.geomspace(.0002,.1,split),np.linspace(.1,.995,nodes-split+1),sparse])
    eqs={'LR':[(q['C'],q['h']) for q in roots_for_ip3(ip3,p,cmax,5000)],
         'PWA':model_roots(vector(model),model['physical_thresholds_C'],ip3,model['transform']['C_ref'],1e-8,cmax-1e-8)}
    tables={}
    for key,system in systems.items():
        capture=tuple(np.array([system.coordinate(c),h]) for c,h in eqs[key]
            if 0<c<cmax and 0<h<1 and np.max(np.linalg.eigvals(system.jac([system.coordinate(c),h])).real)<-1e-8)
        tables[key]=return_table(system,sec,dense,capture_points=capture,**settings(config))
    sparse_tables={key:[table[int(np.argmin(abs(dense-u)))] for u in sparse] for key,table in tables.items()}
    report=dict(label=label,I=ip3,in_training_I_grid=ip3 in config['cycle_fit']['I_nodes'],
        section=dict(C=sec.C,h0=sec.h0,orientation=sec.orientation),
        sparse=counter_report(sparse_tables['PWA'],sparse_tables['LR'],config),
        dense=counter_report(tables['PWA'],tables['LR'],config),
        u_nodes_dense=dense.tolist(),records=tables)
    if structural:
        k=max(4,structure_nodes//3)
        unodes=np.unique(np.r_[np.geomspace(.0002,.08,k),np.linspace(.08,.995,structure_nodes-k+1)])
        solver=dict(max_time=max_time,rtol=3e-10,atol=3e-12,max_step=.25,max_evaluations=200000)
        scan=_validate_point((ip3,model,p,unodes.tolist(),solver,{}))
        report['structure']=scan
        report['coverage']={}
        for key in systems:
            cycles=scan[key]['cycles'];csec=str(float(sec.C))
            report['coverage'][key]=dict(observed_cycles=len(cycles),
                stable=sum(q['kind']=='stable' for q in cycles),unstable=sum(q['kind']=='unstable' for q in cycles),
                cycles_crossing_training_section=sum(bool(q.get('orbit_crossings',{}).get(csec)) for q in cycles),
                unresolved_intervals=sum(len(s['suspect_intervals']) for s in scan[key]['sections']),
                status=scan[key]['status'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,nargs='+',required=True)
    parser.add_argument('--config',type=Path,help='original run protocol.json (defaults otherwise)')
    parser.add_argument('--I',type=float,nargs='+',default=[.55,.65,.75,.78,.8,.8024691358,1.,87/81,1.1,1.14])
    parser.add_argument('--structure-I',type=float,nargs='*',default=[.78,.8024691358,87/81])
    parser.add_argument('--nodes',type=int,default=65)
    parser.add_argument('--structure-nodes',type=int,default=65)
    parser.add_argument('--max-time',type=float,default=600.)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if min(args.nodes,args.structure_nodes)<5 or args.workers<1 or args.max_time<=0:
        parser.error('positive budgets and at least five nodes required')
    levels=sorted(set(args.I+args.structure_I))
    if any(not 0<i<=2 for i in levels):parser.error('I must lie within (0,2]')
    if args.out.exists():parser.error('--out already exists; choose a fresh directory')
    config=read_json(args.config or PACKAGE/'config'/'protocol.json')
    config['cycle_fit']={**DEFAULTS,**config.get('cycle_fit',{})}
    cases=read_json(PACKAGE/'config'/'cases.json');tasks=[];provenance=[]
    for index,path in enumerate(args.candidate):
        loaded=read_json(path);model=loaded.get('candidate',loaded)
        if model.get('case')!='FM':parser.error('this audit currently supports FM candidates')
        p={**cases['shared_parameters'],'K_ER':model['K_ER']}
        label=f'{index:02d}_{path.parent.name}_{path.stem}'
        provenance.append(dict(label=label,path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            saved_loss=loaded.get('training_loss_parts'),parameters=p))
        tasks.extend((label,model,p,float(i),config,args.nodes,i in args.structure_I,args.structure_nodes,args.max_time) for i in levels)
    args.out.mkdir(parents=True)
    (args.out/'logs').mkdir()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(args.out/'logs'/'audit.log')])
    log=logging.getLogger(__name__)
    hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (PACKAGE/'scripts').glob('*.py')}
    (args.out/'manifest.json').write_text(json.dumps(dict(inputs=provenance,config=config,
        config_source=str(args.config) if args.config else 'assumed v3 defaults; original run config not supplied',
        I_nodes=levels,structural_I=args.structure_I,workers=args.workers,code_sha256=hashes,
        dense_nodes_requested=args.nodes,structure_nodes_requested=args.structure_nodes,structural_max_time=args.max_time,
        fitting_performed=False,candidate_selection_performed=False),indent=2)+'\n')
    summaries=[];failures=[]
    log.info('Starting %d audit tasks; results: %s',len(tasks),args.out)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(audit_point,task):(task[0],task[3]) for task in tasks}
        for future in as_completed(pending):
            try:row=future.result()
            except Exception:
                label,ip3=pending[future]
                failures.append(dict(label=label,I=ip3))
                log.exception('Audit failed: %s I=%s',label,ip3)
                continue
            filename=f"{row['label']}_I_{row['I']:.10f}.json"
            (args.out/filename).write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            short={k:v for k,v in row.items() if k not in ['structure','records','u_nodes_dense']}
            summaries.append(short)
            summaries.sort(key=lambda q:(q['label'],q['I']))
            (args.out/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')
            (args.out/'aggregate.json').write_text(json.dumps(aggregate_reports(summaries,config),indent=2,allow_nan=False)+'\n')
            log.info('%s I=%.9f counts=%s closed_orbits=%s',row['label'],row['I'],
                     row['dense']['by_epsilon'][-1]['counts'],row.get('coverage',{}))
    (args.out/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    if failures:raise RuntimeError(f'{len(failures)} audit tasks failed; inspect logs/audit.log')
    from plot_cycle_audit import plot_audit
    plot_audit(args.out)
    log.info('Audit finished: %s',args.out)


if __name__=='__main__':main()
