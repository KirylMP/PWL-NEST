"""Poincare root-count penalty for extra stable and unstable FM cycles.

The reference is the archived finite-grid LR cycle inventory. Candidate roots
are enumerated on equilibrium sections in both time directions. This is a
discrete, nonsmooth loss; incomplete return maps can miss cycles.
"""
import json
import os
import numpy as np
import fit_four as f


def reference_grid():
    rows=json.loads((f.P/'LR_cycles_reference.json').read_text())['rows']
    explicit=os.environ.get('FM_CYCLE_COUNT_I')
    if explicit:
        desired=[float(v.strip()) for v in explicit.split(',') if v.strip()]
        if not desired or not all(np.isfinite(v) for v in desired):raise ValueError('FM_CYCLE_COUNT_I must contain finite I values')
        for I in desired:
            if min(abs(r['I']-I) for r in rows)>1e-8:raise ValueError(f'FM_CYCLE_COUNT_I={I} is not an LR reference point')
    else:
        n=int(os.environ.get('FM_CYCLE_COUNT_POINTS','17'))
        if n<2:raise ValueError('FM_CYCLE_COUNT_POINTS must be at least 2')
        desired=np.r_[np.linspace(.45,1.25,n),[.52,.53,1.12,1.14,1.5,2.]]
    selected={min(rows,key=lambda r:abs(r['I']-float(I)))['I'] for I in desired}
    result=[]
    for I in sorted(selected):
        row=min(rows,key=lambda r:abs(r['I']-I))
        result.append(dict(I=float(row['I']),
            stable=sum(bool(q['stable']) for q in row['cycles']),
            unstable=sum(not bool(q['stable']) for q in row['cycles']),
            diagnostics=row.get('diagnostics',{})))
    return result


def evaluate(x,reference,seeds=32,progress=None):
    if seeds<4:raise ValueError('FM_CYCLE_COUNT_SEEDS must be at least 4')
    m=f.unpack(x);bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
    report=[];losses=[]
    for row in reference:
        I=row['I'];eq=f.equilibrium(I,m)
        found,diag=f.cycles(I,eq,m['co'],m['padded'],bounds,n=seeds,tol=1e-9)
        stable=sum(q['stable'] for q in found);unstable=len(found)-stable
        ds=max(0,stable-row['stable']);du=max(0,unstable-row['unstable'])
        loss=float(ds*ds+du*du);losses.append(loss)
        report.append(dict(I=I,reference_stable=row['stable'],reference_unstable=row['unstable'],
            stable=stable,unstable=unstable,extra_stable=ds,extra_unstable=du,
            loss=loss,equilibrium_sections=len(eq),diagnostics=diag,
            reference_diagnostics=row['diagnostics']))
        if progress is not None:progress(len(report),len(reference),report[-1])
    return float(np.mean(losses)),report
