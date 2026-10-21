"""Reproducible first strict-3D-PWA identification experiment.
Run with OPENBLAS_NUM_THREADS=1 for predictable sparse least-squares performance.
"""
import argparse
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/workspace/PWL-NEST/.cache/matplotlib')
import numpy as np
from scipy.signal import find_peaks
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import fit_field,lr_field
from pwl_nest.protocols import simulate

OUT = ROOT/'results/strict_pwa'
MODELS = ROOT/'models/strict_pwa'


def stats(t,y):
    mask = t >= t[-1]-400
    x,c = t[mask],y[mask,0]
    peaks,_ = find_peaks(c,prominence=.001)
    intervals = np.diff(x[peaks])
    midpoint = len(c)//2
    early,late = np.ptp(c[:midpoint]),np.ptp(c[midpoint:])
    stable_envelope = abs(late-early) < max(.001,.03*max(early,late))
    period = float(np.median(intervals)) if len(intervals)>=3 and stable_envelope else None
    return dict(Cmin=float(c.min()),Cmax=float(c.max()),amplitude=float(np.ptp(c)),
                period=period,envelope_converged=bool(stable_envelope),
                observed_status='cycling' if period is not None else ('rest' if np.ptp(c)<.001 else 'unresolved'))


def compare(protocol,ref,trial):
    t,y,meta = trial
    _,truth,_ = ref
    rmse = np.sqrt(np.mean((y-truth)**2,axis=0))
    out = dict(name=protocol['name'],rmse_C_h_I=rmse.tolist(),max_abs_C=float(np.max(abs(y[:,0]-truth[:,0]))),
               C_min=float(y[:,0].min()),h_range=[float(y[:,1].min()),float(y[:,1].max())],
               IP3_error=meta['ip3_max_abs_error'])
    if protocol['name'].startswith('fixed'):
        a,b = stats(t,truth),stats(t,y)
        out.update(reference=a,pwa=b)
        out['extrema_abs_error'] = max(abs(a['Cmin']-b['Cmin']),abs(a['Cmax']-b['Cmax']))
        out['period_abs_error'] = abs(a['period']-b['period']) if a['period'] is not None and b['period'] is not None else None
        # Working training score, not biological acceptance.
        score = (out['extrema_abs_error']/.02)**2
        score += (out['period_abs_error']/1.)**2 if out['period_abs_error'] is not None else (0. if a['observed_status']==b['observed_status']=='rest' else 100.)
    else:
        score = (rmse[0]/.02)**2
    out['training_score_term'] = float(score)
    return out


def protocols(case,heldout=False):
    c = .6 if case=='AM' else 1.3
    fixed = ([.37,.47,.59,.638,.65] if case=='AM' else [.53,.65,.9,1.12,1.13]) if heldout else ([.4,.5,.6,.7] if case=='AM' else [.55,.7,1.,1.2])
    out = [dict(name=f'fixed_{ip3}',initial=[c,.8,ip3],duration=1600.,tau=20.,baseline=ip3) for ip3 in fixed]
    start,base = (.65,.3) if case=='AM' else (1.15,.45)
    if not heldout:
        out += [dict(name=f'decay_tau_{tau}',initial=[.08,.8,start],duration=180.,tau=tau,baseline=base) for tau in [20.,100.]]
    else:
        out += [dict(name=f'decay_tau_{tau}',initial=[.08,.8,start],duration=180.,tau=tau,baseline=base) for tau in [2.,50.]]
        amplitude = .16 if case=='AM' else .3
        out += [dict(name='periodic_jumps_then_stop',initial=[.08,.8,base],duration=200.,tau=20.,baseline=base,
                     jumps=[(t,amplitude) for t in [10.,30.,50.,70.,90.]])]
        rng = np.random.default_rng(48317)
        times = np.cumsum(rng.uniform(9,19,7))
        out += [dict(name='irregular_jumps_then_stop',initial=[.08,.8,base],duration=220.,tau=12.,baseline=base,
                     jumps=[(float(t),amplitude) for t in times])]
    return out


def run(case,quick=False):
    OUT.mkdir(parents=True,exist_ok=True)
    MODELS.mkdir(parents=True,exist_ok=True)
    training = protocols(case)
    if quick:
        training = [training[1],training[-2]]
    references = {}
    samples = []
    def solve(p,model=None):
        return simulate(case,**{k:v for k,v in p.items() if k!='name'},model=model)
    for p in training:
        references[p['name']] = solve(p)
        t,y,_ = references[p['name']]
        samples.append(y[::max(1,len(y)//1000)])
    samples = np.vstack(samples)
    records, winner, best_score = [],None,np.inf
    for refine in ([False] if quick else [False,True]):
        print(f'{case}: fitting {"refined" if refine else "coarse"}',flush=True)
        fitted,interpolated = fit_field(case,refine=refine,trajectory_points=samples)
        for candidate in [interpolated,fitted]:
            label = ('refined' if refine else 'coarse')+('_fit' if candidate is fitted else '_interpolation')
            candidate.save(MODELS/f'{case}_{label}.npz')
            results = []
            for p in training:
                print(f'{case} {label}: {p["name"]}',flush=True)
                results.append(compare(p,references[p['name']],solve(p,candidate)))
            score = float(np.mean([r['training_score_term'] for r in results]))
            rng = np.random.default_rng(319)
            points = rng.uniform([0,0,0],[2/1.185,1,1.3],(5000,3))
            field_rmse = np.sqrt(np.mean((candidate.field(points)-lr_field(points,case))**2,axis=0))
            records.append(dict(label=label,training_score=score,metadata=candidate.metadata,
                                field_holdout_rmse=field_rmse.tolist(),training_results=results))
            print(f'{case} {label}: training score={score:.5g}, regions={candidate.region_count}',flush=True)
            if score < best_score:
                winner,best_score,selected_label = candidate,score,label
        # Keep incremental raw evidence even if a later operation fails.
        (OUT/f'{case}_training.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    winner.metadata['selected_by'] = 'lowest predeclared training score; holdouts not used for selection'
    winner.metadata['selected_label'] = selected_label
    winner.save(MODELS/f'{case}.npz')
    tests = protocols(case,heldout=True)
    if quick:
        tests = [tests[1],tests[-1]]
    holdouts = []
    for p in tests:
        print(f'{case} selected {selected_label}: holdout {p["name"]}',flush=True)
        ref,trial = solve(p),solve(p,winner)
        result = compare(p,ref,trial)
        result.pop('training_score_term')
        holdouts.append(result)
        if not p['name'].startswith('fixed'):
            np.savez_compressed(OUT/f'{case}_{p["name"]}.npz',t=ref[0],lr=ref[1],pwa=trial[1])
    report = dict(case=case,seed=20261020,selected=selected_label,training_score=best_score,
                  scope='pilot identification, not full blueprint acceptance',
                  training_protocols=training,holdout_protocols=tests,training=records,holdout=holdouts,
                  unrun=['unstable cycles','continuation and bifurcation boundary certification',
                         'SIC comparison','global event-localization solver','C++ strict 3D solver','NEST benchmark'])
    (OUT/f'{case}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{case}: selected {selected_label}; holdouts saved',flush=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',choices=['AM','FM','both'],default='both')
    parser.add_argument('--quick',action='store_true',help='development smoke only, not the full experiment')
    args = parser.parse_args()
    if args.quick:
        OUT = OUT.with_name('strict_pwa_smoke')
        MODELS = MODELS.with_name('strict_pwa_smoke')
    for case in (['AM','FM'] if args.case=='both' else [args.case]):
        run(case,args.quick)
