"""Fit a softplus-coordinate field and construct one neutral-fold start per model.

The previous physical candidates supply the fixed equilibrium anchors, fold input,
threshold levels and initial orbit times. Their coordinate-dependent coefficients
are deliberately not copied into softplus coordinates.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
from fit_transformed_coordinates import (ROOT, KSETS, SOFTPLUS_SCALE,
    make_reference, fit_model, calc)
from optimize_fixed_fold_cycles import Field


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases', nargs='+', default=['AM','AFM','FM'], choices=KSETS)
    p.add_argument('--scale', type=float, default=SOFTPLUS_SCALE, help='C* in C=C* softplus(u), in uM')
    p.add_argument('--fold-nfev', type=int, default=500)
    p.add_argument('--output', type=Path, default=ROOT/'results/coordinate_experiment')
    a=p.parse_args()
    if a.scale <= 0: p.error('--scale must be positive')
    for case in a.cases:
        print('Preparing',case,'softplus scale',a.scale,'uM',flush=True)
        base=json.loads((ROOT/'paper/data/random_selected'/case/'candidate.json').read_text())
        report=json.loads((ROOT/'paper/data/random_selected'/case/'report.json').read_text())
        refs=make_reference(case,base,report)
        model=fit_model('softplus',case,base,refs,scale=a.scale)
        rows=calc(model,refs)
        coord=a.output/f'{case}_softplus.json';coord.parent.mkdir(parents=True,exist_ok=True)
        coord.write_text(json.dumps({'model':model,'results':rows},indent=2,allow_nan=False)+'\n')
        attempt=coord.with_name(f'{case}_softplus_fold_attempt.json')
        cmd=[sys.executable,'-u',str(ROOT/'scripts/constrain_transformed_fold.py'),
             '--case',case,'--kind','softplus','--nfev',str(a.fold_nfev)]
        subprocess.run(cmd,cwd=ROOT,check=True)
        if not attempt.exists(): raise FileNotFoundError(attempt)
        fit=json.loads(attempt.read_text())
        if not all(fit.get('sampled_residence',[])) or abs(fit.get('matching_max',1.))>1e-7 or abs(fit.get('logM',1.))>1e-7:
            raise ValueError(f'{case}: neutral fold attempt failed closure/residence checks')
        coef=np.asarray(fit['candidate_coefficients'],float)  # 2 x [1,u,q,I,hinge1,hinge2]
        rate_a=float(coef[0,2]);rate_b=float(-coef[1,2])
        if rate_a<=0 or rate_b<=0: raise ValueError(f'{case}: invalid coupling rates')
        n=(-coef[0,[0,1,3,4,5]]/rate_a).tolist()
        h=(coef[1,[0,1,3,4,5]]/rate_b).tolist()
        candidate={'case':case,'kind':'softplus','seed':0,'I_fold':fit['input_I'],
                   'n':n,'h':h,'a':rate_a,'b':rate_b,
                   'cuts':model['cuts_u'],'times':fit['times'],
                   'matching_max':fit['matching_max'],'logM':fit['logM'],
                   'source':'softplus constrained field fit; fixed equilibrium/fold geometry from previous physical candidate',
                   'coordinate_scale_uM':a.scale}
        F=Field(case,'softplus',candidate)
        residual,lm=F.route(F.p0,F.Istar,np.log(candidate['times']),'both')
        candidate['matching_max']=float(max(abs(residual)));candidate['logM']=float(lm)
        if candidate['matching_max']>1e-7 or abs(lm)>1e-7: raise ValueError(f'{case}: converted parameterization no longer closes fold')
        out=ROOT/'results/fold_anchors'/case/'softplus'/'seed_0.json';out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n')
        print('WROTE',out,'matching',candidate['matching_max'],'logM',candidate['logM'],flush=True)

if __name__=='__main__': main()
