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
from optimize_fixed_fold_cycles import Field, flow, ROUTES
from scipy.optimize import root, least_squares


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
        if not np.isfinite(fit.get('matching_max',np.nan)) or not np.isfinite(fit.get('logM',np.nan)):
            raise ValueError(f'{case}: constrained fold attempt returned non-finite residuals: {fit}')
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
        # The least-squares constrained fit is only a nearby initializer. Polish
        # exact matching + log(M)=0 by eliminating one locally independent field
        # coefficient and solving it together with the four flight times.
        p0=F.p0.copy(); logs0=np.log(candidate['times'])
        def fold_res(pp,tt):
            mismatch,logm=F.route(pp,F.Istar,tt,'both')
            return np.r_[mismatch,logm]
        eps=1e-5; pivots=[]
        Jt=np.column_stack([(fold_res(p0,logs0+eps*np.eye(4)[k])-fold_res(p0,logs0-eps*np.eye(4)[k]))/(2*eps) for k in range(4)])
        for k in range(len(p0)):
            dp=np.eye(len(p0))[k]*F.scale[k]*eps
            dv=(fold_res(p0+dp,logs0)-fold_res(p0-dp,logs0))/(2*eps)
            sv=np.linalg.svd(np.column_stack((dv,Jt)),compute_uv=False)
            pivots.append((float(sv[-1]),k))
        initial_res=fold_res(p0,logs0)
        solved=None;attempts=[]
        for rank,k in sorted(pivots,reverse=True):
            z0=np.r_[p0[k],logs0]
            def equations(z):
                pp=p0.copy();pp[k]=z[0]
                try: return fold_res(pp,z[1:])
                except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError): return np.full(5,1e6)
            try:
                sol=root(equations,z0,tol=1e-10,options={'maxfev':3000})
                err=float(np.max(np.abs(equations(sol.x))))
                attempts.append({'pivot':int(k),'rank_score':rank,'root_success':bool(sol.success),'max_residual':err})
                if err<1e-8 and abs((sol.x[0]-p0[k])/F.scale[k])<.5 and np.max(np.abs(sol.x[1:]-logs0))<1.1:
                    solved=(k,sol.x);break
            except Exception as e: attempts.append({'pivot':int(k),'rank_score':rank,'error':repr(e)})
        if solved is None:
            raise ValueError(f'{case}: exact fold polish failed; initial residual={initial_res.tolist()}; pivot attempts={attempts}')
        pivot,z=solved;pp=p0.copy();pp[pivot]=z[0];polished_times=np.exp(z[1:])
        n2,h2,a2,b2=F.coeff(pp)
        candidate.update({'n':n2.tolist(),'h':h2.tolist(),'a':float(a2),'b':float(b2),'times':polished_times.tolist(),'fold_polish_pivot':int(pivot),'fold_polish_attempts':attempts})
        residual,lm,times,mats,starts,speeds=F.route(pp,F.Istar,z[1:],'both',True)
        traces=np.array([np.trace(mats[r][0]) for r in range(3)])
        dets=np.array([np.linalg.det(mats[r][0]) for r in range(3)])
        discr=traces*traces-4*dets
        spectral=bool(traces[0]<0 and dets[0]>0 and traces[1]>0 and discr[1]<-1e-6 and traces[2]<0 and discr[2]<-1e-6)
        directions=bool(min(np.asarray(speeds)*[1,1,-1,-1])>1e-7)
        residence=[]
        for ix,r in enumerate(ROUTES['both'][0]):
            A,b=mats[r];u=flow(A,b,starts[ix],np.linspace(0,times[ix],301))[:,0]
            lo=-np.inf if r==0 else F.cut[r-1];hi=np.inf if r==2 else F.cut[r]
            residence.append(bool(np.min(u)>=lo-1e-8 and np.max(u)<=hi+1e-8))
        if not spectral or not directions or not all(residence):
            raise ValueError(f'{case}: polished fold fails checks; spectral={spectral}, directions={directions}, residence={residence}, fold={np.r_[residual,lm].tolist()}')
        candidate.update({'trace':traces.tolist(),'determinant':dets.tolist(),'discriminant':discr.tolist(),'speeds':np.asarray(speeds).tolist(),'sampled_residence':residence})
        candidate['matching_max']=float(max(abs(residual)));candidate['logM']=float(lm)
        if candidate['matching_max']>1e-7 or abs(lm)>1e-7: raise ValueError(f'{case}: converted parameterization no longer closes fold')
        out=ROOT/'results/fold_anchors'/case/'softplus'/'seed_0.json';out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n')
        print('WROTE',out,'matching',candidate['matching_max'],'logM',candidate['logM'],flush=True)

if __name__=='__main__': main()
