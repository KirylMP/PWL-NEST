"""Exploratory continuous three-zone field fit in log/square coordinates.

Fits LR vector-field samples on seven training cycles, enforcing the three
equilibria at I=0,I1,I2. The cycle fold is NOT constrained in this experiment.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import brentq, minimize

ROOT=Path(__file__).resolve().parents[1]
KSETS={'AM':.1,'AFM':.075,'FM':.05}

def lr(I,y,K):
    C,h=y; Q=1.049*(I+.13)/(I+.9434)
    return np.array([(6*(I/(I+.13))**3*(C/(C+.08234))**3*h**3+.11)*(2-1.185*C)-.9*C*C/(C*C+K*K), .2*(Q-(Q+C)*h)])

def equilibrium(I,K):
    Q=1.049*(I+.13)/(I+.9434)
    def g(c): return lr(I,(c,Q/(Q+c)),K)[0]
    return np.array([brentq(g,1e-8,1.6),0.]) + np.array([0.,Q/(Q+brentq(g,1e-8,1.6))])

def fwd(c,kind):
    return np.log(c) if kind=='log' else np.sqrt(c)
def inverse(u,kind):
    return np.exp(u) if kind=='log' else u*u
def deriv_u(c,kind):
    return 1/c if kind=='log' else .5/np.sqrt(c)

def features(u,q,I,cuts):
    return np.array([1.,u,q,I,max(u-cuts[0],0.),max(u-cuts[1],0.)])

def integrate_cycle(fun,initial,duration):
    def peak(t,y): return fun(t,y)[0]
    peak.direction=-1
    sol=solve_ivp(fun,(0.,duration),initial,method='DOP853',rtol=2e-7,atol=2e-9,max_step=.25,events=peak,dense_output=True)
    if not sol.success or not np.isfinite(sol.y).all(): return None
    peaks=sol.t_events[0]; peaks=peaks[peaks>duration*.5]
    if len(peaks)<3:return None
    periods=np.diff(peaks[-3:]); period=periods[-1]
    if abs(periods[-1]/periods[-2]-1)>.03:return None
    tt=peaks[-2]+period*np.arange(256)/256
    return {'initial':sol.y[:,-1].tolist(),'period':float(period),'states':sol.sol(tt).T.tolist()}

def make_reference(case,base,report):
    train=[r['I'] for r in report['best']['cycles']]
    hold=[r['I'] for r in report['holdout']]
    data=[]
    for mode,inputs in (('train',train),('holdout',hold)):
        for I in inputs:
            ref=integrate_cycle(lambda t,y:lr(I,y,KSETS[case]),equilibrium(0.,KSETS[case]),600.)
            if ref is None: raise ValueError(f'LR cycle unresolved: {case} I={I}')
            data.append({'I':I,'set':mode,**ref})
    return data

def fit_model(kind,case,base,data):
    cuts=np.array([fwd(v,kind) for v in base['physical_thresholds_C']]);I1,I2=base['fixed_equilibrium_threshold_I']
    C0,h0=equilibrium(0.,KSETS[case]);anchor=[(0.,fwd(C0,kind),h0)]
    for I,C in zip((I1,I2),base['physical_thresholds_C']):
        Q=1.049*(I+.13)/(I+.9434);h=Q/(Q+C)
        if np.linalg.norm(lr(I,(C,h),KSETS[case]))>1e-8:raise ValueError('specified LR anchor unresolved')
        anchor.append((I,fwd(C,kind),h))
    X0=np.array([features(u,h,I,cuts) for I,u,h in anchor]);N=null_space(X0)
    # homogeneous constraints X0 theta=0 hold independently for both rows
    X=[];Y=[]
    for d in data:
        if d['set']!='train':continue
        I=d['I']
        for C,h in np.array(d['states'])[::4]:
            F=lr(I,(C,h),KSETS[case]);X.append(features(fwd(C,kind),h,I,cuts));Y.append([deriv_u(C,kind)*F[0],F[1]])
    X=np.asarray(X);Y=np.asarray(Y);Z=X@N
    initial=np.linalg.lstsq(Z,Y,rcond=None)[0]
    scales=np.maximum(np.sqrt(np.mean(Y*Y,axis=0)),1e-5)
    def unpack(v):return N@v.reshape(initial.shape)
    def loss(v):return float(np.mean(((Z@v.reshape(initial.shape)-Y)/scales)**2))
    def spectral(v):
        coef=unpack(v); traces=[];dets=[]
        for j in range(3):
            A=np.column_stack((coef[1,:]+np.sum(coef[4:4+j,:],axis=0),coef[2,:]))
            traces.append(np.trace(A));dets.append(np.linalg.det(A))
        T=np.asarray(traces);D=np.asarray(dets);disc=T*T-4*D
        return np.array([-T[0]-.001,D[0]-.00001,T[1]-.001,-disc[1]-.00001,-T[2]-.001,-disc[2]-.00001])
    rng=np.random.default_rng(193); opt=None;best=None
    for attempt in range(12):
        start=initial.ravel() if attempt==0 else initial.ravel()+rng.normal(0,.25*(1+np.abs(initial.ravel())),initial.size)
        trial=minimize(loss,start,method='SLSQP',constraints={'type':'ineq','fun':spectral},options={'maxiter':800,'ftol':1e-9})
        if min(spectral(trial.x)) >= -1e-7 and np.isfinite(trial.fun):
            if best is None or trial.fun<best.fun:best=trial
            if trial.success:break
    opt=best
    if opt is None: raise ValueError(f'spectral fit failed after 12 starts; last: {trial.message} {spectral(trial.x)}')
    coef=unpack(opt.x)
    return {'case':case,'transform':kind,'C_ref_uM':1.,'cuts_u':cuts.tolist(),'coefficient_rows':coef.T.tolist(),
            'rest_state':[anchor[0][1],anchor[0][2]],'anchors':anchor,'fold_constraint':'NOT enforced',
            'spectral_constraint_residuals':spectral(opt.x).tolist(),'optimizer_message':opt.message,'optimizer_success':bool(opt.success),
            'fit':'constrained least squares on LR field along seven train cycles; not previous cycle loss'}

def calc(model,data):
    cuts=model['cuts_u'];a=np.array(model['coefficient_rows']);kind=model['transform'];out=[]
    for d in data:
        I=d['I']
        def f(t,y):return a@features(y[0],y[1],I,cuts)
        v=integrate_cycle(f,model['rest_state'],600.)
        target=np.array(d['states'])[:,0];item={'I':I,'set':d['set'],'reference':{'Cmin':float(min(target)),'Cmax':float(max(target)),'period':d['period']}}
        if v:
            u=np.array(v['states'])[:,0];C=inverse(u,kind)
            item['prediction']={'Cmin':float(min(C)),'Cmax':float(max(C)),'period':v['period'],'umin':float(min(u)),'umax':float(max(u)),'C':C.tolist()}
        else:item['prediction']=None
        item['target_C']=target.tolist();out.append(item)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=KSETS,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    base=json.loads((ROOT/'paper/data/random_selected'/args.case/'candidate.json').read_text()); report=json.loads((ROOT/'paper/data/random_selected'/args.case/'report.json').read_text());args.out.mkdir(parents=True,exist_ok=True)
    refs=make_reference(args.case,base,report)
    for kind in ('log','square'):
        model=fit_model(kind,args.case,base,refs);rows=calc(model,refs)
        path=args.out/(args.case+'_'+kind+'.json');path.write_text(json.dumps({'model':model,'results':rows},indent=2,allow_nan=False)+'\n')
        print(args.case,kind,'cycles',sum(r['prediction'] is not None for r in rows),'/',len(rows),flush=True)
if __name__=='__main__':main()
