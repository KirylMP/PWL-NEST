"""Complete five-term FM objective and epsilon continuation support."""
import json
import numpy as np
from scipy.special import expit
from fit_stage import MixedObjective,model_roots
from prepare_reference_data import roots_for_ip3
from poincare import LRSystem,PWASystem,Section,return_table,smooth_counts

DEFAULTS=dict(enabled=True,I_nodes=[.55,.65,.8,1.,1.1,1.14],u_nodes=sorted(set(
    np.r_[np.geomspace(.001,.1,5),np.linspace(.1,.98,13)].tolist())),
    max_time=400.,rtol=2e-8,atol=2e-10,max_step=.5,max_evaluations=100000,
    max_nfev_per_epsilon=60,min_regular_edge_fraction=.5)


def settings(config):
    c={**DEFAULTS,**config.get('cycle_fit',{})}
    return {k:c[k] for k in ['max_time','rtol','atol','max_step','max_evaluations']}


def reference_section(ip3,p):
    roots=roots_for_ip3(ip3,p,p['C_0']/(1+p['c_1']),5000)
    if not roots:raise ValueError(f'No reference equilibrium for section at I={ip3}')
    preferred=[q for q in roots if q['determinant']>0 and q['trace']>0]
    center=(preferred or roots)[len(preferred or roots)//2]
    return Section(center['C'],center['h'])


def prepare_cycle_reference(run_dir,config):
    from plan import read_json
    run=read_json(run_dir/'run.json');p=run['parameters'];cfg={**DEFAULTS,**config.get('cycle_fit',{})}
    rows=[]
    for ip3 in cfg['I_nodes']:
        sec=reference_section(ip3,p)
        table=return_table(LRSystem(p,ip3),sec,cfg['u_nodes'],**settings(config))
        _,_,edges=smooth_counts(table,.02)
        if edges<2:
            raise ValueError(f'Reference section at I={ip3} has fewer than two regular edges; enlarge return budget or revise section/grid. Not counted as zero cycles.')
        rows.append(dict(I=ip3,section=dict(C=sec.C,h0=sec.h0,orientation=1),records=table,regular_edges=edges))
        print(f'Reference returns I={ip3}: {edges} regular edges',flush=True)
    path=run_dir/'reference'/'poincare.json'
    path.write_text(json.dumps(dict(policy='source equilibrium section; shared physical C,h0,u and positive C orientation',
        config=cfg,solver=settings(config),rows=rows),indent=2,allow_nan=False)+'\n')
    return rows


class FullCycleObjective:
    def __init__(self,run_dir,config):
        self.geometry=MixedObjective(run_dir,config)
        self.spec=config;self.cfg={**DEFAULTS,**config.get('cycle_fit',{})}
        self.run=self.geometry.run;self.m=self.geometry.m;self.c_ref=self.geometry.c_ref
        self.thresholds=self.geometry.thresholds
        data=json.loads((run_dir/'reference'/'poincare.json').read_text())
        if data['solver']!=settings(config) or data['config']['I_nodes']!=self.cfg['I_nodes'] or data['config']['u_nodes']!=self.cfg['u_nodes']:
            raise ValueError('Reference return cache differs from run config; use a fresh experiment directory')
        self.rows=data['rows'];self.nodes=data['config']['u_nodes']
        self.epsilon_schedule=config['cycle_smoothing']['epsilon_schedule']
        self.epsilon=self.epsilon_schedule[-1];self.last_x=None
    def initial_normal_gradient(self,system,section,u):
        y=np.array([system.coordinate(section.C),section.h0+(1-section.h0)*u])
        f=system.field(y);grad=system.partials(y)[0]+system.jac(y)[0,0]*system.coordinate_gradient(section.C)
        return f[0],grad
    def evaluate(self,x):
        if self.last_x is not None and np.array_equal(self.last_x,x) and self.last_epsilon==self.epsilon:return
        same_x=self.last_x is not None and np.array_equal(self.last_x,x)
        n=len(x);m=len(self.rows)
        if same_x:
            base=self.cached_base;base_jac=self.cached_base_jac
        else:
            base=self.geometry.residual(x)
            # Geometry-only differences; return derivatives use variational ODEs.
            base_jac=np.empty((len(base),n))
            for j in range(n):
                step=2e-6*max(1.,abs(x[j]));delta=np.eye(n)[j]*step
                base_jac[:,j]=(self.geometry.residual(x+delta)-self.geometry.residual(x-delta))/(2*step)
            self.tables=[]
        
        cyc=[];cycjac=[];extra=[];extrajac=[];barriers=[];barrierjac=[];reports=[]
        cfg=self.spec;xi=cfg['cycle_smoothing']['xi'];tau=cfg['cycle_smoothing']['softplus_tau']
        cmax=self.run['parameters']['C_0']/(1+self.run['parameters']['c_1'])
        for row_index,row in enumerate(self.rows):
            sec=Section(**row['section']);system=PWASystem(x,self.thresholds,row['I'],self.c_ref,cmax)
            if same_x:table=self.tables[row_index]
            else:
                eq=model_roots(x,self.thresholds,row['I'],self.c_ref,1e-8,cmax-1e-8)
                capture=[]
                for c,h in eq:
                    if not (0<c<cmax and 0<h<1):continue
                    y=np.array([system.coordinate(c),h])
                    if np.max(np.linalg.eigvals(system.jac(y)).real)<-1e-8:capture.append(y)
                table=return_table(system,sec,self.nodes,True,capture_points=tuple(capture),**settings(cfg))
                self.tables.append(table)
            counts,dc,edges=smooth_counts(table,self.epsilon,xi,n)
            truth,_,refedges=smooth_counts(row['records'],self.epsilon,xi)
            required=max(2,int(np.ceil(self.cfg['min_regular_edge_fraction']*refedges)))
            valid=edges>=required
            if not valid:
                # An infeasible return domain is never interpreted as zero cycles.
                cyc.extend([1000.,1000.]);cycjac.extend(np.zeros((2,n)))
                extra.append(1000.);extrajac.append(np.zeros(n))
            else:
                w=np.sqrt(cfg['loss_weights']['cycles']/m)
                cyc.extend(w*(counts-truth));cycjac.extend(w*dc)
                v=counts.sum()-2;t=truth.sum()-2
                w=np.sqrt(cfg['loss_weights']['extra_cycles'])
                extra.append(w*tau*(np.logaddexp(0,v/tau)-np.logaddexp(0,t/tau)))
                extrajac.append(w*expit(v/tau)*dc.sum(axis=0))
            barrier=0.;db=np.zeros(n)
            if not valid:
                barrier=1000.*(required-edges)/required
                # A launch-orientation barrier provides a direction when initial
                # section points are on the wrong side of the candidate nullcline.
                for u,ref in zip(self.nodes,row['records']):
                    if ref['status']!='valid':continue
                    normal,dnormal=self.initial_normal_gradient(system,sec,u)
                    if normal<1e-7:
                        barrier+=10*(1e-7-normal);db-=10*dnormal
            barriers.append(barrier);barrierjac.append(db)
            reports.append(dict(I=row['I'],counts=counts.tolist() if valid else None,
                reference_counts=truth.tolist(),regular_edges=edges,required_edges=required,
                admissible=valid,records=table))
        self.value=np.r_[base,cyc,extra,barriers]
        self.gradient=np.vstack([base_jac,np.asarray(cycjac),np.asarray(extrajac),np.asarray(barrierjac)])
        self.cached_base=base;self.cached_base_jac=base_jac
        self.report=reports;self.last_x=x.copy();self.last_epsilon=self.epsilon
        self.parts_value=self.geometry.parts(x)
        self.parts_value.pop('precycle_total',None)
        self.parts_value.update(cycles_weighted=float(np.dot(cyc,cyc)),extra_cycles_weighted=float(np.dot(extra,extra)),
            return_domain_barrier=float(np.dot(barriers,barriers)),
            full_total=float(self.value@self.value),epsilon=self.epsilon,
            return_domain_admissible=all(q['admissible'] for q in reports))
    def residual(self,x):self.evaluate(x);return self.value
    def jac(self,x):self.evaluate(x);return self.gradient
    def parts(self,x):self.evaluate(x);return self.parts_value
