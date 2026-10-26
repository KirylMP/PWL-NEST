"""Shared moving physical sections with derivatives on BOTH return maps.

The section follows a candidate equilibrium. All numerical comparisons apply
the identical C,h0,u to LR and PWA, including section derivatives on LR.
Coverage/resolution are audited separately; edge counts are not coverage proof.
"""
import json
import numpy as np
from scipy.special import expit
from fit_stage import MixedObjective,model_roots,hinge_design,phi_prime
from prepare_reference_data import jacobian
from poincare import Section,LRSystem,PWASystem,return_table,smooth_counts
from full_cycle_fit import DEFAULTS,reference_section,settings


def anchor_section(system,source,target_C):
    """Nearest physical equilibrium and analytic derivatives of its position."""
    x=system.theta;n=system.nparams;k=system.ncoef
    roots=model_roots(x,system.thresholds,system.ip3,system.c_ref,system.cmin,system.cmax)
    physical=[(c,h) for c,h in roots if system.cmin<c<system.cmax and 0<h<1]
    if not physical:raise ValueError('no_physical_equilibrium_for_moving_section')
    c,h=min(physical,key=lambda q:abs(q[0]-target_C));z=system.coordinate(c)
    region=int(np.searchsorted(system.zt,z,side='right'))
    slope=system.ns[region]-system.hs[region]
    if abs(slope)<1e-8:raise ValueError('ill_conditioned_equilibrium_anchor')
    design=hinge_design(np.array([z]),np.array([system.ip3]),system.zt)[0]
    dN=np.zeros(n);dH=np.zeros(n);dN[:k]=design;dH[k:2*k]=design
    active=z>system.zt
    dN[-4:]=-(system.alpha[3:]*active)@system.dzt
    dH[-4:]=-(system.beta[3:]*active)@system.dzt
    dz=-(dN-dH)/slope
    dc=(dz-system.coordinate_gradient(c))/float(phi_prime(c,system.eta,system.c_ref))
    dh=dH+system.hs[region]*dz
    source_h=source.nullcline(c)
    if source_h is None or not np.isfinite(source_h):raise ValueError('undefined_source_nullcline_at_anchor')
    # This interval includes every physical positive crossing of either field.
    # The min/max branches are frozen locally; changes are logged by callers.
    if min(h,source_h)<=0:h0=0.;dh0=np.zeros(n);branch='physical_boundary'
    elif h<=source_h:h0=h;dh0=dh;branch='candidate_nullcline'
    else:
        j=jacobian(c,source_h,source.ip3,source.p)
        if abs(j[0,1])<1e-12:raise ValueError('source_nullcline_tangency_at_anchor')
        h0=source_h;dh0=(-j[0,0]/j[0,1])*dc;branch='source_nullcline'
    if h0>=1-1e-6:raise ValueError('empty_physical_launch_interval')
    return Section(c,float(h0),1,dc,dh0),dict(C=c,h0=float(h0),branch=branch,
        equilibrium_h=h,region=region,physical_equilibria=len(physical),slope_difference=float(slope),
        equilibrium_on_threshold=bool(np.min(abs(system.zt-z))<1e-9))


class MovingCycleObjective:
    def __init__(self,run_dir,config):
        self.run_dir=run_dir
        self.geometry=MixedObjective(run_dir,config);self.spec=config
        self.cfg={**DEFAULTS,**config.get('cycle_fit',{})};self.run=self.geometry.run
        self.thresholds=self.geometry.thresholds;self.c_ref=self.geometry.c_ref;self.m=self.geometry.m
        self.epsilon_schedule=config['cycle_smoothing']['epsilon_schedule']
        self.epsilon=self.epsilon_schedule[-1];self.last_x=None;self.last_epsilon=None
        self.nodes=np.asarray(self.cfg['u_nodes']);self.levels=list(self.cfg['I_nodes'])
        self.targets={float(i):reference_section(i,self.run['parameters']).C for i in self.levels}
        self.resolution_history=[]
        inventory=run_dir/'reference'/'source_cycle_inventory.json'
        self.source_inventory={r['I']:{k:r[k] for k in ['coverage_C_interval','regular_edges','suspect_intervals']}
            for r in json.loads(inventory.read_text())['rows']} if inventory.exists() else {}

    def set_nodes(self,nodes):
        updated=np.unique(np.asarray(nodes,dtype=float))
        if not np.array_equal(updated,self.nodes):self.nodes=updated;self.last_x=None

    def set_levels(self,levels):
        updated=list(levels)
        if updated==self.levels:return
        self.levels=updated
        for ip3 in self.levels:
            if float(ip3) not in self.targets:self.targets[float(ip3)]=reference_section(ip3,self.run['parameters']).C
        self.last_x=None

    def section(self,system,ip3):
        return anchor_section(system,LRSystem(self.run['parameters'],ip3),self.targets[float(ip3)])

    def evaluate(self,x):
        if self.last_x is not None and np.array_equal(x,self.last_x) and self.last_epsilon==self.epsilon:return
        same=self.last_x is not None and np.array_equal(x,self.last_x)
        n=len(x);cfg=self.spec;xi=cfg['cycle_smoothing']['xi'];tau=cfg['cycle_smoothing']['softplus_tau']
        if same:base=self.base;Jbase=self.base_jac
        else:
            # A rejected/failed proposal must not leave a partially populated
            # table cache attached to the previous successful parameter vector.
            self.last_x=None
            base=self.geometry.residual(x);Jbase=np.empty((len(base),n));self.tables=[]
            for j in range(n):
                h=2e-6*max(1.,abs(x[j]));d=np.eye(n)[j]*h
                Jbase[:,j]=(self.geometry.residual(x+d)-self.geometry.residual(x-d))/(2*h)
        residuals=[];jacobians=[];extra=[];Jextra=[];barriers=[];Jbarriers=[];reports=[]
        cmax=self.run['parameters']['C_0']/(1+self.run['parameters']['c_1'])
        for index,ip3 in enumerate(self.levels):
            system=PWASystem(x,self.thresholds,ip3,self.c_ref,cmax)
            if same:cached=self.tables[index]
            else:
                try:
                    sec,info=self.section(system,ip3)
                    # Two independent section coordinates suffice for LR; chain
                    # them to all 26 model parameters after integration.
                    lrsec=Section(sec.C,sec.h0,1,np.array([1.,0.]),np.array([0.,1.]))
                    truth_table=return_table(LRSystem(self.run['parameters'],ip3),lrsec,self.nodes,True,**settings(cfg))
                    for q in truth_table:
                        if q['status']=='valid':
                            a,b=q['dG'];q['dG']=(a*np.asarray(sec.C_gradient)+b*np.asarray(sec.h0_gradient)).tolist()
                    table=return_table(system,sec,self.nodes,True,**settings(cfg))
                    cached=dict(table=table,truth=truth_table,section=info,error=None)
                except ValueError as exc:
                    cached=dict(table=[],truth=[],section=None,error=str(exc))
                self.tables.append(cached)
            table=cached['table'];truth_table=cached['truth']
            counts,dc,edges=smooth_counts(table,self.epsilon,xi,n)
            truth,dt,redges=smooth_counts(truth_table,self.epsilon,xi,n)
            valid=cached['error'] is None and min(edges,redges)>=2
            covered=True;barrier=0.;dbarrier=np.zeros(n)
            profile=self.source_inventory.get(float(ip3),{})
            interval=profile.get('coverage_C_interval')
            if valid and interval:
                low,high=interval;low+=1e-7;high-=1e-7
                if not low<cached['section']['C']<high:
                    covered=False
                    # A numerical coverage barrier has a derivative and is zero
                    # throughout the covered domain. It is not a sixth fit term
                    # on admissible candidates.
                    current,info=self.section(system,ip3)
                    if current.C<=low:barrier=100*(low-current.C)/.05;dbarrier=-2000*np.asarray(current.C_gradient)
                    else:barrier=100*(current.C-high)/.05;dbarrier=2000*np.asarray(current.C_gradient)
            # The v4 half-reference-edge quota created an artificial feasibility
            # cliff. V5 requires actual regular intervals; post-fit audits also
            # require cycle coverage and counter convergence, independently.
            if valid:
                w=np.sqrt(cfg['loss_weights']['cycles']/len(self.levels))
                residuals.extend(w*(counts-truth));jacobians.extend(w*(dc-dt))
                v=counts.sum()-2;t=truth.sum()-2;w=np.sqrt(cfg['loss_weights']['extra_cycles'])
                extra.append(w*tau*(np.logaddexp(0,v/tau)-np.logaddexp(0,t/tau)))
                Jextra.append(w*(expit(v/tau)*dc.sum(axis=0)-expit(t/tau)*dt.sum(axis=0)))
                barriers.append(barrier);Jbarriers.append(dbarrier)
            else:
                residuals.extend([1000.,1000.]);jacobians.extend(np.zeros((2,n)))
                extra.append(1000.);Jextra.append(np.zeros(n))
                barriers.append(1000.);Jbarriers.append(np.zeros(n))
            reports.append(dict(I=ip3,counts=counts.tolist() if valid and covered else None,
                partial_variation=counts.tolist(),
                reference_counts=truth.tolist() if valid else None,regular_edges=edges,reference_edges=redges,
                admissible=valid and covered,regular_domain=valid,source_coverage=covered,
                section=cached['section'],error=cached['error'],records=table,reference_records=truth_table))
        self.value=np.r_[base,residuals,extra,barriers]
        self.gradient=np.vstack([Jbase,np.asarray(jacobians),np.asarray(Jextra),np.asarray(Jbarriers)])
        self.base=base;self.base_jac=Jbase;self.report=reports;self.last_x=x.copy();self.last_epsilon=self.epsilon
        self.parts_value=self.geometry.parts(x);self.parts_value.pop('precycle_total',None)
        self.parts_value.update(cycles_weighted=float(np.dot(residuals,residuals)),
            extra_cycles_weighted=float(np.dot(extra,extra)),return_domain_barrier=float(np.dot(barriers,barriers)),
            full_total=float(self.value@self.value),epsilon=self.epsilon,
            return_domain_admissible=all(q['admissible'] for q in reports),
            section_policy='moving_candidate_equilibrium_shared_with_LR')

    def residual(self,x):self.evaluate(x);return self.value
    def jac(self,x):self.evaluate(x);return self.gradient
    def parts(self,x):self.evaluate(x);return self.parts_value

    def refine_nodes(self,x,max_nodes=257):
        """Insert observed sign/curvature intervals between least-squares stages.

        The residual layout stays fixed during each solver invocation. Final
        ranking uses a separate common grid, so stage grids cannot bias winners.
        """
        self.evaluate(x);intervals=set()
        for row in self.report:
            for records in (row['records'],row['reference_records']):
                for j,(a,b) in enumerate(zip(records[:-1],records[1:])):
                    if a['status']!=b['status']:intervals.add(j)
                    elif a['status']=='valid' and (a['G']*b['G']<=0 or min(abs(a['G']),abs(b['G']))<4*self.epsilon):intervals.add(j)
                for j in range(1,len(records)-1):
                    a,b,c=records[j-1:j+2]
                    if a['status']==b['status']==c['status']=='valid' and (b['G']-a['G'])*(c['G']-b['G'])<0:
                        intervals.update([j-1,j])
        add=[.5*(self.nodes[j]+self.nodes[j+1]) for j in sorted(intervals)]
        if len(self.nodes)+len(add)>max_nodes:add=add[:max(0,max_nodes-len(self.nodes))]
        previous=len(self.nodes);self.set_nodes(np.r_[self.nodes,add])
        self.resolution_history.append(dict(epsilon=self.epsilon,before=previous,after=len(self.nodes),criterion='sign/extremum/domain-boundary refinement'))
        return len(self.nodes)>previous
