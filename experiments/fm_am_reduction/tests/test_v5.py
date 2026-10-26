import sys,json,unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.optimize import OptimizeResult
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from regional_coordinates import RegionalCoordinates
from moving_cycle_fit import anchor_section,MovingCycleObjective
from poincare import Section,PWASystem,LRSystem,first_return,smooth_counts
from full_cycle_fit import reference_section,DEFAULTS
from analyze_candidate import vector
from guarded_optimizer import solve_stage
from fit_v5 import finalize
from cycle_training_audit import collocation_levels,nested_nodes,calibrate_source_epsilon
from fit_stage import initial_vector
from run_fm import candidate_snapshot

ROOT=Path(__file__).resolve().parents[1]
MODEL=json.loads((ROOT/'tests'/'fixtures'/'fm_v4_stalled.json').read_text())
CASES=json.loads((ROOT/'config'/'cases.json').read_text())
PARAMETERS={**CASES['shared_parameters'],'K_ER':.05}


class Geometry:
    def residual(self,x):return np.array([x[0]-.1])
    def parts(self,x):return dict(nullclines=float((x[0]-.1)**2),precycle_total=float((x[0]-.1)**2))


class V5Checks(unittest.TestCase):
    def system(self,x,ip3=.8):return PWASystem(x,MODEL['physical_thresholds_C'],ip3,.5,2/1.185)

    def test_regional_roundtrip_and_derivative(self):
        x=vector(MODEL);c=RegionalCoordinates(MODEL['physical_thresholds_C']);y=c.from_raw(x)
        np.testing.assert_allclose(c.to_raw(y),x,atol=1e-12)
        J=c.jacobian(y);d=np.random.default_rng(42).normal(size=len(y));h=1e-6
        numeric=(c.to_raw(y+h*d)-c.to_raw(y-h*d))/(2*h)
        np.testing.assert_allclose(J@d,numeric,rtol=1e-6,atol=2e-8)

    def test_exact_warm_start_preserves_unresolved_regional_slopes(self):
        class CoarseGeometry:
            m=7;c_ref=.5;thresholds=np.array(MODEL['physical_thresholds_C'])
            c_null=np.linspace(.03,1.15,56);i_null=np.full(56,.8)
        result=initial_vector(CoarseGeometry(),0,MODEL)
        np.testing.assert_allclose(result,vector(MODEL),atol=1e-13)

    def test_snapshot_retries_partial_json_and_pins_used_bytes(self):
        payload=json.dumps({'candidate':MODEL}).encode()
        path=unittest.mock.Mock()
        path.read_bytes.side_effect=[b'{partial',payload]
        with patch('run_fm.time.sleep'):
            data,model=candidate_snapshot(path)
        self.assertEqual(data,payload);self.assertEqual(model,MODEL)

    def test_anchor_derivatives_both_nullcline_branches(self):
        x=vector(MODEL)
        for ip3 in [.65,.8]:
            lr=LRSystem(PARAMETERS,ip3);target=reference_section(ip3,PARAMETERS).C
            section,_=anchor_section(self.system(x,ip3),lr,target)
            for j in range(len(x)):
                h=1e-6;d=np.eye(len(x))[j]*h
                a,_=anchor_section(self.system(x+d,ip3),lr,target)
                b,_=anchor_section(self.system(x-d,ip3),lr,target)
                np.testing.assert_allclose([section.C_gradient[j],section.h0_gradient[j]],
                    [(a.C-b.C)/(2*h),(a.h0-b.h0)/(2*h)],rtol=2e-5,atol=1e-7)

    def test_external_section_derivatives_on_lr(self):
        source=LRSystem(PARAMETERS,.8);sec=Section(.359,.596,1,np.array([1.,0.]),np.array([0.,1.]))
        opts=dict(max_time=100,rtol=3e-10,atol=3e-12,max_step=.2)
        result=first_return(source,sec,.4,True,**opts)
        self.assertEqual(result['status'],'valid')
        for j in range(2):
            h=1e-6;a=[sec.C,sec.h0];b=a.copy();a[j]+=h;b[j]-=h
            plus=first_return(source,Section(*a),.4,**opts);minus=first_return(source,Section(*b),.4,**opts)
            numeric=(plus['G']-minus['G'])/(2*h)
            np.testing.assert_allclose(result['dG'][j],numeric,rtol=2e-5,atol=2e-6)

    def test_moving_section_recovers_small_cycle(self):
        x=vector(MODEL);source=LRSystem(PARAMETERS,.8)
        sec,_=anchor_section(self.system(x),source,reference_section(.8,PARAMETERS).C)
        self.assertLess(sec.C,.3670671);self.assertGreater(sec.C,.35098)
        nodes=np.unique(np.r_[np.geomspace(.0001,.1,18),np.linspace(.1,.98,18)])
        records=[first_return(self.system(x),sec,u,max_time=100) for u in nodes]
        count,_,edges=smooth_counts(records,.001)
        self.assertGreater(edges,2);np.testing.assert_allclose(count,[1,0],atol=.025)

    def test_full_moving_objective_directional_derivative(self):
        obj=MovingCycleObjective.__new__(MovingCycleObjective);x=vector(MODEL)
        obj.geometry=Geometry();obj.run={'parameters':PARAMETERS};obj.thresholds=np.array(MODEL['physical_thresholds_C'])
        obj.c_ref=.5;obj.m=7;obj.cfg={};obj.spec=dict(cycle_fit=dict(max_time=100,rtol=2e-9,atol=2e-11,max_step=.2),
            cycle_smoothing=dict(xi=1e-7,softplus_tau=.02),loss_weights=dict(cycles=30.,extra_cycles=30.))
        obj.epsilon=.5;obj.last_x=None;obj.last_epsilon=None;obj.nodes=[.02,.1,.3,.6,.9]
        obj.levels=[.8];obj.targets={.8:reference_section(.8,PARAMETERS).C};obj.source_inventory={}
        r=obj.residual(x).copy();J=obj.jac(x).copy()
        for j in [0,22]:
            h=2e-6;d=np.eye(len(x))[j]*h
            numeric=(obj.residual(x+d)-obj.residual(x-d))/(2*h)
            np.testing.assert_allclose(J[:,j],numeric,rtol=3e-3,atol=3e-4)
        obj.source_inventory={.8:{'coverage_C_interval':[.4,.5]}}
        obj.last_x=None;r=obj.residual(x).copy();J=obj.jac(x).copy()
        self.assertFalse(obj.parts(x)['return_domain_admissible'])
        self.assertIsNone(obj.report[0]['counts'])
        h=2e-6;d=np.eye(len(x))[0]*h
        numeric=(obj.residual(x+d)[-1]-obj.residual(x-d)[-1])/(2*h)
        self.assertAlmostEqual(J[-1,0],numeric,delta=1e-4)

    def test_xtol_success_does_not_mean_stationarity(self):
        x=vector(MODEL)
        class Toy:
            thresholds=np.array(MODEL['physical_thresholds_C']);c_ref=.5
            def residual(self,x):return x.copy()
            def jac(self,x):return np.eye(len(x))
        def stopped(fun,y,**kwargs):
            self.assertIsNone(kwargs['xtol']);self.assertIsNone(kwargs['ftol']);fun(y)
            return OptimizeResult(x=y,success=True,status=3,message='fake xtol')
        with patch('guarded_optimizer.least_squares',stopped):
            result=solve_stage(Toy(),x,1,label='test')
        self.assertFalse(result.success);self.assertFalse(result.converged_stationary)
        self.assertTrue(result.solver_success);self.assertGreater(result.optimality,1e-6)

    def test_training_levels_refine_missing_band_and_nested_nodes(self):
        coarse,fine=collocation_levels([.55,.65,.75,.78,.8,1.,1.1,1.14],warm_model=MODEL)
        self.assertTrue(any(.66<i<.72 for i in fine))
        self.assertLessEqual(max(np.diff(coarse)),.050000001)
        self.assertLessEqual(max(np.diff(fine)),.025000001)
        self.assertFalse(any(abs(i-.8064561482294)<1e-9 for i in fine))
        original=np.linspace(.001,.98,9);dense=nested_nodes(original,17)
        self.assertEqual(len(dense),17);self.assertTrue(all(x in dense for x in original))

    def test_failed_numerical_audit_cannot_be_successful(self):
        x=vector(MODEL)
        class Mini:
            thresholds=np.array(MODEL['physical_thresholds_C']);c_ref=.5;m=7
            cfg=dict(I_nodes=[.8],u_nodes=[.1,.5,.9],selection_nodes=5,stationarity_gtol=1e-6)
            epsilon_schedule=[.001];run=dict(case='FM',K_ER=.05)
            spec={};run_dir=Path('/unused');report=[];resolution_history=[]
            def set_levels(self,levels):pass
            def set_nodes(self,nodes):pass
            def residual(self,y):return y-x
            def jac(self,y):return np.eye(len(y))
            def parts(self,y):return dict(full_total=0.,return_domain_admissible=True)
        failed=dict(parts=dict(return_domain_admissible=True,cycle_coverage_admissible=True,cycle_resolution_admissible=False))
        with patch('fit_v5.audit_candidate',return_value=failed):
            result=finalize(Mini(),OptimizeResult(x=x,nfev=0,message='test'))
        self.assertTrue(result.gradient_stationarity)
        self.assertFalse(result.success);self.assertFalse(result.converged_stationary)

    def test_shared_epsilon_calibration_resolves_source_counter(self):
        cfg=dict(cycle_fit=dict(profile_nodes=9,count_tolerance=.025),
                 cycle_smoothing=dict(epsilon_schedule=[.1,.02],xi=1e-7))
        row=dict(I=.65,records=[dict(status='valid',G=g) for g in [1.,-.007,1.,-1.]],
                 cycles=[dict(kind=k) for k in ['stable','unstable','stable']],suspect_intervals=[])
        with tempfile.TemporaryDirectory() as tmp,patch('cycle_training_audit.prepare_source_inventory',return_value={'rows':[row]}):
            result=calibrate_source_epsilon(Path(tmp),cfg)
            self.assertTrue(result['resolved']);self.assertLess(result['actual_final_epsilon'],.02)
            self.assertEqual(cfg['cycle_smoothing']['epsilon_schedule'][-1],result['actual_final_epsilon'])

    def test_invalid_proposal_is_rejected_without_losing_incumbent(self):
        x=vector(MODEL)
        class Toy:
            thresholds=np.array(MODEL['physical_thresholds_C']);c_ref=.5
            def residual(self,y):
                if abs(y[0]-x[0])>1:raise ValueError('invalid proposal')
                return y.copy()
            def jac(self,y):return np.eye(len(y))
        def probe(fun,y,**kwargs):
            fun(y);bad=y.copy();bad[0]+=10;fun(bad)
            return OptimizeResult(x=bad,success=False,status=0,message='test')
        with patch('guarded_optimizer.least_squares',probe):
            result=solve_stage(Toy(),x,1,label='test')
        np.testing.assert_allclose(result.x,x,atol=1e-12)
        self.assertIn('error',result.iteration_history[-1]);self.assertFalse(result.iteration_history[-1]['admissible'])


if __name__=='__main__':unittest.main()
