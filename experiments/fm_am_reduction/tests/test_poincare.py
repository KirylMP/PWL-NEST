import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from poincare import Section,first_return,smooth_counts,isolate_cycles,PWASystem
from validate_structure import same_orbit
from full_cycle_fit import reference_section
from poincare import LRSystem
import json

class Radial:
    nparams=0
    bounds=np.array([[-1.2,1.2],[-1.2,1.2]])
    def coordinate(self,c):return c
    def coordinate_gradient(self,c):return np.empty(0)
    def physical(self,z):return z
    def field(self,y):
        x,h=y;s=x*x+h*h;q=(s-.09)*(.49-s)
        return np.array([-h+x*q,x+h*q])
    def jac(self,y):
        x,h=y;s=x*x+h*h;q=(s-.09)*(.49-s);dq=.58-2*s
        return np.array([[q+2*x*x*dq,-1+2*x*h*dq],[1+2*x*h*dq,q+2*h*h*dq]])
    def partials(self,y):return np.empty((2,0))

class ReturnChecks(unittest.TestCase):
    def test_first_return_is_full_period_and_multiplier(self):
        rec=first_return(Radial(),Section(0,0,-1),.3,max_time=20,max_step=.15)
        self.assertEqual(rec['status'],'valid');self.assertAlmostEqual(rec['period'],2*np.pi,places=6)
        self.assertAlmostEqual(rec['P'],.3,places=6)
        expected=np.exp(2*.09*(.49-.09)*2*np.pi)
        self.assertAlmostEqual(rec['multiplier'],expected,places=5)

    def test_repelling_and_attracting_cycles_enumerated(self):
        records=isolate_cycles(Radial(),Section(0,0,-1),np.linspace(.1,.95,19),dict(max_time=30,max_step=.2))
        self.assertEqual(sorted(q['kind'] for q in records['cycles']),['stable','unstable'])
        np.testing.assert_allclose(sorted(q['s'] for q in records['cycles']),[.3,.7],atol=1e-6)
        for cycle in records['cycles']:
            self.assertEqual(cycle['time_direction'],1 if cycle['kind']=='stable' else -1)
            self.assertLess(abs(cycle['return_residual']),1e-9)
        counts,_,_=smooth_counts(records['records'],.001)
        np.testing.assert_allclose(counts,[1,1],atol=.02)

    def test_backward_return_multiplier_is_inverse(self):
        result=first_return(Radial(),Section(0,0,1),.3,time_direction=-1,max_time=20,max_step=.15)
        self.assertEqual(result['status'],'valid')
        expected=np.exp(-2*.09*(.49-.09)*2*np.pi)
        self.assertAlmostEqual(result['multiplier'],expected,places=6)
        self.assertAlmostEqual(result['period'],2*np.pi,places=6)

    def test_brent_convergence_without_closure_is_rejected(self):
        table=[dict(status='valid',u=.1,G=1),dict(status='valid',u=.9,G=-1)]
        result=dict(status='valid',u=.5,G=.01,return_residual=.001,log_multiplier=-1.,Cmin=.1,Cmax=.9)
        with patch('poincare.return_table',return_value=table),patch('poincare.brentq',return_value=.5),patch('poincare.first_return',return_value=result):
            scan=isolate_cycles(Radial(),Section(0,0,-1),[.1,.9],{})
        self.assertEqual(scan['cycles'],[])
        self.assertTrue(any(q['message']=='root_does_not_close' for q in scan['suspect_intervals']))

    def test_near_zero_neighbor_requires_refined_contraction_checks(self):
        nodes=[.65,.66,.67,.68];root=.6600228
        def returned(system,section,u,*args,**kwargs):
            return dict(status='valid',u=u,G=(root-u)/u,return_residual=root-u,
                        log_multiplier=-2.,Cmin=.1,Cmax=.9,time_direction=kwargs.get('time_direction',1))
        table=[returned(None,None,u) for u in nodes]
        for direction in [1,-1]:
            for issue in [None,'turn','expansion','failure']:
                def probe(system,section,u,*args,**kwargs):
                    result=returned(system,section,u,*args,**kwargs)
                    if .65<u<.66:
                        if issue=='turn':result['G']=-.01
                        if issue=='expansion':result['log_multiplier']=.1
                        if issue=='failure':result=dict(status='timeout',u=u)
                    return result
                tables=[table,[]] if direction==1 else [[],table]
                with self.subTest(issue=issue,direction=direction),patch('poincare.return_table',side_effect=tables),\
                     patch('poincare.brentq',return_value=root),patch('poincare.first_return',side_effect=probe):
                    scan=isolate_cycles(Radial(),Section(0,0,-1),nodes,{})
                    self.assertEqual(len(scan['cycles']),1)
                    self.assertEqual(scan['cycles'][0]['kind'],'stable' if direction==1 else 'unstable')
                    self.assertEqual(bool(scan['suspect_intervals']),issue is not None)
                    self.assertEqual(len(scan['resolved_near_zero_intervals']),1 if issue is None else 0)

    def test_near_zero_without_closed_neighbor_remains_unresolved(self):
        table=[dict(status='valid',u=u,G=3e-5,log_multiplier=-2.) for u in [.65,.66,.67]]
        with patch('poincare.return_table',side_effect=[table,[]]):
            scan=isolate_cycles(Radial(),Section(0,0,-1),[.65,.66,.67],{})
        self.assertTrue(scan['suspect_intervals']);self.assertFalse(scan['resolved_near_zero_intervals'])

    def test_lr_server_calibration_neighbor_is_resolved(self):
        from cycle_training_audit import _source_profile
        cases=json.loads((Path(__file__).resolve().parents[1]/'config'/'cases.json').read_text())
        p={**cases['shared_parameters'],'K_ER':.05}
        row=_source_profile(.580960283961,p,[.65,.66,.67,.68],
            dict(max_time=100,rtol=2e-8,atol=2e-10,max_step=.5,max_evaluations=100000))
        self.assertEqual([q['kind'] for q in row['cycles']],['stable'])
        self.assertFalse(row['suspect_intervals']);self.assertTrue(row['resolved_near_zero_intervals'])
        count,_,edges=smooth_counts(row['records'],.001)
        self.assertGreaterEqual(edges,2);np.testing.assert_allclose(count,[1,0],atol=.025)

    def test_orbit_deduplication_uses_stability_and_crossings(self):
        orbit=dict(kind='stable',period=20.,Cmin=.03,Cmax=1.,orbit_crossings={'0.5':[.8]})
        self.assertTrue(same_orbit(orbit,{**orbit,'period':20.000001}))
        self.assertFalse(same_orbit(orbit,{**orbit,'kind':'unstable'}))
        self.assertFalse(same_orbit(orbit,{**orbit,'orbit_crossings':{'0.5':[.800002]}}))
        self.assertFalse(same_orbit(orbit,{**orbit,'orbit_crossings':{}}))

    def test_lr_strongly_repelling_cycle_is_closed_backward(self):
        cases=json.loads((Path(__file__).resolve().parents[1]/'config'/'cases.json').read_text())
        p={**cases['shared_parameters'],'K_ER':.05};ip3=87/81
        system=LRSystem(p,ip3);sec=reference_section(ip3,p)
        settings=dict(max_time=120,rtol=3e-10,atol=3e-12,max_step=.25,
                      capture_points=([sec.C,sec.h0],),witness_sections=[sec.C,.5])
        scan=isolate_cycles(system,sec,np.linspace(.02,.95,21),settings)
        self.assertEqual(sorted(q['kind'] for q in scan['cycles']),['stable','unstable'])
        for cycle in scan['cycles']:
            direction=cycle['time_direction'];other=Section(.5,system.nullcline(.5),direction)
            s=cycle['orbit_crossings']['0.5'][0];u=(s-other.h0)/(1-other.h0)
            second=first_return(system,other,u,time_direction=direction,**settings)
            self.assertEqual(second['status'],'valid')
            self.assertTrue(same_orbit(cycle,{**second,'kind':cycle['kind']}))
        unstable=next(q for q in scan['cycles'] if q['kind']=='unstable')
        self.assertGreater(unstable['multiplier'],1e6)
        self.assertLess(abs(unstable['G']),1e-7)
        independent=first_return(system,Section(sec.C,sec.h0,-1),unstable['u'],time_direction=-1,
            method='RK45',max_time=120,max_step=.1,rtol=1e-10,atol=1e-12)
        self.assertEqual(independent['status'],'valid')
        self.assertLess(abs(independent['return_residual']),2e-8)
        self.assertAlmostEqual(independent['period'],unstable['period'],places=5)

    def test_invalid_nodes_do_not_form_edges(self):
        records=[dict(status='valid',G=2),dict(status='timeout'),dict(status='valid',G=-2)]
        counts,_,edges=smooth_counts(records,.02)
        self.assertEqual(edges,0);np.testing.assert_array_equal(counts,[0,0])
        # Callers must inspect edges=0 and mark the domain unresolved.

    def test_independent_integrators_agree_on_pwa_orbit(self):
        theta=np.r_[[1.5,2,0,-2.5,2.5],[.5,-1,0,0,0],0,np.log(.1),2,-2,0,0]
        system=PWASystem(theta,np.array([.3,.7]),.7,.5,1.68)
        a=first_return(system,Section(.55,.5),.4,max_time=100,rtol=2e-10,atol=2e-12)
        b=first_return(system,Section(.55,.5),.4,max_time=100,rtol=2e-10,atol=2e-12,method='RK45')
        self.assertEqual(a['status'],'valid');self.assertEqual(b['status'],'valid')
        np.testing.assert_allclose([a['P'],a['period'],a['Cmin'],a['Cmax']],
                                   [b['P'],b['period'],b['Cmin'],b['Cmax']],rtol=2e-6,atol=2e-7)

    def test_counter_derivative(self):
        x=.13;nodes=np.linspace(-1,1,17)
        def records(v):return [dict(status='valid',G=t-v,dG=[-1.]) for t in nodes]
        count,jac,_=smooth_counts(records(x),.2,nparams=1)
        step=1e-6
        numeric=(smooth_counts(records(x+step),.2)[0]-smooth_counts(records(x-step),.2)[0])/(2*step)
        np.testing.assert_allclose(jac[:,0],numeric,atol=1e-8)

    def test_pwa_moving_section_sensitivities(self):
        theta=np.r_[[1.5,2,0,-2.5,2.5],[.5,-1,0,0,0],0,np.log(.1),2,-2,0,0]
        thresholds=np.array([.3,.7]);section=Section(.55,.5)
        def system(v):return PWASystem(v,thresholds,.7,.5,1.68)
        opts=dict(max_time=100,max_step=.1,rtol=2e-9,atol=2e-11)
        result=first_return(system(theta),section,.4,True,**opts)
        self.assertEqual(result['status'],'valid')
        plain=first_return(system(theta),section,.4,False,**opts)
        self.assertEqual(plain['status'],'valid')
        np.testing.assert_allclose([plain['P'],plain['period'],plain['log_multiplier']],
                                   [result['P'],result['period'],result['log_multiplier']],rtol=2e-6,atol=2e-7)
        for j in range(len(theta)):
            d=np.eye(len(theta))[j]*2e-5
            plus=first_return(system(theta+d),section,.4,False,**opts)
            minus=first_return(system(theta-d),section,.4,False,**opts)
            self.assertEqual(plus['status'],'valid');self.assertEqual(minus['status'],'valid')
            numeric=(plus['G']-minus['G'])/(4e-5)
            np.testing.assert_allclose(result['dG'][j],numeric,rtol=.01,atol=3e-4,err_msg=f'parameter {j}')

if __name__=='__main__':unittest.main()
