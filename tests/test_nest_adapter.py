"""Integration tests require the built NEST 3.10 extension."""
import unittest
import numpy as np

from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import simulate_coordinates
from pwl_nest.protocols import simulate
from pwl_nest.nest_adapter import build_network, load_candidate, sic

try:
    import nest
    HAVE_NEST=(ROOT/'.build/pwl_astro_module.so').exists()
except ImportError:
    HAVE_NEST=False


@unittest.skipUnless(HAVE_NEST,'Build the NEST module first')
class NestAdapterTests(unittest.TestCase):
    def test_both_models_and_sic_delivery_against_independent_reference(self):
        for case in ['AM','FM']:
            model,_=load_candidate(case)
            initial=[.08,.8,.2] if case=='AM' else model.metadata['exact_rest_constraint']
            base=initial[2]
            protocol=dict(duration=65.,tau=20.,baseline=base,jumps=[[20.,.7]])
            for kind in ['lr','pwa']:
                with self.subTest(case=case,kind=kind):
                    astro,_,meters=build_network(case,model,initial,protocol,kind=kind,
                        resolution_ms=10.,record_interval_ms=10.)
                    nest.Simulate(protocol['duration']*1000.)
                    data=meters[0].events;td=data['times']/1000.
                    x=np.column_stack([data[f] for f in ['Ca_astro','h_IP3R','IP3']])
                    if kind=='pwa':
                        tr,xr,_=simulate_coordinates(model,initial,**protocol,dt=.01,
                            rtol=2e-11,atol=2e-13,max_step=.01)
                    else:
                        tr,xr,_=simulate(case,initial,**protocol,dt=.01,
                            rtol=2e-11,atol=2e-13,max_step=.01)
                    ref=np.column_stack([np.interp(td,tr,xr[:,d]) for d in range(3)])
                    np.testing.assert_allclose(x[:,:2],ref[:,:2],rtol=0,atol=3e-5)
                    # Reference protocol returns the left I value at the exact jump.
                    away=np.abs(td-20.)>1e-7
                    np.testing.assert_allclose(x[away,2],ref[away,2],rtol=0,atol=1e-9)
                    incoming=meters[1].events['I_SIC']
                    np.testing.assert_allclose(incoming[1:],sic(x[:-1,0]),rtol=0,atol=1e-11)
                    self.assertGreater(float(incoming.max()),0.)
                    if kind=='pwa':
                        self.assertGreater(astro.get('region_crossings'),0)
                        self.assertEqual(astro.get('input_spikes'),1)

    def test_split_simulation_and_threaded_population(self):
        model,_=load_candidate('FM')
        initial=model.metadata['exact_rest_constraint']
        p=dict(tau=20.,baseline=.45,jumps=[[1.,.7]])
        endings=[]
        for split in [False,True]:
            a,_,_=build_network('FM',model,initial,p,population=8,threads=2,record=False)
            if split:
                nest.Simulate(1000.);nest.Simulate(9000.)
            else:
                nest.Simulate(10000.)
            endings.append(np.array(a.get('Ca_astro')))
        np.testing.assert_allclose(endings[0],endings[1],rtol=0,atol=1e-13)
        np.testing.assert_allclose(endings[0],endings[0][0],rtol=0,atol=1e-13)


if __name__=='__main__':
    unittest.main()
