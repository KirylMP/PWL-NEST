import unittest
import numpy as np
from pwl_nest.peak_shape import peak_shape_check,peak_shape_residual,autonomous_shape_check
from pwl_nest.peak_timing import absolute_peak_time_residual


class PeakShapeTest(unittest.TestCase):
    def trace(self,shift=0.,amplitude=1.,width=1.):
        t=np.linspace(0,100,2001)
        c=.04+amplitude*sum(np.exp(-.5*((t-center-shift)/width)**2) for center in [30.,55.,80.])
        return t,np.column_stack([c,np.full(len(t),.8),np.full(len(t),.5)])

    def test_shape_does_not_hide_timing_or_accept_amplitude_width_tradeoff(self):
        t,x=self.trace();_,shifted=self.trace(shift=2.)
        np.testing.assert_allclose(peak_shape_residual(t,shifted,t,x),0,atol=1e-10)
        np.testing.assert_allclose(absolute_peak_time_residual(t,shifted,t,x),2.,atol=1e-10)
        for kwargs in [dict(amplitude=.8),dict(width=1.2)]:
            _,bad=self.trace(**kwargs)
            self.assertFalse(peak_shape_check(t,bad,t,x)['passes'])

    def test_missing_and_extra_peaks_fail_with_fixed_residual_length(self):
        t,x=self.trace();missing=x.copy();missing[:,0]=.04
        self.assertFalse(peak_shape_check(t,missing,t,x)['passes'])
        self.assertEqual(len(peak_shape_residual(t,missing,t,x)),len(peak_shape_residual(t,x,t,x)))
        extra=x.copy();extra[:,0]+=np.exp(-.5*(t-92.)**2)
        self.assertFalse(peak_shape_check(t,extra,t,x)['passes'])

    def test_tiny_autonomous_oscillations_cannot_replace_full_spikes(self):
        t=np.linspace(0,1600,8001);c=.04+.5*(1+np.sin(2*np.pi*t/30))
        truth=np.column_stack([c,np.full(len(t),.8),np.full(len(t),.53)])
        tiny=truth.copy();tiny[:,0]=.04+.001*(1+np.sin(2*np.pi*t/6.4))
        check=autonomous_shape_check(t,tiny,t,truth,cycling=True)
        self.assertFalse(check['passes'])
        self.assertGreater(abs(check['relative_errors'][0]),.99)
        self.assertTrue(autonomous_shape_check(t,truth,t,truth,cycling=True)['passes'])
