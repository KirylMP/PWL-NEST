import unittest
import numpy as np
from pwl_nest.peak_timing import interpolated_peak_times, absolute_peak_time_residual


def pulses(t, centers):
    c = .05 + sum(.5 * np.exp(-((t - center) / 1.3) ** 2) for center in centers)
    return np.column_stack([c, np.full(len(t), .8), np.full(len(t), .5)])


class PeakTimingTest(unittest.TestCase):
    def test_subsample_interpolation_recovers_peak_times(self):
        t = np.linspace(0, 90, 361)
        centers = np.array([30.07, 50.13, 70.21])
        np.testing.assert_allclose(interpolated_peak_times(t, pulses(t, centers)),
                                   centers - 20., atol=.003)

    def test_cumulative_drift_is_penalized_for_every_peak(self):
        t = np.linspace(0, 90, 1801)
        centers = np.array([30., 50., 70.])
        truth = pulses(t, centers)
        drifting = pulses(t, centers + [0., 1., 2.])
        residual = absolute_peak_time_residual(t, drifting, t, truth)
        np.testing.assert_allclose(residual, [0., 1., 2.], atol=1e-8)
        self.assertAlmostEqual(float(residual @ residual), 5.)
        np.testing.assert_allclose(absolute_peak_time_residual(t, truth, t, truth), 0.)

    def test_missing_peaks_preserve_residual_length(self):
        t = np.linspace(0, 90, 901)
        truth = pulses(t, [30., 50., 70.])
        missing = pulses(t, [30., 50.])
        np.testing.assert_allclose(absolute_peak_time_residual(t, missing, t, truth),
                                   [0., 0., 25.], atol=1e-8)

    def test_peak_times_are_consistent_across_sampling_grids(self):
        coarse = np.linspace(0, 90, 361)
        fine = np.linspace(0, 90, 1801)
        centers = np.array([30.07, 50.13, 70.21])
        residual = absolute_peak_time_residual(coarse, pulses(coarse, centers + .1),
                                               fine, pulses(fine, centers))
        np.testing.assert_allclose(residual, .1, atol=.003)


if __name__ == '__main__':
    unittest.main()
