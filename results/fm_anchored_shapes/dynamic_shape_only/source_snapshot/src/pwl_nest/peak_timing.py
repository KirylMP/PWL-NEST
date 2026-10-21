"""Absolute peak-time residuals, with sub-sample peak interpolation."""
import numpy as np
from scipy.signal import find_peaks


def interpolated_peak_times(t, x, onset=20., prominence=.02):
    """Locate sampled peaks, then interpolate their local quadratic vertices.

    The interpolation uses actual times, including unequal event sample spacing.
    It improves the continuity of the timing objective as parameters vary; it
    does not shift trajectories or match peaks independently across signals.
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(x)[:, 0]
    if t.ndim != 1 or len(t) != len(c) or np.any(np.diff(t) <= 0):
        raise ValueError('Peak times require a strictly increasing time grid')
    indices = find_peaks(c, prominence=prominence)[0]
    times = []
    for i in indices:
        left, right = t[i - 1] - t[i], t[i + 1] - t[i]
        slope_left = (c[i - 1] - c[i]) / left
        slope_right = (c[i + 1] - c[i]) / right
        curvature = (slope_right - slope_left) / (right - left)
        slope = slope_right - curvature * right
        offset = -slope / (2 * curvature) if curvature < 0 else 0.
        if not left <= offset <= right:
            offset = 0.
        peak_time = t[i] + offset
        if peak_time >= onset:
            times.append(peak_time - onset)
    return np.asarray(times)


def absolute_peak_time_residual(t, x, target_t, target_x, onset=20., sigma_s=1.):
    """Fixed-length residual of input-locked peak times in chronological order.

    Each reference peak has one residual; absent peaks contribute 25. Extra
    peaks are penalized by the existing event-count criterion in the caller.
    """
    if not np.isfinite(sigma_s) or sigma_s <= 0:
        raise ValueError('Peak timing sigma must be positive')
    reference = interpolated_peak_times(target_t, target_x, onset)
    candidate = interpolated_peak_times(t, x, onset)
    paired = min(len(reference), len(candidate))
    residual = np.full(len(reference), 25.)
    residual[:paired] = (candidate[:paired] - reference[:paired]) / sigma_s
    return residual


def burst_end_residual(t, x, target_t, target_x, onset=20., sigma_s=1.):
    """Penalize last-peak timing and excess calcium after the reference burst.

    The tail mask is fixed by the reference, starting five seconds after its
    last peak. A 0.02 uM tolerance avoids treating tiny tail differences as
    full impulses. The original peak-count criterion remains unchanged.
    """
    if not np.isfinite(sigma_s) or sigma_s <= 0:
        raise ValueError('Burst timing sigma must be positive')
    reference = interpolated_peak_times(target_t, target_x, onset)
    candidate = interpolated_peak_times(t, x, onset)
    last = reference[-1] if len(reference) else 0.
    timing = ((candidate[-1] - last) / sigma_s if len(candidate) else 25.) if len(reference) else 25. * bool(len(candidate))
    mask = np.asarray(target_t) >= onset + last + 5.
    calcium = np.interp(np.asarray(target_t)[mask], t, np.asarray(x)[:, 0])
    tail = 2. * np.maximum(calcium - np.asarray(target_x)[mask, 0] - .02, 0.) / .05 / np.sqrt(max(1, int(mask.sum())))
    return np.r_[timing, tail]
