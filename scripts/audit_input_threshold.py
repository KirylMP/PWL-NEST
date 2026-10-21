"""Audit actual fitted coefficients: regional affine flow and I crossings."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.strict_pwa import regional_flow


def audit(model):
    limits = np.r_[model.coordinate.forward(.001), model.latent.thresholds, model.coordinate.forward(1.5)]
    input_limits = np.r_[0., model.input_thresholds, 1.3]
    maximum = 0.;samples = 0
    for u in (limits[:-1] + limits[1:]) / 2:
        for ip3 in (input_limits[:-1] + input_limits[1:]) / 2:
            y = np.array([u, float(model.gate.forward(.8)), ip3])
            a, b, region = model.region(y, 20., .45)
            duration = 1e-5
            exact = regional_flow(a, b, y, duration)
            while model.region(exact, 20., .45)[2] != region:
                duration *= .1;exact = regional_flow(a, b, y, duration)
                if duration < 1e-14:
                    raise ValueError('Cannot find an interior regional sample')
            numerical = solve_ivp(model.rhs(20., .45), (0., duration), y,
                                  rtol=1e-11, atol=1e-13, max_step=duration).y[:, -1]
            maximum = max(maximum, float(abs(exact - numerical).max()))
            samples += 1
    continuity = 0.
    if len(model.input_thresholds):
        local = model.geometry_coefficients()
        for s, eta in enumerate(model.input_thresholds):
            np.testing.assert_array_equal(local[s, :, :, 1:3], local[s + 1, :, :, 1:3])
            for j, u in enumerate((limits[:-1] + limits[1:]) / 2):
                state = np.array([1., u, float(model.gate.forward(.8)), eta])
                continuity = max(continuity, float(abs((local[s, j] - local[s + 1, j]) @ state).max()))
    crossings = model.input_crossing_times(1.15, 20., .45, 200.)
    crossing_error = max([abs(.45 + .7 * np.exp(-t / 20.) - eta) for t, eta in crossings] + [0.])
    assert maximum < 2e-9 and continuity < 1e-10 and crossing_error < 1e-13
    return dict(regional_samples=samples, affine_expm_vs_DOP_max_error=maximum,
                input_hinge_continuity_max_error=continuity, crossing_times=crossings,
                exponential_crossing_max_error=float(crossing_error),
                planar_C_h_jacobian_unchanged_across_I_knot=True,
                scope='Interior regional affine propagation and analytic I thresholds; not a complete first-exit solver')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('output')
    out = Path(parser.parse_args().output).resolve()
    rows = json.loads((out / 'summary.json').read_text())['selected']
    result = {}
    for row in rows:
        path = Path(row['model']) if row['family'] == 'original' else out / row['model']
        result[row['family']] = audit(CalciumTransformPWA.load(path))
    (out / 'analytic_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
