"""Compute and plot Figure 1 directly from the repository's nonlinear Li--Rinzel.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/paper_figure1.py
Replot cached data: .venv/bin/python scripts/paper_figure1.py --plot-only

Periodic orbits use collocation and pseudo-arclength continuation, not a sweep
of transient extrema. The nontrivial planar Floquet multiplier is obtained from
the divergence integral. An independent IVP checks each displayed orbit.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy
from scipy.integrate import cumulative_trapezoid, simpson, solve_bvp, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks

from pwl_nest.models import lr_rhs
from pwl_nest.strict_pwa import C_MAX, equilibrium_h, lr_field, spectral_events

OUT = ROOT / 'paper/figures'
DATA = OUT / 'figure1_data'
BVP_TOL = 1e-8
CLOSURE_TOL = 2e-6
PHASE = np.linspace(0, 1, 2001)
COLORS = dict(calcium='#e88719', gate='#26854b', stable='#265f98',
              unstable='#c43b40', maximum='#151515', minimum='#39a65a')
PARAMETERS = dict(C0=2., c1=.185, rC=6., rL=.11, vER=.9, d1=.13,
                  d2=1.049, d3=.9434, d5=.08234, a2=.2,
                  KER_AM=.1, KER_FM=.05)


def field(case, y, ip3):
    return lr_field(np.array([y[0], y[1], np.full_like(y[0], ip3)]).T, case).T


def jacobian(case, y, ip3):
    """Analytic state Jacobian, scalar or vectorized over its last axis."""
    c, h = y
    ker = PARAMETERS['KER_' + case]
    q = 1.049 * (ip3 + .13) / (ip3 + .9434)
    gain = 6 * (ip3 / (ip3 + .13))**3
    n, drive = c / (c + .08234), 2 - 1.185 * c
    fc = 3 * gain * n**2 * .08234 / (c + .08234)**2 * h**3 * drive
    fc -= 1.185 * (gain * n**3 * h**3 + .11)
    fc -= 1.8 * c * ker**2 / (c**2 + ker**2)**2
    fh = 3 * gain * n**3 * h**2 * drive
    return np.array([[fc, fh], [-.2 * h, -.2 * (q + c)]])


def trace(case, y, ip3):
    jac = jacobian(case, y, ip3)
    return jac[0, 0] + jac[1, 1]


def seed_cycle(case, hopf, amplitude):
    """Leave a Hopf point using its eigenvector; fix C at its maximum."""
    center = np.array([hopf['C'], hopf['h']])
    ip3 = hopf['I']
    values, vectors = np.linalg.eig(jacobian(case, center, ip3))
    idx = np.argmax(values.imag)
    vector = vectors[:, idx] / vectors[0, idx]
    mesh = np.linspace(0, 1, 401)
    orbit = center[:, None] + amplitude * (
        vector.real[:, None] * np.cos(2 * np.pi * mesh)
        - vector.imag[:, None] * np.sin(2 * np.pi * mesh))

    def bc(ya, yb, p):
        return np.r_[ya - yb, field(case, ya, p[1])[0],
                     ya[0] - center[0] - amplitude]

    sol = solve_bvp(lambda s, y, p: p[0] * field(case, y, p[1]), bc,
                    mesh, orbit, p=[2 * np.pi / values[idx].imag, ip3],
                    tol=BVP_TOL, max_nodes=24000)
    if not sol.success or np.ptp(sol.y[0]) < amplitude:
        raise RuntimeError(f'{case} Hopf seed failed: {sol.message}')
    return sol


def continue_cycles(case, hopf, lower=False):
    """Full-profile arclength handles both folds and the FM canard segment.

    The metric is integral |dy|^2 ds + (dT/30)^2 + dI^2. An auxiliary
    integral state imposes the arclength condition. Phase is fixed at Cmax.
    """
    previous = seed_cycle(case, hopf, .001)
    current = seed_cycle(case, hopf, .002)
    solutions = [previous, current]
    mesh = np.linspace(0, 1, 701)
    step, maximum_step = .006, (.022 if lower else .016)
    for iteration in range(700):
        difference = current.sol(mesh)[:2] - previous.sol(mesh)[:2]
        dp = current.p - previous.p
        distance = np.sqrt(simpson(np.sum(difference**2, axis=0), x=mesh)
                           + (dp[0] / 30)**2 + dp[1]**2)
        tangent_y, tangent_p = difference / distance, dp / distance
        last = CubicSpline(mesh, current.sol(mesh)[:2].T)
        tangent = CubicSpline(mesh, tangent_y.T)
        predicted_p = current.p + step * tangent_p

        def fun(s, y, p):
            ty = tangent(s).T
            predicted_y = last(s).T + step * ty
            return np.vstack([p[0] * field(case, y[:2], p[1]),
                              np.sum((y[:2] - predicted_y) * ty, axis=0)])

        def bc(ya, yb, p):
            arc = yb[2] + (p[0] - predicted_p[0]) * tangent_p[0] / 900
            arc += (p[1] - predicted_p[1]) * tangent_p[1]
            return np.r_[ya[:2] - yb[:2], field(case, ya[:2], p[1])[0],
                         ya[2], arc]

        guess = np.vstack([current.sol(mesh)[:2] + step * tangent_y,
                           np.zeros_like(mesh)])
        sol = solve_bvp(fun, bc, mesh, guess, p=predicted_p,
                        tol=BVP_TOL, max_nodes=24000)
        valid = (sol.success and 0 < sol.p[0] < 150 and 0 < sol.p[1] < 1.3
                 and np.ptp(sol.y[0]) > 1e-5)
        if not valid:
            step *= .5
            if step < 2e-6:
                raise RuntimeError(f'{case}: continuation stalled: {sol.message}')
            continue
        solutions.append(sol)
        previous, current = current, sol
        if iteration % 30 == 0:
            print(f'{case} {"lower" if lower else "upper"}: '
                  f'I={sol.p[1]:.9f}, T={sol.p[0]:.4f} s', flush=True)
        if case == 'AM' and sol.p[1] < .355 and sol.y[0].max() < .17:
            break
        # The independently resolved part is selected by IVP validation later.
        if case == 'FM' and sol.p[0] >= 65:
            break
        step = min(step * 1.12, maximum_step)
    else:
        raise RuntimeError(f'{case}: continuation iteration limit')
    return solutions


def cycle_metrics(case, sol):
    ip3, period = float(sol.p[1]), float(sol.p[0])
    dense_phase = np.linspace(0, 1, 12001)
    y = sol.sol(dense_phase)[:2]
    imin = np.argmin(y[0])
    bounds = (dense_phase[max(0, imin - 2)],
              dense_phase[min(len(dense_phase) - 1, imin + 2)])
    minimum = minimize_scalar(lambda s: sol.sol(s)[0], bounds=bounds,
                              method='bounded', options={'xatol': 1e-14}).fun
    log_multiplier = period * simpson(trace(case, y, ip3), x=dense_phase)
    orbit = sol.sol(PHASE)[:2]
    amp = np.ptp(orbit[0])
    repeated = np.tile(orbit[0, :-1], 3)
    peaks = find_peaks(repeated, prominence=amp * 1e-4)[0]
    number_peaks = np.count_nonzero((peaks >= len(PHASE) - 1)
                                   & (peaks < 2 * (len(PHASE) - 1)))
    # Reverse-time integration attracts to an unstable planar cycle.
    direction = 1 if log_multiplier < 0 else -1
    ivp = solve_ivp(lr_rhs(case, ip3), (0, direction * period), orbit[:, 0],
                    method='DOP853', rtol=2e-11, atol=2e-13,
                    max_step=.06, dense_output=True)
    closure = float(np.max(np.abs(ivp.y[:, -1] - orbit[:, 0])))
    ivp_orbit = ivp.sol(direction * period * PHASE)
    waveform_error = float(np.max(np.abs(ivp_orbit - sol.sol(
        PHASE if direction == 1 else 1 - PHASE)[:2])))
    physical = (y[0].min() >= 0 and y[0].max() <= C_MAX
                and y[1].min() >= 0 and y[1].max() <= 1)
    if not physical or number_peaks != 1:
        raise RuntimeError('Nonphysical or nonprimitive cycle encountered')
    resolved = bool(ivp.success and closure < CLOSURE_TOL
                    and waveform_error < 5 * CLOSURE_TOL)
    row = dict(I=ip3, period_s=period, C_min=float(minimum),
               C_max=float(orbit[0, 0]), log_floquet_multiplier=float(log_multiplier),
               stable=bool(log_multiplier < 0), ivp_closure=closure,
               ivp_waveform_error=waveform_error, displayed=resolved,
               bvp_residual=float(max(sol.rms_residuals)), nodes=len(sol.x))
    return row, orbit


def refine_cycle_fold(case, solutions, rows):
    logs = np.array([r['log_floquet_multiplier'] for r in rows])
    crossings = np.flatnonzero(logs[:-1] * logs[1:] < 0)
    if len(crossings) != 1:
        raise RuntimeError(f'Expected one upper cycle fold for {case}')
    k = crossings[0]
    seed = solutions[k if abs(logs[k]) < abs(logs[k + 1]) else k + 1]
    mesh = np.linspace(0, 1, 1001)
    y = seed.sol(mesh)[:2]
    integral = cumulative_trapezoid(seed.p[0] * trace(case, y, seed.p[1]),
                                   mesh, initial=0)

    def fun(s, y, p):
        return np.vstack([p[0] * field(case, y[:2], p[1]),
                          p[0] * trace(case, y[:2], p[1])])

    def bc(ya, yb, p):
        return np.r_[ya[:2] - yb[:2], field(case, ya[:2], p[1])[0], ya[2], yb[2]]

    sol = solve_bvp(fun, bc, mesh, np.vstack([y, integral]), p=seed.p,
                    tol=2e-9, max_nodes=30000)
    if not sol.success:
        raise RuntimeError(f'{case} cycle fold correction failed')
    row, orbit = cycle_metrics(case, sol)
    if abs(row['log_floquet_multiplier']) > 2e-5 or not row['displayed']:
        raise RuntimeError(f'{case} cycle fold verification failed: {row}')
    return dict(kind='cycle_fold', case=case, **row), orbit


def equilibria(case):
    force = lambda c, ip3: lr_field([c, equilibrium_h(c, ip3), ip3], case)[0]
    low = brentq(lambda c: force(c, 0), 1e-8, C_MAX)
    high = brentq(lambda c: force(c, 1.3), 1e-8, C_MAX)
    events = spectral_events(case)
    calcium = np.unique(np.r_[np.linspace(low + 1e-10, high - 1e-10, 1801),
                              [e['C'] for e in events]])
    inputs, gates, stable, saddle = [], [], [], []
    for c in calcium:
        ip3 = brentq(lambda i: force(c, i), 0, 1.3, xtol=2e-14)
        h = equilibrium_h(c, ip3)
        eig = np.linalg.eigvals(jacobian(case, [c, h], ip3))
        inputs.append(ip3)
        gates.append(h)
        stable.append(max(eig.real) < 0)
        saddle.append(np.linalg.det(jacobian(case, [c, h], ip3)) < 0)
    return dict(I=np.array(inputs), C=calcium, h=np.array(gates),
                stable=np.array(stable), saddle=np.array(saddle)), events


def nullclines(case, ip3, calcium):
    ker = PARAMETERS['KER_' + case]
    gain = 6 * (ip3 / (ip3 + .13))**3 * (calcium / (calcium + .08234))**3
    calcium_nullcline = np.cbrt((.9 * calcium**2 / (calcium**2 + ker**2)
                                / (2 - 1.185 * calcium) - .11) / gain)
    return calcium_nullcline, equilibrium_h(calcium, ip3)


def compute():
    DATA.mkdir(parents=True, exist_ok=True)
    metadata = dict(parameters=PARAMETERS, bvp_tolerance=BVP_TOL,
                    ivp_closure_tolerance=CLOSURE_TOL,
                    versions=dict(numpy=np.__version__, scipy=scipy.__version__,
                                  matplotlib=matplotlib.__version__),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    reference_figure_doi='10.1007/s10867-009-9155-y',
                    events=[], validation={})
    all_rows = []
    for case in ['AM', 'FM']:
        eq, spectral = equilibria(case)
        np.savez_compressed(DATA / f'{case}_equilibria.npz', **eq)
        hopfs = sorted([e for e in spectral if e['kind'] == 'trace_zero'], key=lambda e: e['I'])
        metadata['events'].extend([dict(case=case, **e) for e in spectral
                                   if e['kind'] in ('trace_zero', 'fold')])
        branches = {'upper': continue_cycles(case, hopfs[-1])}
        if case == 'AM':
            last_max = branches['upper'][-1].y[0, 0]
            branches['upper'].extend(seed_cycle(case, hopfs[0], a)
                                     for a in [.012, .008, .004, .002, .001, .0002]
                                     if hopfs[0]['C'] + a < last_max)
        else:
            branches['lower'] = continue_cycles(case, hopfs[0], lower=True)
        for name, solutions in branches.items():
            rows, orbits = zip(*(cycle_metrics(case, s) for s in solutions))
            if name == 'upper':
                event, orbit = refine_cycle_fold(case, solutions, rows)
                metadata['events'].append(event)
                np.savez_compressed(DATA / f'{case}_cycle_fold.npz', orbit=orbit,
                                    phase=PHASE)
            np.savez_compressed(DATA / f'{case}_{name}_cycles.npz',
                                phase=PHASE, orbits=np.array(orbits))
            all_rows.extend(dict(case=case, branch=name, index=k, **row)
                            for k, row in enumerate(rows))
            displayed = [r for r in rows if r['displayed']]
            if len(displayed) < .65 * len(rows):
                raise RuntimeError('Too many unresolved periodic solutions')
            metadata['validation'][f'{case}_{name}'] = dict(
                computed=len(rows), displayed=len(displayed),
                max_displayed_closure=max(r['ivp_closure'] for r in displayed),
                max_displayed_waveform_error=max(r['ivp_waveform_error'] for r in displayed),
                max_bvp_residual=max(r['bvp_residual'] for r in rows),
                displayed_period_range_s=[min(r['period_s'] for r in displayed),
                                          max(r['period_s'] for r in displayed)])
        # Independent algebraic checks of equilibria, nullclines, and Jacobian.
        residual = np.max(np.abs(lr_field(np.column_stack([eq['C'], eq['h'], eq['I']]), case)))
        c = np.linspace(.005, 1.2, 4001)
        ip3 = .4 if case == 'AM' else .6
        hc, hh = nullclines(case, ip3, c)
        physical = (hc > 0) & (hc < 1)
        calcium_error = np.max(np.abs(field(case, np.array([c[physical], hc[physical]]), ip3)[0]))
        gate_error = np.max(np.abs(field(case, np.array([c, hh]), ip3)[1]))
        y = np.array([.2, .7])
        numerical = np.column_stack([
            field(case, y.astype(complex) + 1e-25j * np.eye(2)[k], ip3).imag / 1e-25
            for k in range(2)])
        jac_error = np.max(np.abs(numerical - jacobian(case, y, ip3)))
        assert residual < 1e-10 and calcium_error < 1e-10 and gate_error < 1e-10 and jac_error < 1e-10
        metadata['validation'][case + '_algebra'] = dict(equilibrium_residual=float(residual),
            calcium_nullcline_residual=float(calcium_error), gate_nullcline_residual=float(gate_error),
            jacobian_complex_step_error=float(jac_error))
        # A full cycle at the phase-plane input, obtained independently by IVP.
        rhs = lr_rhs(case, ip3)
        transient = solve_ivp(rhs, (0, 900), [.6 if case == 'AM' else 1., .8],
                              method='DOP853', rtol=2e-11, atol=2e-13, max_step=.15)
        def peak(t, state):
            return rhs(t, state)[0]
        peak.direction = -1
        sol = solve_ivp(rhs, (0, 150), transient.y[:, -1], events=peak,
                        method='DOP853', rtol=2e-11, atol=2e-13,
                        max_step=.06, dense_output=True)
        t0, t1 = sol.t_events[0][-2:]
        phase_orbit = sol.sol(t0 + PHASE * (t1 - t0))
        np.savez_compressed(DATA / f'{case}_phase_plane.npz', phase=PHASE,
                            orbit=phase_orbit, I=ip3, period_s=t1 - t0)
    with (DATA / 'cycles.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (DATA / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata['validation'], indent=2), flush=True)


def load_rows():
    with (DATA / 'cycles.csv').open() as stream:
        result = list(csv.DictReader(stream))
    for row in result:
        for key in ('I', 'period_s', 'C_min', 'C_max', 'log_floquet_multiplier'):
            row[key] = float(row[key])
        for key in ('stable', 'displayed'):
            row[key] = row[key] == 'True'
    return result


def phase_panel(ax, case):
    data = np.load(DATA / f'{case}_phase_plane.npz')
    ip3, orbit = float(data['I']), data['orbit']
    xmax = .8 if case == 'AM' else 1.15
    c = np.linspace(.0005, xmax, 5000)
    hc, hh = nullclines(case, ip3, c)
    cx, hy = np.meshgrid(np.linspace(.025, xmax - .015, 19), np.linspace(.42, .98, 17))
    u, v = field(case, np.array([cx, hy]), ip3)
    length = np.hypot(u / xmax, v / .6)
    ax.quiver(cx, hy, u / length, v / length, angles='xy', scale_units='xy',
              scale=27, color='.68', width=.003, headwidth=3.5, zorder=0)
    physical = (hc >= 0) & (hc <= 1.05)
    ax.plot(c[physical], hc[physical], color=COLORS['calcium'], lw=1.4)
    ax.plot(c, hh, color=COLORS['gate'], lw=1.4)
    ax.plot(orbit[0], orbit[1], color=COLORS['maximum'], lw=1.0)
    # Place arrows by arc length in axes coordinates, avoiding the slow portion.
    distance = np.r_[0, np.cumsum(np.hypot(np.diff(orbit[0]) / xmax,
                                          np.diff(orbit[1]) / .6))]
    for fraction in [.2, .7]:
        k = np.searchsorted(distance, fraction * distance[-1])
        k0, k1 = max(0, k - 10), min(len(PHASE) - 1, k + 10)
        ax.annotate('', xy=orbit[:, k1], xytext=orbit[:, k0],
                    arrowprops=dict(arrowstyle='-|>', color=COLORS['maximum'], lw=.8, mutation_scale=7))
    ax.text(.97, .96, rf'$I={ip3:.1f}\,\mu\mathrm{{M}}$', transform=ax.transAxes,
            ha='right', va='top', fontsize=7,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    ax.set(xlim=(0, xmax), ylim=(.4, 1), xlabel=r'$[\mathrm{Ca}^{2+}]$ ($\mu$M)',
           ylabel=r'IP$_3$R availability, $h$')
    ax.set_yticks([.4, .6, .8, 1.])


def plot():
    rows = load_rows()
    metadata = json.loads((DATA / 'metadata.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8.5, 'axes.linewidth': .75, 'xtick.direction': 'in',
        'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 600})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    fig.subplots_adjust(left=.09, right=.985, bottom=.17, top=.89,
                        wspace=.47, hspace=.63)
    for n, case in enumerate(['AM', 'FM']):
        phase_panel(axes[n, 0], case)
        ax, period_ax = axes[n, 1:]
        eq = np.load(DATA / f'{case}_equilibria.npz')
        for stable, color, style in [(True, COLORS['stable'], '-'),
                                     (False, COLORS['unstable'], '--')]:
            ax.plot(eq['I'], np.where(eq['stable'] == stable, eq['C'], np.nan),
                    color=color, ls=style, lw=1.3)
        branches = ['upper'] if case == 'AM' else ['upper', 'lower']
        for branch in branches:
            selected = [r for r in rows if r['case'] == case and r['branch'] == branch and r['displayed']]
            for stable in [True, False]:
                part = [r for r in selected if r['stable'] == stable]
                # Thin markers are actual computed solutions, never transient peaks.
                for key, color in [('C_max', COLORS['maximum']), ('C_min', COLORS['minimum'])]:
                    ax.scatter([r['I'] for r in part], [r[key] for r in part], s=8,
                               facecolors=color if stable else 'white',
                               edgecolors=color, linewidths=.7, zorder=3)
                period_ax.scatter([r['I'] for r in part], [r['period_s'] for r in part],
                                  s=8, facecolors=COLORS['maximum'] if stable else 'white',
                                  edgecolors=COLORS['maximum'], linewidths=.7, zorder=3)
        for event in metadata['events']:
            if event['case'] != case or event['kind'] not in ('trace_zero', 'cycle_fold'):
                continue
            if event['kind'] == 'trace_zero':
                ax.scatter(event['I'], event['C'], marker='s', s=17,
                           facecolors='white', edgecolors=COLORS['unstable'], linewidths=.9, zorder=5)
            else:
                for key, color in [('C_max', COLORS['maximum']), ('C_min', COLORS['minimum'])]:
                    ax.scatter(event['I'], event[key], marker='D', s=15,
                               facecolors=color, edgecolors='white', linewidths=.3, zorder=5)
                period_ax.scatter(event['I'], event['period_s'], marker='D', s=15,
                                  color=COLORS['maximum'], zorder=5)
        xlim = (.0, .8) if case == 'AM' else (.38, 1.2)
        ax.set(xlim=xlim, ylim=(0, .7 if case == 'AM' else 1.2),
               xlabel=r'$[\mathrm{IP}_3]$ ($\mu$M)', ylabel=r'$[\mathrm{Ca}^{2+}]$ ($\mu$M)')
        period_ax.set(xlim=(.34, .66) if case == 'AM' else (.5, 1.16),
                      ylim=(9, 14) if case == 'AM' else (8, 65),
                      xlabel=r'$[\mathrm{IP}_3]$ ($\mu$M)', ylabel='Period (s)')
        if case == 'FM':
            ax.set_xticks([.4, .6, .8, 1., 1.2])
            period_ax.set_xticks([.5, .7, .9, 1.1])
            period_ax.set_yticks([10, 20, 30, 40, 50, 60])
    for ax, letter in zip(axes.flat, 'abcdef'):
        ax.text(-.25, 1.055, letter, fontsize=10.5, fontweight='bold', transform=ax.transAxes)
    for row, case in enumerate(['AM', 'FM']):
        axes[row, 0].set_title(rf'{case}: $K_{{ER}}={PARAMETERS["KER_" + case]:.2f}\,\mu$M',
                               loc='left', fontsize=8.5, pad=11)
    for col, title in enumerate(['Phase plane', 'Equilibria and limit cycles', 'Oscillation period']):
        fig.text((.21, .545, .87)[col], .973, title, ha='center', fontsize=9)
    handles = [Line2D([], [], color=COLORS['calcium'], lw=1.8, label=r'$\dot C=0$'),
               Line2D([], [], color=COLORS['gate'], lw=1.8, label=r'$\dot h=0$'),
               Line2D([], [], color=COLORS['stable'], lw=1.6, label='Stable equilibrium'),
               Line2D([], [], color=COLORS['unstable'], lw=1.6, ls='--', label='Unstable equilibrium'),
               Line2D([], [], marker='o', ls='', color=COLORS['maximum'], ms=4, label='Stable cycle'),
               Line2D([], [], marker='o', ls='', color=COLORS['maximum'], mfc='white', ms=4, label='Unstable cycle')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .005),
               ncol=3, frameon=False, fontsize=7, columnspacing=1.4)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'figure1_li_rinzel.{suffix}', dpi=600)
    plt.close(fig)
    metadata['plot_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (DATA / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Saved paper/figures/figure1_li_rinzel.{pdf,svg,png}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if not args.plot_only:
        compute()
    plot()
