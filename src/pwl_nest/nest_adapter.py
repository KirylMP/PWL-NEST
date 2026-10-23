"""NEST 3.10 C++ astrocyte adapter; physical protocol times are in seconds."""
import hashlib
from pathlib import Path

import numpy as np

from .calcium_coordinates import CalciumTransformPWA
from .models import ROOT

MODELS = {
    'AM': ROOT / 'models/server_candidates/AM_sqrt_tails__cube.json',
    'FM': ROOT / 'models/phase_shape_candidates/FM_width_soft_candidate.json',
}


def install():
    import nest
    if 'astrocyte_lr_pwa' not in nest.node_models:
        nest.Install(str(ROOT / '.build/pwl_astro_module'))


def load_candidate(case, path=None):
    path = Path(path or MODELS[case])
    model = CalciumTransformPWA.load(path)
    if model.case != case:
        raise ValueError('Case/model mismatch')
    return model, dict(path=str(path.relative_to(ROOT)),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def pwa_parameters(model, initial, tau, baseline):
    if np.any(model.threshold_slopes) or model.has_state_mix:
        raise ValueError('NEST adapter requires vertical thresholds without state mixing')
    c_kinds = {'identity': 0, 'sqrt_tails': 1}
    h_kinds = {'identity': 0, 'cube': 1, 'log': 2}
    y = model.to_latent(np.asarray(initial, dtype=float))
    return dict(thresholds=model.latent.thresholds.tolist(),
                coefficients=model.geometry_coefficients().ravel().tolist(),
                input_thresholds=model.input_thresholds.tolist(),
                c_coordinate=c_kinds[model.coordinate.kind],
                h_coordinate=h_kinds[model.h_kind],
                c_reference=model.coordinate.reference, c_scale=model.coordinate.scale,
                h_reference=model.h_reference, u=float(y[0]), v=float(y[1]), IP3=float(y[2]),
                IP3_0=float(baseline), tau_IP3=float(tau*1000), delta_IP3=1.)


def lr_parameters(case, initial, tau, baseline):
    return dict(Ca_astro=float(initial[0]), h_IP3R=float(initial[1]), IP3=float(initial[2]),
                IP3_0=float(baseline), tau_IP3=float(tau*1000), delta_IP3=1.,
                Km_SERCA=.1 if case == 'AM' else .05)


def sic(calcium, scale=1., threshold=.19669):
    arg=(np.asarray(calcium)-threshold)*1000.
    return scale*np.log(np.maximum(arg, 1.))


def build_network(case, model, initial, protocol, kind='pwa', population=1,
                  resolution_ms=10., record_interval_ms=50., record=True,
                  with_targets=True, threads=1, delay_ms=None):
    """Build spike generators → astrocytes → SIC synapses → AEIF targets.

    A jump at protocol t is delivered at the END of the NEST bin ending at t.
    Generators are shifted back by the explicit synaptic delay to match this.
    Each source event carries delta-IP3 as its nonnegative static synapse weight.
    """
    import nest
    nest.ResetKernel()
    nest.verbosity=nest.VerbosityLevel.ERROR
    nest.SetKernelStatus(dict(resolution=resolution_ms, local_num_threads=threads,
                              rng_seed=20261021))
    if kind == 'pwa':
        install()
        params=pwa_parameters(model, initial, protocol['tau'], protocol['baseline'])
        name='astrocyte_lr_pwa'
    elif kind == 'lr':
        params=lr_parameters(case, initial, protocol['tau'], protocol['baseline'])
        name='astrocyte_lr_1994'
    else:
        raise ValueError(kind)
    astro=nest.Create(name, population, params=params)
    delay=resolution_ms if delay_ms is None else delay_ms
    if delay<resolution_ms or not np.isclose(delay/resolution_ms,round(delay/resolution_ms)):
        raise ValueError('Synaptic delay must be a positive integer number of ticks')
    # One deterministic source per jump, broadcast to every member of the population.
    for t, delta in protocol.get('jumps', ()):
        when=t*1000.-delay
        if delta < 0 or when <= 0 or not np.isclose(when/resolution_ms, round(when/resolution_ms)):
            raise ValueError('Jumps must be positive, on the NEST grid and later than one delay')
        source=nest.Create('spike_generator', params={'spike_times': [when]})
        nest.Connect(source, astro, syn_spec=dict(weight=float(delta), delay=delay))
    targets=None
    if with_targets:
        targets=nest.Create('aeif_cond_alpha_astro', population)
        nest.Connect(astro, targets, 'one_to_one',
                     syn_spec=dict(synapse_model='sic_connection', weight=1., delay=delay))
    meters=[]
    if record:
        for nodes, fields in [(astro, ['Ca_astro', 'h_IP3R', 'IP3']),
                              (targets, ['I_SIC', 'V_m'])]:
            if nodes is None:
                continue
            meter=nest.Create('multimeter', params=dict(record_from=fields,
                                                       interval=record_interval_ms))
            nest.Connect(meter, nodes[:1])
            meters.append(meter)
    return astro, targets, meters
