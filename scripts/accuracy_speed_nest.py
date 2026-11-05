from accuracy_speed_core import np, KER, parameters

def diagnostic(case, kind, data, I, dt_ms, interval_ms, duration_s):
    import nest
    from pwl_nest.nest_adapter import install
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': dt_ms, 'local_num_threads': 1, 'print_time': False})
    install()
    C, q = map(float, data['rest_target'])
    if kind == 'PWA':
        name = 'astrocyte_lr_pwa'; params = parameters(data, I, C, q)
    else:
        name = 'astrocyte_lr_1994'
        params = dict(Ca_astro=C, h_IP3R=q, IP3=I, IP3_0=I, tau_IP3=20000.,
                      delta_IP3=0., Km_SERCA=KER[case])
    node = nest.Create(name, params=params)
    meter = nest.Create('multimeter', params={'interval': interval_ms,
                         'record_from': ['Ca_astro', 'h_IP3R', 'IP3']})
    nest.Connect(meter, node)
    nest.Simulate(duration_s*1000.)
    ev = meter.get('events')
    t = np.asarray(ev['times'], float)/1000.
    values = np.array([ev['Ca_astro'], ev['h_IP3R']], float)
    if len(t) == 0 or not np.all(np.diff(t)>0) or not np.isfinite(values).all():
        raise RuntimeError('empty, non-monotone or non-finite diagnostic')
    if t[-1] > duration_s + 1e-9:
        raise RuntimeError('diagnostic sample exceeds simulation duration')
    return t, values, float(np.max(np.abs(np.asarray(ev['IP3'])-I)))


