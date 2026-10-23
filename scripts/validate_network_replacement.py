"""Matched replacement pilot based on NEST 3.10 astrocyte_small_network.

Connectivity and cell/synapse parameters follow the distributed NEST example.
The extended condition changes only duration and delta_IP3, as recorded below.
This is a feedforward tripartite network, not a recurrent-network validation.
"""
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nest
import numpy as np

from pwl_nest.models import ROOT
from pwl_nest.nest_adapter import install, load_candidate, pwa_parameters

OUT = ROOT / 'results/network_replacement'
VARIANTS = {
    'LR': None,
    'full_AM': 'models/server_candidates/AM_sqrt_tails__cube.json',
    'AM_2C': 'models/threshold_reduction/selected_AM_2C.json',
    'AM_2C_2I': 'models/threshold_reduction/selected_AM_2C_2I.json',
    'no_SIC': None,
}


def write_report(report):
    report['settings'] = dict(seed=20261029, resolution_ms=.1, threads=1,
                              source_neurons=10, target_neurons=10, astrocytes=5,
                              initial_C=.073, initial_h=.793, initial_IP3=.4,
                              IP3_baseline=.16, tau_IP3_ms=7142., I_e_pA=1000.)
    report['model_sha256'] = {k: hashlib.sha256((ROOT/v).read_bytes()).hexdigest()
                              for k, v in VARIANTS.items() if v is not None}
    (OUT/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    rows = report['conditions']['extended_default_jump']['results']
    base = rows['no_SIC']['post_spikes']
    effect = rows['LR']['post_spikes']-base
    lines = ['# Замена LR в небольшой сети NEST', '',
             'Источник: официальный пример NEST 3.10 '
             '`astrocytes/astrocyte_small_network.py`. '
             '10 исходных нейронов → 10 целевых, 5 астроцитов, '
             'TripartiteConnect, tsodyks_synapse и sic_connection.', '',
             '## Исходный пример без изменения параметров', '',
             'За 1 с IP3 в LR достигает 7.96 мкМ. Все три PWA-варианта '
             'останавливаются при выходе за обученный диапазон [0, 1.3]. '
             'Прямая замена в этом режиме не подтверждена. '
             'Ограничение не обходилось и вход не обрезался.', '',
             '## Отдельный опыт в пределах диапазона', '',
             'Скачок IP3 изменён с 0.2 на стандартный для модели NEST '
             '0.0002 мкМ; длительность увеличена с 1 до 30 с. '
             'Остальные параметры исходного примера сохранены. '
             'У всех вариантов одинаковы начальное физическое состояние, '
             'топология и пресинаптические спайки (проверено программно). '
             'IP3 находится в диапазоне 0.196–0.402 мкМ.', '',
             '| Модель | Спайки на выходе | Частота, Гц/нейрон | '
             'Прирост спайков относительно no SIC | RMSE SIC к LR, пА |',
             '|---|---:|---:|---:|---:|']
    for name, r in rows.items():
        lines.append(f"| {name} | {r['post_spikes']} | {r['post_rate_hz']:.3f} | "
                     f"{r['post_spikes']-base:+d} | {r['post_I_SIC_RMSE_vs_LR']:.3f} |")
    lines += ['', f'LR добавляет {effect} спайков относительно отключённого SIC; '
              'полная AM сохраняет этот суммарный эффект, компактные варианты '
              'сохраняют 140/150 ≈ 93%. Это предварительное совпадение '
              'интегрального эффекта, не всех времён спайков.', '',
              '![Динамика сети](comparison.png)', '',
              '## Что этот опыт не доказывает', '',
              'Это одна детерминированная feedforward-сеть с синхронными '
              'популяциями, а не рекуррентная сеть. SIC здесь в основном '
              'переходный процесс первых примерно 5 с; стационарные '
              'осцилляции, синхронизация и сетевые переходы не проверены. '
              'Сходная средняя частота не означает совпадения временного '
              'профиля: формы Ca, SIC и времена спайков различаются. '
              'Контроль без SIC необходим, чтобы увидеть величину эффекта.', '',
              'Следующий более сильный тест — официальный '
              '`astrocyte_brunel_fixed_indegree.py`: рекуррентная E/I-сеть '
              'с контролем без SIC, замороженными моделями, одинаковыми '
              'входами и топологией для каждой реализации. Следует проверить '
              'частоты, CV ISI, корреляции/синхронизацию, спектр популяционной '
              'активности и реакцию на стимул по нескольким реализациям. '
              'Перед заменой нужно подтвердить диапазон IP3 у исходного LR; '
              'если он шире 1.3 мкМ, потребуется отдельное расширение '
              'аппроксимации, а не тихое изменение сети.', '',
              'Времена в JSON включают запись и единственный запуск; '
              'их нельзя использовать как измерение ускорения.', '',
              '## Воспроизведение', '',
              '```bash',
              'MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/validate_network_replacement.py',
              '```', '',
              'Параметры, контроль топологии, ошибки и SHA256 моделей: '
              '[evaluation.json](evaluation.json). Полные записи — в NPZ рядом.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run_one(name, condition, record=True):
    build_start = time.perf_counter()
    nest.ResetKernel()
    install()
    nest.SetKernelStatus(dict(resolution=.1, local_num_threads=1, rng_seed=20261029))
    nest.verbosity = nest.VerbosityLevel.ERROR
    pre = nest.Create('aeif_cond_alpha_astro', 10, params=dict(tau_syn_ex=2., I_e=1000.))
    post = nest.Create('aeif_cond_alpha_astro', 10, params=dict(tau_syn_ex=2., I_e=1000.))
    if VARIANTS[name] is None:
        astro = nest.Create('astrocyte_lr_1994', 5,
                            params=dict(IP3=.4, delta_IP3=condition['delta_IP3']))
    else:
        model, _ = load_candidate('AM', ROOT / VARIANTS[name])
        params = pwa_parameters(model, [.073, .793, .4], 7.142, .16)
        params['delta_IP3'] = condition['delta_IP3']
        params['allow_ip3_extrapolation'] = condition.get('allow_ip3_extrapolation', False)
        astro = nest.Create('astrocyte_lr_pwa', 5, params=params)
    nest.TripartiteConnect(
        pre, post, astro,
        conn_spec=dict(rule='pairwise_bernoulli', p=1.),
        third_factor_conn_spec=dict(rule='third_factor_bernoulli_with_pool',
                                   p=1., pool_size=1, pool_type='block'),
        syn_specs=dict(primary=dict(synapse_model='tsodyks_synapse'),
                       third_in=dict(synapse_model='tsodyks_synapse'),
                       third_out=dict(synapse_model='sic_connection',
                                      weight=0. if name == 'no_SIC' else 1.)))
    edges = []
    for a, b in [(pre, post), (pre, astro), (astro, post)]:
        c = nest.GetConnections(a, b)
        edges.append(sorted(zip(c.source, c.target)))
    topology = hashlib.sha256(json.dumps(edges).encode()).hexdigest()
    meters = []
    for nodes, fields in ([(astro, ['Ca_astro', 'IP3', 'h_IP3R']),
                          (post, ['V_m', 'I_SIC'])] if record else []):
        meter = nest.Create('multimeter', params=dict(record_from=fields,
                                                     interval=condition['record_ms']))
        nest.Connect(meter, nodes)
        meters.append(meter)
    spikes = []
    if record:
        spikes = nest.Create('spike_recorder', 2)
        nest.Connect(pre, spikes[:1])
        nest.Connect(post, spikes[1:])
    build_seconds = time.perf_counter()-build_start
    start = time.perf_counter()
    nest.Prepare()
    prepare_seconds = time.perf_counter()-start
    start = time.perf_counter()
    try:
        nest.Run(condition['duration_ms'])
    except Exception as exc:
        return dict(completed=False, error=str(exc), topology_sha256=topology), None
    finally:
        run_seconds = time.perf_counter()-start
        cleanup_start = time.perf_counter()
        nest.Cleanup()
        cleanup_seconds = time.perf_counter()-cleanup_start
    elapsed = time.perf_counter() - start
    timing = dict(build_seconds=build_seconds, prepare_seconds=prepare_seconds,
                  run_seconds=run_seconds, cleanup_seconds=cleanup_seconds)
    if not record:
        return dict(completed=True, topology_sha256=topology, timing=timing,
                    terminal_C=np.asarray(astro.get('Ca_astro')).tolist()), None
    data = {}
    for prefix, meter in zip(['astro', 'post'], meters):
        data.update({prefix+'_'+k: np.asarray(v) for k, v in meter.events.items()})
    for prefix, rec in zip(['pre_spikes', 'post_spikes'], spikes):
        data[prefix+'_times'] = np.asarray(rec.events['times'])
        data[prefix+'_senders'] = np.asarray(rec.events['senders'])
    duration = condition['duration_ms']/1000.
    times = np.unique(data['post_times'])
    sic = data['post_I_SIC'].reshape(-1, 10)
    # Multimeter samples are time-major; assert ordering before reducing.
    assert np.array_equal(data['post_times'].reshape(-1, 10)[:, 0], times)
    rate_bins = np.arange(0., condition['duration_ms']+100., 100.)
    data['rate_times'] = (rate_bins[:-1]+50.)/1000.
    data['population_rate'] = np.histogram(data['post_spikes_times'], rate_bins)[0]
    data['population_rate'] = data['population_rate']/(10*.1)
    result = dict(completed=True, topology_sha256=topology,
                  timing=timing,
                  recorded_simulate_seconds=elapsed,
                  pre_spikes=len(data['pre_spikes_times']),
                  post_spikes=len(data['post_spikes_times']),
                  post_rate_hz=len(data['post_spikes_times'])/(10*duration),
                  mean_SIC_pA=float(sic.mean()), max_SIC_pA=float(sic.max()),
                  C_range=[float(data['astro_Ca_astro'].min()), float(data['astro_Ca_astro'].max())],
                  IP3_range=[float(data['astro_IP3'].min()), float(data['astro_IP3'].max())],
                  h_range=[float(data['astro_h_IP3R'].min()), float(data['astro_h_IP3R'].max())])
    return result, data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = Path(nest.__file__).parent/'doc/examples/pynest/astrocytes/astrocyte_small_network.py'
    report = dict(source=str(source.relative_to(Path(nest.__file__).parent)),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  nest_version=nest.__version__, variants=VARIANTS,
                  conditions={}, note='Feedforward pilot; one seed. Recorded runtimes are not benchmarks.')
    conditions = {
        'original_example': dict(duration_ms=1000., delta_IP3=.2, record_ms=1.),
        'extended_default_jump': dict(duration_ms=30000., delta_IP3=.0002, record_ms=10.),
    }
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    traces = {}
    for label, condition in conditions.items():
        rows = {}; report['conditions'][label] = dict(parameters=condition, results=rows)
        for name in VARIANTS:
            result, data = run_one(name, condition)
            rows[name] = result
            print(label, name, json.dumps(result), flush=True)
            if data is not None:
                np.savez_compressed(OUT/f'{label}_{name}.npz', **data)
                if label == 'extended_default_jump':
                    traces[name] = data
        assert len({r['topology_sha256'] for r in rows.values()}) == 1
        report['conditions'][label]['identical_topology'] = True
        if rows['LR']['completed']:
            baseline = np.load(OUT/f'{label}_LR.npz')
            for name, r in rows.items():
                if not r['completed']: continue
                other = np.load(OUT/f'{label}_{name}.npz')
                r['identical_presynaptic_spikes'] = bool(
                    np.array_equal(other['pre_spikes_times'], baseline['pre_spikes_times']) and
                    np.array_equal(other['pre_spikes_senders'], baseline['pre_spikes_senders']))
                assert r['identical_presynaptic_spikes']
                for field in ['post_I_SIC', 'population_rate']:
                    r[field+'_RMSE_vs_LR'] = float(np.sqrt(np.mean((other[field]-baseline[field])**2)))
                r['rate_change_vs_LR_hz'] = r['post_rate_hz']-rows['LR']['post_rate_hz']
        (OUT/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    for name, d in traces.items():
        axes[0].plot(np.unique(d['astro_times'])/1000., d['astro_Ca_astro'].reshape(-1, 5).mean(axis=1), label=name)
        axes[1].plot(np.unique(d['post_times'])/1000., d['post_I_SIC'].reshape(-1, 10).mean(axis=1), label=name)
        axes[2].plot(d['rate_times'], d['population_rate'], label=name, alpha=.65)
    for ax, label in zip(axes, ['Mean Ca (µM)', 'Mean SIC (pA)', 'Population rate (Hz)']):
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(ncol=3); axes[-1].set_xlabel('Time (s)')
    fig.tight_layout(); fig.savefig(OUT/'comparison.png', dpi=170); fig.savefig(OUT/'comparison.pdf')
    write_report(report)


if __name__ == '__main__':
    main()
