"""Fixed neuronal population, varying astrocyte count; matched frozen inputs.

Load/transfer benchmark, not a recurrent-network validation. All times in ms
unless a field explicitly says seconds. A neuron receives SIC weights summing
to a fixed gain at every astrocyte/neuron ratio.
"""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nest
import numpy as np

from pwl_nest.models import ROOT
from pwl_nest.nest_adapter import install, load_candidate, pwa_parameters, lr_parameters

MODEL=ROOT/'models/high_ip3_hinge/full_AM.json'
SEED=20261101
NEURONS=20
COUNTS=[2,20,200]
DURATION=60000.
DT=1.
GAIN=10.


def inputs():
    banks=[]
    for index in range(max(COUNTS)):
        rng=np.random.default_rng(SEED+index)
        ticks=np.arange(50.,DURATION-1.,50.)+rng.uniform(-20.,20.,len(np.arange(50.,DURATION-1.,50.)))
        banks.append(np.round(ticks/DT)*DT)
    return banks


def connections(na,nn):
    if not nn or not na:return [],[],[]
    if na<=nn:
        src=np.arange(nn)*na//nn;dst=np.arange(nn)
    else:
        src=np.arange(na);dst=src%nn
    weights=GAIN/np.bincount(dst,minlength=nn)[dst]
    np.testing.assert_allclose(np.bincount(dst,weights=weights,minlength=nn),GAIN)
    return src.tolist(),dst.tolist(),weights.tolist()


def run(kind,na,nn,record,bank):
    start=time.perf_counter()
    nest.ResetKernel();install()
    nest.SetKernelStatus(dict(resolution=DT,local_num_threads=1,rng_seed=SEED))
    nest.verbosity=nest.VerbosityLevel.ERROR
    model,_=load_candidate('AM',MODEL)
    astro=None;neurons=None;meters=[];spikes=None
    if na:
        if kind=='pwa':
            params=pwa_parameters(model,[.073,.793,.4],7.142,.16)
            params['allow_ip3_extrapolation']=True
            name='astrocyte_lr_pwa'
        else:
            params=lr_parameters('AM',[.073,.793,.4],7.142,.16)
            name='astrocyte_lr_1994'
        params['delta_IP3']=.2
        astro=nest.Create(name,na,params=params)
        generators=nest.Create('spike_generator',na)
        for i in range(na):generators[i:i+1].set(spike_times=bank[i])
        nest.Connect(generators,astro,'one_to_one',syn_spec=dict(weight=1.,delay=DT))
    if nn:
        neurons=nest.Create('aeif_cond_alpha_astro',nn,params=dict(tau_syn_ex=2.))
        neurons.set(I_e=np.linspace(700.,900.,nn))
        if na:
            src,dst,weights=connections(na,nn)
            for i,j,w in zip(src,dst,weights):
                nest.Connect(astro[i:i+1],neurons[j:j+1],syn_spec=dict(
                    synapse_model='sic_connection',weight=w,delay=DT))
    if record:
        for prefix,nodes,fields in [('astro',astro,['Ca_astro','h_IP3R','IP3']),
                                    ('neuron',neurons,['I_SIC','V_m'])]:
            if nodes is None:continue
            meter=nest.Create('multimeter',params=dict(interval=50.,record_from=fields))
            nest.Connect(meter,nodes);meters.append((prefix,meter,nodes.tolist()))
        if nn:
            spikes=nest.Create('spike_recorder');nest.Connect(neurons,spikes)
    build=time.perf_counter()-start
    start=time.perf_counter();nest.Prepare();prepare=time.perf_counter()-start
    start=time.perf_counter()
    try:nest.Run(DURATION)
    finally:
        elapsed=time.perf_counter()-start
        start=time.perf_counter();nest.Cleanup();cleanup=time.perf_counter()-start
    start=time.perf_counter();data={}
    for prefix,meter,ids in meters:
        ev=meter.events
        # Explicit stable ordering by (time, node), independent of internal logging order.
        order=np.lexsort((ev['senders'],ev['times']))
        for key,value in ev.items():data[prefix+'_'+key]=np.asarray(value)[order]
    if spikes is not None:
        ev=spikes.events
        data['spike_times']=np.asarray(ev['times'])
        data['spike_neurons']=np.searchsorted(neurons.tolist(),ev['senders'])
    readout=time.perf_counter()-start
    terminal={} if astro is None else {k:np.asarray(astro.get(k)).tolist() for k in ['Ca_astro','h_IP3R','IP3']}
    counters={} if astro is None or kind=='lr' else {k:int(np.sum(astro.get(k))) for k in
             ['affine_flows','region_crossings','input_crossings','input_spikes']}
    return data,dict(timing=dict(build_seconds=build,prepare_seconds=prepare,run_seconds=elapsed,
                     cleanup_seconds=cleanup,readout_seconds=readout),terminal=terminal,counters=counters)


def metrics(data,na,nn):
    r={}
    if na:
        for field in ['Ca_astro','h_IP3R','IP3']:
            x=data['astro_'+field]
            r[field+'_range']=[float(x.min()),float(x.max())]
        c=data['astro_Ca_astro'];h=data['astro_h_IP3R']
        r['physical_recorded_states']=bool(np.isfinite(c).all() and np.isfinite(h).all() and
            np.all((c>=0)&(c<=2/1.185)) and np.all((h>=0)&(h<=1)))
    if nn:
        t=np.unique(data['neuron_times'])/1000.
        sic=data['neuron_I_SIC'].reshape(-1,nn)
        r['spike_count']=len(data['spike_times'])
        r['mean_rate_hz']=r['spike_count']/(nn*DURATION/1000)
        r['mean_SIC_pA']=float(sic.mean());r['max_SIC_pA']=float(sic.max())
        r['mean_SIC_area_pA_seconds']=float(np.trapezoid(sic.mean(axis=1),t))
        cvs=[];binned=[]
        for node in range(nn):
            times=data['spike_times'][data['spike_neurons']==node]
            isi=np.diff(times)
            if len(isi)>1:cvs.append(float(np.std(isi)/np.mean(isi)))
            binned.append(np.histogram(times,np.arange(0,DURATION+100.,100.))[0])
        r['mean_ISI_CV']=float(np.mean(cvs)) if cvs else None
        binned=np.asarray(binned);active=binned.std(axis=1)>0
        r['mean_pair_correlation_100ms']=(float(np.corrcoef(binned[active])[np.triu_indices(active.sum(),1)].mean())
            if active.sum()>1 else None)
    return r


def report(out,d):
    # Check that removing downstream neurons leaves exactly the same astrocyte task.
    d['isolation_checks']={}
    d['input_sha256']=hashlib.sha256((out/'inputs.npz').read_bytes()).hexdigest()
    for na in COUNTS:
        for kind in ['lr','pwa']:
            a=np.load(out/'traces'/f'network_{na}_{kind}.npz')
            b=np.load(out/'traces'/f'astro_only_{na}_{kind}.npz')
            equal=all(np.array_equal(a['astro_'+k],b['astro_'+k]) for k in
                      ['times','Ca_astro','h_IP3R','IP3'])
            assert equal,(na,kind)
            d['isolation_checks'][f'{na}_{kind}']=equal
    rng=np.random.default_rng(SEED+20000)
    for c in d['configs']:
        tim=d['timing'][c['key']]
        for kind in (['lr','pwa'] if 'pwa' in tim else ['lr']):
            r=tim[kind]
            values=np.array([s['timing']['run_seconds'] for s in r['samples']])
            assert len(values)==7
            r['median_seconds']=float(np.median(values))
            r['min_seconds']=float(values.min());r['max_seconds']=float(values.max())
        if 'pwa' in tim:
            a=np.array([s['timing']['run_seconds'] for s in tim['lr']['samples']])
            b=np.array([s['timing']['run_seconds'] for s in tim['pwa']['samples']])
            idx=rng.integers(0,7,size=(5000,7))
            tim['speedup_bootstrap_95_interval']=np.quantile(
                np.median(a[idx],axis=1)/np.median(b[idx],axis=1),[.025,.975]).tolist()
    tn=d['timing']['neuron_only']['lr']['median_seconds']
    d['additive_cost_estimate']={}
    for na in COUNTS:
        a=d['timing'][f'astro_only_{na}']['lr']['median_seconds']
        b=d['timing'][f'astro_only_{na}']['pwa']['median_seconds']
        d['additive_cost_estimate'][str(na)]=dict(astro_share_LR=a/(a+tn),
            predicted_network_speedup=(a+tn)/(b+tn),
            measured_network_speedup=d['timing'][f'network_{na}']['lr']['median_seconds']/
                                     d['timing'][f'network_{na}']['pwa']['median_seconds'])
    (out/'evaluation.json').write_text(json.dumps(d,indent=2)+'\n')
    rows=[];timingrows=[];lines=['# Масштабирование числа астроцитов', '',
      'Нагрузочный тест AM с фиксированными 20 AEIF-нейронами. '
      'Это заданный спайковый вход → астроциты → нейроны, без обратной связи. '
      'Используется полная AM с дополнительным IP3-порогом 1.3.', '',
      'Каждый астроцит получает свой заранее заданный нерегулярный поток '
      'около 20 Гц (период 50 мс с независимым джиттером ±20 мс), '
      'скачок IP3=0.2 мкМ, tau=7.142 с. '
      'Для LR/PWA входы совпадают; при росте популяции сохраняются входы '
      'ранее добавленных клеток. Потоки разных клеток различаются. '
      'Сумма весов SIC на каждом нейроне равна 10 во всех отношениях. '
      'Нормировка сохраняет усиление, но увеличение числа независимых '
      'источников меняет усреднение сигнала; поведение разных размеров '
      'не обязано совпадать.', '',
      '60 с моделируемого времени, шаг NEST 1 мс, один поток. '
      'Время — nest.Run без регистраторов: отдельный прогрев для каждой '
      'конфигурации, семь повторов с перемешанным порядком. '
      'Построение, Prepare, Cleanup и считывание сохранены отдельно.', '',
      '| Конфигурация | Астроцитов | Нейронов | LR, с | PWA, с | Ускорение | 95% интервал |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for config in d['configs']:
        key=config['key'];tim=d['timing'][key]
        if 'pwa' in tim:
            lr=tim['lr']['median_seconds'];pw=tim['pwa']['median_seconds']
            speed=lr/pw
            ci=tim['speedup_bootstrap_95_interval']
            lines.append(f"| {config['scope']} | {config['astrocytes']} | {config['neurons']} | {lr:.3f} | {pw:.3f} | {speed:.2f}× | {ci[0]:.2f}–{ci[1]:.2f} |")
        else:
            lines.append(f"| Только нейроны | 0 | {NEURONS} | {tim['lr']['median_seconds']:.3f} | — | — | — |")
        for kind,r in d['diagnostics'][key].items():
            rows.append(dict(config=key,kind=kind,**{k:v for k,v in r['metrics'].items() if not isinstance(v,(dict,list))}))
        for kind in (['lr','pwa'] if 'pwa' in tim else ['lr']):
            r=tim[kind]
            for sample in r['samples']:
                timingrows.append(dict(config=key,kind=kind,repeat=sample['repeat'],**sample['timing']))
    for filename,items in [('metrics.csv',rows),('timings.csv',timingrows)]:
        with (out/filename).open('w') as f:
            writer=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in items))))
            writer.writeheader();writer.writerows(items)
    lines+=['', '## Точность и нейронный эффект', '',
      '| Астроцитов/нейрон | RMSE Ca, мкМ | RMSE SIC, пА | Частота LR/PWA, Гц | Прирост спайков LR/PWA к контролю без астроцитов |',
      '|---|---:|---:|---:|---:|']
    no=d['diagnostics']['neuron_only']['lr']['metrics']['spike_count']
    for na in COUNTS:
        x=d['diagnostics'][f'network_{na}'];a=x['lr']['metrics'];b=x['pwa']['metrics']
        lines.append(f"| {na/NEURONS:g} | {b['Ca_astro_RMSE_vs_LR']:.4f} | {b['I_SIC_RMSE_vs_LR']:.3f} | "
          f"{a['mean_rate_hz']:.3f}/{b['mean_rate_hz']:.3f} | {a['spike_count']-no}/{b['spike_count']-no} |")
    lines+=['', '![Время и ускорение](scaling.png)', '',
      'Интервалы получены парным bootstrap по семи повторам и описывают '
      'только разброс времени на этой машине. Динамика астроцитов с нейронами '
      'и без них совпала побитово для каждого размера и обеих моделей. '
      'В JSON сохранена также аддитивная оценка по закону Амдала: '
      '(T_нейронов + T_астро_LR)/(T_нейронов + T_астро_PWA). '
      'Это оценка по отдельным запускам, а не профилирование сети.', '',
      'Тест без нейронов измеряет астроцитарную популяцию вместе с '
      'доставкой спайков и инфраструктурой NEST, не чистое время C++-ядра. '
      'Простая сумма времени изолированных компонентов может отличаться '
      'от сети из-за передачи SIC и планирования событий. '
      'Число SIC-соединений также меняется и сохранено в плане.', '',
      'Это одна нагрузочная конфигурация при одном шаге интегрирования; '
      'точность двух моделей не равна. Выигрыш нельзя переносить на '
      'произвольные сети или выдавать за сравнение при равной ошибке.', '',
      '[Полные результаты и сырые повторы](evaluation.json) · '
      '[Метрики](metrics.csv) · [Времена стадий](timings.csv). '
      'Полные траектории — `traces/*.npz`, входы — `inputs.npz`. '
      'В JSON также сохранены физические диапазоны, CV ISI, корреляции, '
      'ошибки частот и интеграл SIC.', '',
      'Воспроизведение: `.venv/bin/python scripts/benchmark_astro_ratio.py '
      '--output results/astro_ratio_repeat`. Только отчёт: '
      '`--report-only --output results/astro_ratio`.' ]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for scope,style in [('network','-o'),('astro_only','--s')]:
        for kind in ['lr','pwa']:
            values=[d['timing'][f'{scope}_{na}'][kind]['median_seconds'] for na in COUNTS]
            axes[0].plot(COUNTS,values,style,label=f'{scope} {kind}')
        values=[d['timing'][f'{scope}_{na}']['lr']['median_seconds']/d['timing'][f'{scope}_{na}']['pwa']['median_seconds'] for na in COUNTS]
        axes[1].plot(COUNTS,values,style,label=scope)
    axes[0].set_ylabel('nest.Run time (s)');axes[1].set_ylabel('Speedup LR/PWA')
    for ax in axes:
        ax.set_xscale('log');ax.set_xlabel('Number of astrocytes');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'scaling.png',dpi=170);fig.savefig(out/'scaling.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/astro_ratio')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();out=args.output
    if args.report_only:
        report(out,json.loads((out/'evaluation.json').read_text()));return
    if out.exists():raise FileExistsError(f'Use a new output directory: {out}')
    out.mkdir(parents=True);(out/'traces').mkdir()
    bank=inputs();np.savez_compressed(out/'inputs.npz',**{f'astro_{i}':s for i,s in enumerate(bank)})
    configs=[dict(key='neuron_only',scope='neuron_only',astrocytes=0,neurons=NEURONS)]
    for na in COUNTS:
        for scope,nn in [('network',NEURONS),('astro_only',0)]:
            configs.append(dict(key=f'{scope}_{na}',scope=scope,astrocytes=na,neurons=nn,
                                sic_connections=len(connections(na,nn)[0])))
    d=dict(configs=configs,parameters=dict(duration_ms=DURATION,resolution_ms=DT,threads=1,
             seed=SEED,sic_weight_sum=GAIN,delta_IP3=.2,tau_IP3_seconds=7.142,
             neuron_I_e_pA=[700.,900.],
             repetitions=7,warmups=1),model=str(MODEL.relative_to(ROOT)),
             model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),diagnostics={},timing={})
    jobs=[]
    for c in configs:
        key=c['key'];na=c['astrocytes'];nn=c['neurons']
        d['diagnostics'][key]={};d['timing'][key]={};truth=None
        for kind in (['lr','pwa'] if na else ['lr']):
            data,r=run(kind,na,nn,True,bank)
            r['metrics']=metrics(data,na,nn)
            if kind=='lr':truth=data
            else:
                for field in ['Ca_astro','h_IP3R','IP3']:
                    delta=data['astro_'+field]-truth['astro_'+field]
                    r['metrics'][field+'_RMSE_vs_LR']=float(np.sqrt(np.mean(delta**2)))
                np.testing.assert_allclose(data['astro_IP3'],truth['astro_IP3'],rtol=0,atol=1e-7)
                if nn:
                    r['metrics']['I_SIC_RMSE_vs_LR']=float(np.sqrt(np.mean((data['neuron_I_SIC']-truth['neuron_I_SIC'])**2)))
                    bins=np.arange(0,DURATION+1000.,1000.)
                    a=np.histogram(data['spike_times'],bins)[0]/nn
                    b=np.histogram(truth['spike_times'],bins)[0]/nn
                    r['metrics']['population_rate_RMSE_1s']=float(np.sqrt(np.mean((a-b)**2)))
            np.savez_compressed(out/'traces'/f'{key}_{kind}.npz',**data)
            d['diagnostics'][key][kind]=r
            jobs.append((key,kind,na,nn))
            print('diagnostic',key,kind,flush=True)
        (out/'evaluation.json').write_text(json.dumps(d,indent=2)+'\n')
    for key,kind,na,nn in jobs:
        _,r=run(kind,na,nn,False,bank)
        d['timing'][key][kind]=dict(warmup=r,samples=[])
    rng=np.random.default_rng(SEED+10000)
    for repeat in range(7):
        order=rng.permutation(len(jobs)).tolist()
        for i in order:
            key,kind,na,nn=jobs[i]
            _,r=run(kind,na,nn,False,bank);r['repeat']=repeat
            d['timing'][key][kind]['samples'].append(r)
        print('timing repetition',repeat+1,flush=True)
        (out/'evaluation.json').write_text(json.dumps(d,indent=2)+'\n')
    for c in configs:
        for kind,r in d['timing'][c['key']].items():
            values=[s['timing']['run_seconds'] for s in r['samples']]
            r['median_seconds']=statistics.median(values)
            r['min_seconds']=min(values);r['max_seconds']=max(values)
            assert all(s['terminal']==r['samples'][0]['terminal'] for s in r['samples'])
    d['environment']=dict(python=platform.python_version(),nest=nest.__version__,
          cpu=next(s.split(':',1)[1].strip() for s in open('/proc/cpuinfo') if s.startswith('model name')),
          module_sha256=hashlib.sha256((ROOT/'.build/pwl_astro_module.so').read_bytes()).hexdigest())
    d['source_sha256']={}
    for p in ['scripts/benchmark_astro_ratio.py','native/nest_pwa_module.cpp','native/pwa_step.h',
              'native/pwa_event.cpp','src/pwl_nest/nest_adapter.py']:
        d['source_sha256'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    (out/'evaluation.json').write_text(json.dumps(d,indent=2)+'\n')
    report(out,d)


if __name__=='__main__':main()
