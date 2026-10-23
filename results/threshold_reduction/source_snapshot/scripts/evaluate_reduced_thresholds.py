"""Matched SIC diagnostics and timing after reduced-model TRAIN selection."""
import hashlib
import json
from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark_nest import network_run,compare,isolated
from overnight_coordinate_grid import source_rest
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates
from pwl_nest.rest_protocols import stable_rest
from fit_reduced_thresholds import SOURCES

OUT=ROOT/'results/threshold_reduction'
TASKS=['AM_cube_2C','AM_log_2C','AM_cube_2C_2I','AM_log_2C_2I','FM_4C']


def run():
    plan=json.loads((OUT/'evaluation_plan.json').read_text())
    modeldir=ROOT/'models/threshold_reduction';modeldir.mkdir(parents=True,exist_ok=True)
    models={k:CalciumTransformPWA.load(ROOT/p) for k,p in SOURCES.items()}
    records={}
    for name in TASKS:
        selection=json.loads((OUT/name/'selection.json').read_text())
        config=json.loads((OUT/name/'config.json').read_text())
        model=CalciumTransformPWA.load(OUT/name/'selected.json')
        model.metadata['current_validation']='results/threshold_reduction/REPORT.md'
        path=modeldir/f'{name}.json';model.save(path);models[name]=model
        records[name]=dict(model=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source=config['source'],full_train_loss=selection['full_model_loss'],
            reduced_train_loss=selection['selected']['train_loss'],C_thresholds=model.metadata['physical_thresholds_C'],
            input_thresholds=model.input_thresholds.tolist(),regions=model.latent.region_count*(len(model.input_thresholds)+1),
            active_parameters=model.metadata['active_fit_parameters'])
    chosen={}
    for geometry,names in [('AM_2C',TASKS[:2]),('AM_2C_2I',TASKS[2:4]),('FM_4C',[TASKS[4]])]:
        winner=min(names,key=lambda k:records[k]['reduced_train_loss']);chosen[geometry]=winner
        models[winner].save(modeldir/f'selected_{geometry}.json')
    report=dict(plan=plan,model_records=records,selected_by_train=chosen,
                loss='Common physical RestObjective without source spectral-boundary anchors; values are not old full-objective losses',
                diagnostics={},timing={},numerical_validation={})
    traces={};data_dir=OUT/'nest';data_dir.mkdir(exist_ok=True)
    for case in ['AM','FM']:
        names=([k for k in models if k.startswith('AM')] if case=='AM' else ['FM','FM_4C'])
        rows=[];traces[case]={}
        for p in plan['protocols'][case]:
            initial=source_rest(case,p['baseline'])
            truth,_=network_run(case,models[names[0]],initial,p,'lr',10.,record=True)
            traces[case][p['name']]={'LR':truth};one={}
            np.savez_compressed(data_dir/f'{case}_{p["name"]}_LR.npz',**truth)
            for name in names:
                model=models[name]
                try:
                    rest,count=stable_rest(model,p['baseline'],initial)
                    data,_=network_run(case,model,rest,p,'pwa',10.,record=True)
                    if any(not np.isfinite(data[k]).all() for k in ['C','h','I_SIC']):
                        raise ValueError('Nonfinite physical output')
                    result=compare(truth,data,.01)
                    result.update(completed=True,own_rest=rest.tolist(),stable_rest_count=count,
                        physical=bool(np.all((data['C']>0)&(data['C']<2/1.185)&(data['h']>0)&(data['h']<1))))
                    traces[case][p['name']][name]=data
                    np.savez_compressed(data_dir/f'{p["name"]}_{name}.npz',**data)
                except Exception as error:
                    result=dict(completed=False,error=f'{type(error).__name__}: {error}')
                one[name]=result
            rows.append(dict(protocol=p,models=one))
            print(json.dumps(dict(case=case,protocol=p['name'],SIC_RMSE={k:r.get('I_SIC',{}).get('rmse') for k,r in one.items()})),flush=True)
        report['diagnostics'][case]=rows
        report['diagnostics'][case+'_aggregate']={name:dict(
            complete=sum(r['models'][name]['completed'] for r in rows),
            mean_SIC_RMSE=float(np.mean([r['models'][name]['I_SIC']['rmse'] for r in rows])) if all(r['models'][name]['completed'] for r in rows) else None,
            SIC_count_matches=sum(r['models'][name].get('lr',{}).get('sic_peak_count')==r['models'][name].get('pwa',{}).get('sic_peak_count') for r in rows if r['models'][name]['completed']),
            physical=sum(r['models'][name].get('physical',False) for r in rows)) for name in names}
        shown=(['LR','AM_cube',chosen['AM_2C'],chosen['AM_2C_2I']] if case=='AM' else ['LR','FM','FM_4C'])
        fig,axes=plt.subplots(2,3,figsize=(13,6),sharex='col')
        for col,p in enumerate([plan['protocols'][case][k] for k in [0,2,4]]):
            for name,color in zip(shown,['black','tab:blue','tab:orange','tab:green']):
                data=traces[case][p['name']].get(name)
                if data is None:continue
                axes[0,col].plot(data['t'][::5],data['C'][::5],label=name,color=color,lw=1)
                axes[1,col].plot(data['t'][::5],data['I_SIC'][::5],label=name,color=color,lw=1)
            axes[0,col].set_title(p['name']);axes[1,col].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Ca (µM)');axes[1,0].set_ylabel('Delivered SIC (pA)')
        axes[0,0].legend(fontsize=7);fig.suptitle(f'{case}: threshold reduction, unseen input protocols')
        fig.tight_layout();fig.savefig(OUT/f'{case}_transfer.png',dpi=160);fig.savefig(OUT/f'{case}_transfer.pdf');plt.close(fig)
        for name in [n for n in names if n in records]:
            p=plan['protocols'][case][0];initial=source_rest(case,p['baseline'])
            rest,_=stable_rest(models[name],p['baseline'],initial)
            data=traces[case][p['name']].get(name)
            if data is None:continue
            try:
                t,x,_=simulate_coordinates(models[name],rest,**{k:v for k,v in p.items() if k!='name'},dt=.01,
                    rtol=2e-11,atol=2e-13,max_step=.01)
                error=[float(np.max(np.abs(data[k]-np.interp(data['t'],t,x[:,d])))) for d,k in enumerate(['C','h'])]
                if max(error)>3e-5:raise ValueError(f'DOP mismatch: {error}')
                report['numerical_validation'][name]=dict(max_abs_C=error[0],max_abs_h=error[1])
            except Exception as e:report['numerical_validation'][name]=dict(error=str(e))
    (OUT/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Timing never participates in choosing a reduced model.
    for case in ['AM','FM']:
        names=(['AM_cube','AM_log',chosen['AM_2C'],chosen['AM_2C_2I']] if case=='AM' else ['FM','FM_4C'])
        config=plan['timing'];resolution=config['resolution_ms'];n=config['population']
        base=.3 if case=='AM' else .45;initial=source_rest(case,base)
        p=dict(duration=config['duration'],tau=20.,baseline=base,jumps=[[20.,.35 if case=='AM' else .7]])
        timed={}
        for mode in ['astrocytes_only','network_with_SIC']:
            raw={name:[] for name in ['LR']+names}
            for repeat in range(config['repeats']+1):
                order=['LR']+names
                if repeat%2:order=order[::-1]
                for name in order:
                    model=models[names[0]] if name=='LR' else models[name]
                    _,result=network_run(case,model,initial,p,'lr' if name=='LR' else 'pwa',resolution,
                        population=n,record=False,with_targets=mode=='network_with_SIC',delay_ms=10.)
                    if repeat:raw[name].append(result)
            median={name:statistics.median(r['run_seconds'] for r in values) for name,values in raw.items()}
            timed[mode]=dict(raw=raw,median_seconds=median,
                speedup_vs_LR={name:median['LR']/median[name] for name in names})
            print(json.dumps(dict(case=case,mode=mode,median=median)),flush=True)
        kernels={name:[isolated(case,models[name],initial,p,resolution,n) for _ in range(config['repeats'])] for name in names}
        timed['isolated_solver']=dict(raw=kernels,
            median_PWA_seconds={name:statistics.median(r['pwa']['solve_seconds'] for r in rows) for name,rows in kernels.items()},
            speedup_vs_LR={name:statistics.median(r['lr']['solve_seconds']/r['pwa']['solve_seconds'] for r in rows) for name,rows in kernels.items()})
        report['timing'][case]=timed
        (OUT/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_report(report)


def write_report(report):
    lines=['# Сокращение числа порогов AM/FM','',
        'Сравниваются две геометрии AM: два C-порога без I-порогов и те же',
        'два C-порога с сохранёнными I-узлами .35,.64. Для FM — четыре C-порога.',
        'Непрерывность поля, экспонента IP3 и физический readout сохранены.',
        'Коэффициенты и упорядоченные C-пороги подбирались заново.',
        'Старая последовательность спектральных типов не навязывалась;',
        'константы двух строк выводятся из точного устойчивого покоя LR.',
        'Это отдельный класс приближений, а не чистая абляция порогов.','',
        '## Выбор по обучению','',
        'Общий физический RestObjective: статические режимы, собственный покой,',
        'пики, ширины, интервалы и слабая поточечная ошибка. Спектральные',
        'якоря границ исключены и для полного контроля, и для сокращённой',
        'модели. Эти числа нельзя сравнивать со старыми loss с другими остатками.',
        'Выбор — только по полному независимому DOP853 replay обучения.',
        'SIC-тесты и скорость не участвуют в выборе.','',
        'Перебраны все подмножества исходных порогов: по 6 стартовых проекций',
        'на вариант AM и 70 на FM (включая два способа полевой инициализации).',
        'Два лучших старта в каждом варианте уточнялись LS и локальным Powell,',
        'по 1000 вызовов остатка на стадию. Ранние остановки сохранены.',
        'Оптимальность этого локального поиска не утверждается.','',
        '| Модель | C/I-порогов | Областей | Параметров | Полный контроль: loss | Сокращённый: loss |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in report['model_records'].items():
        lines.append(f"| {name} | {len(r['C_thresholds'])}/{len(r['input_thresholds'])} | {r['regions']} | {r['active_parameters']} | {r['full_train_loss']:.4f} | {r['reduced_train_loss']:.4f} |")
    lines+=['','Выбранные внутри каждой геометрии модели:','']
    for geometry,name in report['selected_by_train'].items():
        lines.append(f'- {geometry}: {name}, [JSON](../../models/threshold_reduction/selected_{geometry}.json).')
    lines+=['','## Что получилось','',
        '| Вариант | Средняя RMSE SIC: полный → компактный, pA | Время всей сети: полный → компактный, с |',
        '|---|---:|---:|']
    for geometry,name in report['selected_by_train'].items():
        case='AM' if geometry.startswith('AM') else 'FM'
        full=('AM_log' if 'log' in name else 'AM_cube') if case=='AM' else 'FM'
        metrics=report['diagnostics'][case+'_aggregate']
        timing=report['timing'][case]['network_with_SIC']['median_seconds']
        lines.append(f"| {geometry} | {metrics[full]['mean_SIC_RMSE']:.4f} → {metrics[name]['mean_SIC_RMSE']:.4f} | {timing[full]:.4f} → {timing[name]:.4f} |")
    lines+=['','Сеть остаётся примерно в 2.3 раза быстрее LR в этой конфигурации.',
        'Дополнительное снижение медианы времени после удаления порогов',
        'составляет лишь несколько процентов и не выделяется убедительно',
        'на фоне разброса трёх повторов. Для AM без I-порогов ускорение самих',
        'астроцитов заметнее, но ошибка SIC выше. У AM с I-порогами средняя',
        'ошибка ближе к полной модели, однако число SIC-пиков отличается',
        'чаще. У FM сокращение до пяти областей повышает ошибку SIC.','',
        'Физические C-пороги выбранных моделей, μM:','']
    for geometry,name in report['selected_by_train'].items():
        values=', '.join(f'{v:.8f}' for v in report['model_records'][name]['C_thresholds'])
        lines.append(f'- {geometry}: {values}.')
    lines+=['','## Выход SIC в NEST','',
        'Пять новых входов на AM/FM, предварительно заданных в evaluation_plan.json.',
        'Начало каждого кандидата — из собственного устойчивого покоя,',
        'измеряется доставленный I_SIC целевого нейрона. Шаг 10 мс.',
        'Все полные и сокращённые модели оцениваются на одинаковых спайках.','',
        '| Модель | Завершено | В физических диапазонах | Совпало число SIC-пиков | Средняя RMSE SIC, pA |',
        '|---|---:|---:|---:|---:|']
    for case in ['AM','FM']:
        for name,r in report['diagnostics'][case+'_aggregate'].items():
            value=f"{r['mean_SIC_RMSE']:.4f}" if r['mean_SIC_RMSE'] is not None else 'FAIL'
            lines.append(f"| {name} | {r['complete']}/5 | {r['physical']}/5 | {r['SIC_count_matches']}/5 | {value} |")
    lines+=['','![AM: кальций и SIC](AM_transfer.png)','','![FM: кальций и SIC](FM_transfer.png)','',
        '## Скорость','',
        'Один поток, 64 пары, шаг 1 мс, 60 с модельного времени. Три повтора',
        'после исключённого прогрева; запись траекторий отключена. Build,',
        'Prepare и чтение данных не входят в Run. Изолированные ядра измерены',
        'отдельно, три повтора без отдельного прогрева. Эти результаты',
        'не смешиваются с прежним benchmark .1 мс.','',
        '| Модель | Астроциты: с | Сеть с SIC: с | Ускорение сети против LR | Изолированное ядро: с |',
        '|---|---:|---:|---:|---:|']
    for case,timing in report['timing'].items():
        astro=timing['astrocytes_only']['median_seconds'];network=timing['network_with_SIC']['median_seconds']
        pure=timing['isolated_solver']['median_PWA_seconds']
        for name in network:
            lines.append(f"| {case+'/'+name} | {astro[name]:.4f} | {network[name]:.4f} | {network['LR']/network[name]:.2f}× | {pure[name]:.4f} |" if name!='LR' else
                         f"| {case}/LR | {astro[name]:.4f} | {network[name]:.4f} | 1× | — |")
    lines+=['','## Проверка реализации','',
        'Сокращённые модели отдельно сверены с собственным DOP853 на первом новом входе:','',
        '| Модель | max ошибка C, μM | max ошибка h |','|---|---:|---:|']
    for name,r in report['numerical_validation'].items():
        lines.append(f"| {name} | {r['max_abs_C']:.3g} | {r['max_abs_h']:.3g} |" if 'max_abs_C' in r else f"| {name} | FAIL | {r['error']} |")
    lines+=['','Ускорение и компактность не означают равноточность LR или сохранение',
        'бифуркаций. Время поиска C-пересечения основано на bracketing внутри',
        'тика; глобальной сертификации первого выхода нет. Полная приёмка',
        'моделей не заявляется. Исходные модели не перезаписаны.','',
        '[Формулировки для статьи и границы вывода](../../docs/article_claims.md).',
        'Сырые метрики, времена и все неудачи: evaluation.json и каталоги поиска.',
        'NEST-траектории — nest/*.npz; графики для экспорта — AM_transfer.pdf и FM_transfer.pdf.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(OUT/'REPORT.md',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args();OUT=a.output.resolve()
    if a.report_only:write_report(json.loads((OUT/'evaluation.json').read_text()))
    else:run()
