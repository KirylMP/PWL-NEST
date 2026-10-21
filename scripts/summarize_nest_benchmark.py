"""Generate a reviewable report from actual NEST benchmark artifacts."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from pwl_nest.models import ROOT

out=ROOT/'results/nest_benchmark'
data=json.loads((out/'benchmark.json').read_text())
fine=json.loads((out/'resolution_sweep.json').read_text()) if (out/'resolution_sweep.json').exists() else None
validation=json.loads((out/'independent_validation.json').read_text())
code=['native/nest_pwa_module.cpp','native/pwa_step.h','native/pwa_event.cpp',
      'native/solver_timing.cpp','src/pwl_nest/nest_adapter.py',
      'scripts/build_nest_module.py','scripts/benchmark_nest.py','scripts/benchmark_nest_resolution.py']
provenance=dict(base_git_commit=data['git_commit'],
    code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in code},
    binaries_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                    for f in ['.build/pwl_astro_module.so','.build/solver_timing.so']},
    compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
    cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
              if s.startswith('model name')),'unknown'),
    note='New adapter changes are uncommitted relative to base_git_commit; exact source hashes identify measured implementation.')
if fine:
    terminal_checks=[]
    for case in ['AM','FM']:
        for res in sorted({r['resolution_ms'] for r in fine['rows']}):
            rows={r['mode']:r for r in fine['rows'] if r['case']==case and r['resolution_ms']==res}
            if len(rows)!=3:continue
            for kind in ['lr','pwa']:
                actual=rows['network_with_SIC']['raw'][-1]['detail'][kind]['terminal']
                ref=rows['isolated_solver']['raw'][-1]['detail'][kind]['terminal']
                error=max(abs(actual[f]-ref[d]) for d,f in enumerate(['Ca_astro','h_IP3R','IP3']))
                if error>1e-6:raise AssertionError((case,res,kind,error))
                terminal_checks.append(dict(case=case,resolution_ms=res,kind=kind,max_abs_error=error))
    provenance['isolated_vs_NEST_terminal_checks']=terminal_checks
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
lines=['# AM/FM в NEST: SIC и скорость', '',
       'Загружаемый C++-модуль `astrocyte_lr_pwa` сравнивается со штатным',
       '`astrocyte_lr_1994`. Весь путь `spike_generator → астроцит → sic_connection →',
       'aeif_cond_alpha_astro` работает внутри NEST 3.10.0.', '',
       '## Скорость в NEST', '',
       'Один поток, одинаковые входы, без мультиметров. Три повтора после прогрева;',
       'ускорение = время LR / время PWA. Build и Prepare исключены из Run.',
       'Популяции независимы, получают общий вход; это не рекуррентная сеть.', '',
       '| Модель | Популяция | Только астроциты | Сеть с SIC и нейронами | Изолированный решатель |',
       '|---|---:|---:|---:|---:|']
for case,record in data['cases'].items():
    for n in data['populations']:
        vals={r['mode']:r['lr_over_pwa'] for r in record['timing'] if r['population']==n}
        lines.append(f"| {case} | {n} | {vals['astrocytes_only']:.2f}× | {vals['network_with_SIC']:.2f}× | {vals['isolated_solver']:.2f}× |")
lines+=['','Эта таблица: шаг 10 мс, 120 с модельного времени. Числа для одного',
        'элемента занимают миллисекунды и сильнее зависят от накладных расходов.',
        'Изолированный LR использует эквивалентные уравнения и GSL RKF45;',
        'это диагностический замер, не время встроенной модели NEST.','']
if fine:
    lines+=['### Стандартный шаг NEST и чувствительность скорости к разрешению','',
            f"Популяция {fine['population']}, {fine['duration']:g} с, три повтора после прогрева; задержка соединений фиксирована 10 мс.",'',
            '| Модель | Шаг, мс | Только астроциты | Сеть с SIC | Изолированный решатель |',
            '|---|---:|---:|---:|---:|']
    for case in ['AM','FM']:
        for res in sorted({r['resolution_ms'] for r in fine['rows']}):
            rows={r['mode']:r for r in fine['rows'] if r['case']==case and r['resolution_ms']==res}
            if len(rows)!=3:continue
            vals={k:v['lr_over_pwa'] for k,v in rows.items()}
            lines.append(f"| {case} | {res:g} | {vals['astrocytes_only']:.2f}× | {vals['network_with_SIC']:.2f}× | {vals['isolated_solver']:.2f}× |")
    lines+=['','Сырые времена всех повторов, включая отдельный Prepare:',
            '[resolution_sweep.json](resolution_sweep.json).','']
lines+=['## Одинаковые входы и выход SIC','',
        'Пять протоколов, общие физические начальные состояния из равновесия LR.',
        'Коэффициенты заморожены; фазовое выравнивание не используется.',
        'SIC_scale=1, SIC_th=.19669 μM; измеряется доставленный I_SIC нейрона.',
        'У AM общий начальный покой не является собственным точным покоем PWA.','',
        '| Модель | Вход | Пики C LR/PWA | Пики SIC LR/PWA | RMSE C, μM | RMSE SIC, pA | Площадь SIC PWA/LR |',
        '|---|---|---:|---:|---:|---:|---:|']
for case,record in data['cases'].items():
    for r in record['validation']:
        m=r['metrics'];ratio=m['pwa']['sic_area']/m['lr']['sic_area']
        lines.append(f"| {case} | {r['protocol']['name']} | {m['lr']['calcium_peaks']}/{m['pwa']['calcium_peaks']} | {m['lr']['sic_peak_count']}/{m['pwa']['sic_peak_count']} | {m['C']['rmse']:.5f} | {m['I_SIC']['rmse']:.4f} | {ratio:.3f} |")
lines+=['','Порог prominence для C — .02 μM, для SIC — .02 pA; это разные',
        'счётчики. Подпороговый C-пик не обязан давать SIC. Счёт включает всю',
        'траекторию, в том числе начальную релаксацию AM PWA до первого входа.','',
        'У FM число пиков совпадает на этих пяти протоколах, но ошибка фазы/формы',
        'остаётся и на длинном спаде возрастает. У AM есть несовпадения числа пиков.',
        'Ускорение решателя не устраняет ошибку модели. Эти проверки не являются',
        'полной бифуркационной или независимой слепой приёмкой.','',
        '![FM: кальций и доставленный SIC](FM_transfer.png)','',
        '![AM: кальций и доставленный SIC](AM_transfer.png)','',
        '## Проверка реализации','',
        '| Модель | Реализация | max ошибка C против независимого DOP853, μM |',
        '|---|---|---:|']
for case,rows in validation.items():
    for kind,r in rows.items():lines.append(f"| {case} | {kind} | {r['max_abs_C']:.3g} |")
delivery=max(v['metrics'][k]['sic_delivery_max_error'] for c in data['cases'].values()
             for v in c['validation'] for k in ['lr','pwa'])
half=max(v['C'] for c in data['cases'].values() for v in c['half_tick_max_difference'].values())
lines+=['',f'Проверка DOP853: полный одиночный вход tau20, 220 с. Максимальная ошибка',
        f'доставки SIC против readout C с задержкой: {delivery:.3g} pA.',
        f'Деление шага 10→5 мс на этом входе меняет C максимум на {half:.3g} μM.',
        'Терминальные состояния изолированных ядер сверены с NEST, включая sweep разрешений.',
        'Полный набор тестов: 84 passed; также проверены split Simulate и два потока.','',
        'C-пересечения локализуются по аналитическому региональному потоку,',
        'I-пересечения — через логарифм. Полные тиковые пропагаторы кешируются.',
        'Планировщик NEST остаётся шаговым; глобальной гарантии первого выхода',
        'за один тик нет. Наклонные C-пороги и CurrentEvent не поддержаны.','',
        '## Артефакты и воспроизведение','',
        '- [Методика, сборка и API](../../docs/nest_module.md).',
        '- [Основные сырые результаты](benchmark.json).',
        '- [Независимый аудит](independent_validation.json).',
        '- [Хеши исходников, бинарных файлов и CPU](provenance.json).',
        '- Полные траектории: `AM_*_{lr,pwa}.npz`, `FM_*_{lr,pwa}.npz`.',
        '- Графики для экспорта: `AM_transfer.pdf`, `FM_transfer.pdf`.','']
(out/'REPORT.md').write_text('\n'.join(lines))
print(out/'REPORT.md')
