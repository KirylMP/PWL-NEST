"""Plots and peak diagnostics from saved experiments; never refits/selects models."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/workspace/PWL-NEST/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/PWL-NEST/.cache')
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pwl_nest.models import ROOT

OUT = ROOT/'results/strict_pwa'
fig,axes = plt.subplots(2,3,figsize=(15,7))
peaks_report = {}
for row,case in enumerate(['AM','FM']):
    result = json.loads((OUT/f'{case}.json').read_text())
    peaks_report[case] = {}
    for label in ['decay_tau_2.0','decay_tau_50.0','periodic_jumps_then_stop','irregular_jumps_then_stop']:
        with np.load(OUT/f'{case}_{label}.npz') as z:
            t,lr,pwa = z['t'],z['lr'],z['pwa']
            a,b = find_peaks(lr[:,0],prominence=.02)[0],find_peaks(pwa[:,0],prominence=.02)[0]
            same_count = len(a)==len(b)
            lag = float(np.max(abs(t[a]-t[b]))) if same_count and len(a) else None
            heights = float(np.max(abs(lr[a,0]-pwa[b,0]))) if same_count and len(a) else None
            peaks_report[case][label] = dict(reference_peaks=len(a),pwa_peaks=len(b),
                same_count=same_count,max_abs_peak_time_error_s=lag,max_abs_peak_height_error_uM=heights,
                prominence_uM=.02,time_sample_spacing_approx_s=.2,
                caveat='Peaks paired chronologically only when counts agree; sampled, no sub-grid refinement')
            if label in ['decay_tau_50.0','periodic_jumps_then_stop']:
                col = 0 if label=='decay_tau_50.0' else 1
                axes[row,col].plot(t,lr[:,0],label='Li-Rinzel',lw=1.4)
                axes[row,col].plot(t,pwa[:,0],label='strict PWA',lw=1.1,ls='--')
                axes[row,col].set(xlabel='time (s)',ylabel='C (uM)',title=f'{case}: {label}')
                axes[row,col].legend(fontsize=8)
    for trial in result['training']:
        x,y = trial['metadata']['regions'],trial['training_score']
        axes[row,2].scatter(x,y,marker='o' if trial['label'].endswith('_fit') else 'x',
                            label=trial['label'])
    axes[row,2].set(xscale='log',yscale='log',xlabel='number of tetrahedral regions',
                    ylabel='training dynamical score (lower is better)',title=f'{case}: accuracy / complexity')
    axes[row,2].margins(x=.5,y=.7)
    axes[row,2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT/'strict_pwa_summary.png',dpi=160)
(OUT/'peaks.json').write_text(json.dumps(peaks_report,indent=2)+'\n')
lines=['# Первый строгий PWA-эксперимент','',
       'Все модели используют физические (C,h,I), непрерывное полное поле и точную релаксацию IP3.',
       'Выбраны refined_fit по обучающему score; holdout не участвовал в выборе.',
       '','| Модель | Узлы | Регионы | Обучающий score |','|---|---:|---:|---:|']
for case in ['AM','FM']:
    z=json.loads((OUT/f'{case}.json').read_text())
    candidate=next(c for c in z['training'] if c['label']==z['selected'])
    lines.append(f"| {case} | {candidate['metadata']['nodes']} | {candidate['metadata']['regions']} | {z['training_score']:.5g} |")
lines += ['','## Отложенные динамические входы','',
          '| Модель | Вход | RMSE C, мкМ | Макс. ошибка C, мкМ | Пики LR/PWA | Макс. ошибка времени пика, с |',
          '|---|---|---:|---:|---:|---:|']
for case in ['AM','FM']:
    z=json.loads((OUT/f'{case}.json').read_text())
    for r in z['holdout']:
        if r['name'].startswith('fixed'):
            continue
        p=peaks_report[case][r['name']]
        lines.append(f"| {case} | {r['name']} | {r['rmse_C_h_I'][0]:.6g} | {r['max_abs_C']:.6g} | {p['reference_peaks']}/{p['pwa_peaks']} | {p['max_abs_peak_time_error_s']:.3g} |")
lines += ['','## Ограничения','',
          '- Число областей велико: это плотный стартовый baseline, не модель умеренной сложности.',
          '- AM: ошибка экстремумов при I=.638 около .027 мкМ; устойчивые циклы/покой на пяти фиксированных holdout распознаны, но бифуркационные границы не сертифицированы.',
          '- FM: при tau=50 с заметна ошибка фазы поздних пиков; нельзя утверждать полную точность динамического ответа.',
          '- На автономных траекториях накопление фазы даёт заметную raw RMSE даже при близких экстремумах и периодах; raw ошибки сохранены в JSON.',
          '- Метрики периода и задержек имеют дискретизацию по выходной сетке около .2 с. Нулевая разность периодов не означает математического совпадения.',
          '- Нестабильные циклы, SIC, события первого выхода, строгий C++ интегратор и NEST не проверены.',
          '- Аналитический поток проверен внутри региона; полные траектории пока получены DOP853.',
          '- Отложенные протоколы отделены от обучения этого эксперимента, но это не слепой внешний тест: blueprint и архивы известны.',
          '', 'Следующий шаг: адаптивное разбиение/редукция с ограничениями на границах циклов и динамических откликах; оценка стоимости переключений.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(OUT/'REPORT.md')
