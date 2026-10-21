"""Report the compact-threshold experiment, including failed checks."""
import json,os
os.environ.setdefault('MPLCONFIGDIR','/workspace/PWL-NEST/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/PWL-NEST/.cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pwl_nest.models import ROOT

OUT=ROOT/'results/calcium_strips'
fig,axes=plt.subplots(2,3,figsize=(14,7))
lines=['# Компактная PWA на исходной спектральной эвристике','',
       'Только исходные спектральные C-пороги. Физические координаты (C,h,I), непрерывное полное поле, точная строка IP3.',
       '**Обе модели — исследовательские кандидаты, приёмка не пройдена.**',
       '','| Модель | C-пороги | Области | Свободные параметры | Физические диапазоны на прогонах |',
       '|---|---:|---:|---:|---:|']
reports={}
for row,case in enumerate(['AM','FM']):
    z=json.loads((OUT/f'{case}.json').read_text());reports[case]=z
    results=z['results'];metadata=z['model_metadata']
    completed=[r for r in results if r['completed']]
    physical=sum(r['physical'] for r in completed)
    lines.append(f"| {case} | {metadata['regions']-1} | {metadata['regions']} | {metadata['free_parameters']} | {physical}/{len(results)} |")
    byname={p['name']:p for p in z['protocols']}
    fixed=sorted([r for r in completed if r['name'].startswith('fixed')],key=lambda r:byname[r['name']]['baseline'])
    inputs=[byname[r['name']]['baseline'] for r in fixed]
    for label,key,color in [('Li-Rinzel','reference','C0'),('compact PWA','candidate','C1')]:
        axes[row,0].plot(inputs,[r[key]['amplitude'] for r in fixed],'o-',label=label,color=color,ms=4)
        axes[row,1].plot(inputs,[r[key]['period'] if r[key]['period'] is not None else np.nan for r in fixed],'o-',label=label,color=color,ms=4)
    axes[row,0].set(title=f'{case}: sampled cycle amplitude',xlabel='IP3 (uM)',ylabel='C amplitude (uM)')
    axes[row,1].set(title=f'{case}: sampled cycle period',xlabel='IP3 (uM)',ylabel='period (s)')
    with np.load(OUT/f'{case}_decay_tau_50.0_holdout.npz') as d:
        axes[row,2].plot(d['t'],d['lr'][:,0],label='Li-Rinzel',lw=1.3)
        axes[row,2].plot(d['t'],d['candidate'][:,0],label='compact PWA',lw=1.1,ls='--')
    axes[row,2].set(title=f'{case}: exponential IP3, tau=50 s',xlabel='time (s)',ylabel='C (uM)')
    for ax in axes[row]:ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'compact_summary.png',dpi=160)
lines += ['','## Отложенные экспоненциальные входы','',
          '| Модель | tau, с | RMSE C, мкМ | Пики LR / PWA | Физические C,h |',
          '|---|---:|---:|---:|---|']
for case,z in reports.items():
    for r in z['results']:
        if r['split']=='holdout' and r['name'].startswith('decay') and r['completed']:
            p=r['peaks'];lines.append(f"| {case} | {r['name'].split('_')[-1]} | {r['rmse'][0]:.5g} | {p['reference']}/{p['candidate']} | {'да' if r['physical'] else 'нет'} |")
lines += ['','## Что получилось','',
          '- AM: последовательность stable node → stable focus → unstable focus → stable focus сохранена. Периоды основных циклов близки, но сырой waveform RMSE на длинных автономных прогонах велик из-за накопления фазы.',
          '- AM при I=.638: Li–Rinzel продолжает колебаться, кандидат в данном протоколе приходит к покою. Это не доказательство отсутствия иных циклов/бассейнов.',
          '- AM на периодическом импульсном входе даёт лишний пик (12 вместо 11).',
          '- FM: все восемь региональных типов сохранены, но периоды существенно завышены, есть пропуски/лишние пики и выходы C,h за физические диапазоны.',
          '- Тип равновесия и положение C-порога сами по себе не определяют амплитуду, период, бассейн или отклик при меняющемся IP3.',
          '', '## Диагностика и границы вывода','',
          'Выполнены две локальные попытки: совместный подбор v1 и staged v2. Во второй жёстко сохранялись спектральные подтипы, сначала настраивались автономные характеристики, затем физическая и динамическая точность. Пороги не двигались и не добавлялись.',
          'В выбранном классе h- и I-столбцы общие для всех полос, а спектр кальциевого якобиана внутри каждой полосы постоянен по I. В legacy коэффициенты могли зависеть от I — это существенная потеря свободы при переходе к строгой 3D аффинности.',
          'Структурные проверки в structure.json: непрерывность всего поля, региональный аналитический поток и точная экспонента I. Полные траектории проверялись DOP853; фит использовал exact-flow статистики при постоянном I и отдельно проверенный RK4 для динамических входов.',
          'Нет заявления о невозможности дальнейшей подгонки этого класса, глобальном оптимуме или необходимости тысяч областей. Глобальный поиск не выполнен.',
          'Train/holdout разделены по протоколам, однако результаты holdout v1 уже были известны перед изменением процедуры v2. Это повторно используемые диагностики, а не слепой тест.',
          'Скачки изменяют только физический I, C,h сохраняются. Отрицательные значения не обрезались; все нарушения показаны в JSON.',
          'Метрики периодов имеют выходную дискретизацию около .2 с; динамические траектории v2 сохраняются с шагом около .1 с.',
          '', 'Полный событийный решатель, неустойчивые циклы, сертификация бифуркационных границ, SIC и NEST не реализованы этим экспериментом.',
          '', '## Файлы','',
          '- models/calcium_strips/AM.json и FM.json: компактные коэффициенты текущего v2.',
          '- joint_v1/: предыдущая попытка сохранена отдельно, включая модели и траектории.',
          '- AM_refinement.json, FM_refinement.json: процедура v2 и история подбора.',
          '- AM.json, FM.json в results/calcium_strips/: все независимые сравнения и физические нарушения.',
          '- docs/calcium_strips.md: уравнения, ограничения и команды воспроизведения.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(OUT/'REPORT.md')
