"""Compare all C transforms and the procedure-matched identity control."""
import json,os
os.environ.setdefault('MPLCONFIGDIR','/workspace/PWL-NEST/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/PWL-NEST/.cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pwl_nest.models import ROOT

OUT=ROOT/'results/calcium_coordinates';KINDS=['identity','sqrt','cbrt','log']
COLORS={'identity':'0.5','sqrt':'C1','cbrt':'C2','log':'C3'}
fig,axes=plt.subplots(2,3,figsize=(15,7));allresults={}
lines=['# Преобразования кальция: квадратный корень, кубический корень, логарифм','',
       'Сравниваются PWA в (phi(C),logit(h),I), с фиксированными аффинными центрированием и нормировкой координат. '
       'Исходные физические C-пороги не менялись: AM 4 области/14 параметров, FM 8/22. '
       'Identity — контроль с той же процедурой нового подбора.',
       '', '**Все варианты остаются исследовательскими кандидатами, полная приёмка не пройдена.**',
       '', '## Выбор только по обучающим данным','',
       '| Модель | identity | sqrt | cbrt | log | Выбранное нелинейное преобразование |',
       '|---|---:|---:|---:|---:|---|']
for case in ['AM','FM']:
    selection=json.loads((OUT/f'{case}_selection.json').read_text());scores=selection['training_scores']
    lines.append(f"| {case} | {scores['identity']:.5g} | {scores['sqrt']:.5g} | {scores['cbrt']:.5g} | {scores['log']:.5g} | {selection['selected']} |")
for row,case in enumerate(['AM','FM']):
    data={kind:json.loads((OUT/kind/f'{case}.json').read_text()) for kind in KINDS};allresults[case]=data
    reference_plotted=False
    for kind,z in data.items():
        with np.load(OUT/kind/f'{case}_decay_tau_50.0_historical_diagnostic.npz') as trace:
            if not reference_plotted:
                axes[row,0].plot(trace['t'],trace['lr'][:,0],label='Li-Rinzel',lw=1.5,color='C0');reference_plotted=True
            axes[row,0].plot(trace['t'],trace['candidate'][:,0],label=kind,lw=1.1,ls='--' if kind!='sqrt' else '-',color=COLORS[kind],alpha=.9)
        byname={p['name']:p for p in z['training_protocols']+z['historical_diagnostics']}
        fixed=sorted([r for r in z['results'] if r['completed'] and r['name'].startswith('fixed')],key=lambda r:byname[r['name']]['baseline'])
        inputs=[byname[r['name']]['baseline'] for r in fixed]
        if kind=='identity':
            axes[row,1].plot(inputs,[r['reference']['period'] if r['reference']['period'] is not None else np.nan for r in fixed],'o-',label='Li-Rinzel',lw=1.3,color='C0',ms=3)
            axes[row,2].plot(inputs,[r['reference']['amplitude'] for r in fixed],'o-',label='Li-Rinzel',lw=1.3,color='C0',ms=3)
        axes[row,1].plot(inputs,[r['candidate']['period'] if r['candidate']['period'] is not None else np.nan for r in fixed],'o-',label=kind,lw=1.1,color=COLORS[kind],ms=3)
        axes[row,2].plot(inputs,[r['candidate']['amplitude'] for r in fixed],'o-',label=kind,lw=1.1,color=COLORS[kind],ms=3)
    axes[row,0].set(title=f'{case}: exponential IP3, tau=50 s',xlabel='time (s)',ylabel='C (uM)');axes[row,0].legend(fontsize=7)
    axes[row,1].set(title=f'{case}: sampled periods',xlabel='IP3 (uM)',ylabel='period (s)');axes[row,1].legend(fontsize=7)
    axes[row,2].set(title=f'{case}: sampled amplitudes',xlabel='IP3 (uM)',ylabel='C amplitude (uM)');axes[row,2].legend(fontsize=7)
fig.tight_layout();fig.savefig(OUT/'calcium_coordinates_summary.png',dpi=160)
lines += ['','## Отложенный экспоненциальный спад, tau=50 с','',
          '| Модель | Преобразование | RMSE C, мкМ | Пики LR/PWA | Физические диапазоны на всех протоколах |',
          '|---|---|---:|---:|---:|']
for case,data in allresults.items():
    for kind,z in data.items():
        r=next(r for r in z['results'] if r['name']=='decay_tau_50.0')
        physical=sum(r.get('physical',False) for r in z['results']);p=r['peaks']
        lines.append(f"| {case} | {kind} | {r['rmse'][0]:.6g} | {p['reference']}/{p['candidate']} | {physical}/{len(z['results'])} |")
lines += ['','## Конкретные улучшения sqrt','',
          '- AM при I=.638: контроль приходит к покою, sqrt-кандидат колеблется. Период LR 10.6 с, PWA 11.2 с, но размах C составляет лишь .034 мкМ против .225 у LR. Исправлен только наблюдаемый ярлык режима; точность цикла всё ещё недостаточна.',
          '- FM при I=1.13: контроль продолжает колебаться, sqrt-кандидат, как и LR, приходит к покою. Положение покоя .446 мкМ против .471 у LR остаётся неточным.',
          '- FM при I=.7: период LR 22.6 с, identity-контроль 30.2 с, sqrt 27.6 с.',
          '', '## Что не исправилось','',
          '- На tau=50 AM sqrt даёт 10 пиков вместо 9 и более высокий RMSE, чем identity. Улучшение обучающего критерия не означает улучшение каждого отклика.',
          '- На tau=50 FM sqrt даёт 3 пика вместо 5. Разница фаз и пропущенные ответы всё ещё велики.',
          '- AM при начальном h=.9: sqrt достигает C≈1.773 мкМ, выше физической границы 2/1.185≈1.688. cbrt≈1.980, log≈2.712 мкМ. Эти выходы не обрезались.',
          '- FM log при h0=.9 достиг защитного предела C=20 мкМ около t=1.343 с; прогон завершён как неудачный, не как успешная траектория.',
          '- Новые начальные C=.03,.15 не использовались при подборе. Они и остальные диагностические результаты сохранены в JSON.',
          '', '## Проверки и корректная интерпретация','',
          'sqrt выбран для обеих моделей по общему обучающему data-loss до просмотра диагностик. Identity получил тот же бюджет и те же residual scales. Выполнен локальный поиск из нескольких стартов, не глобальная оптимизация.',
          'Степенные координаты проверяют положительную ветвь. Нельзя возводить отрицательный sqrt-state в квадрат и объявлять кальций положительным. Native guard проверяет каждый принятый внутренний RK-шаг; независимый DOP853 имеет событие выхода из области. Это численная диагностика, не сертифицированный алгоритм касаний и первого выхода.',
          'Обратное преобразование root — квадрат/куб аффинной функции u, для log — экспонента. Интегрируется PWA в u,v,I; физическое поле не PWA. Третья строка и IP3-скачки остались точными; региональная матричная экспонента и source-Jacobian similarity проверены.',
          'Проверки source chain rule, обратимости и домена, положительных ветвей, сохранения C,h при скачке, сериализации и совместимости старого native API входят в тесты. Независимые проверки RK4/.005 и DOP853 сохранены; расхождения малы относительно модельных ошибок.',
          'Физические верхние границы C и точность динамики не гарантируются одной положительностью C. Глобальная инвариантность не доказана. Особенно важно: affine dependence on logit(h) не ограничивает влияние h→1 без дополнительных ограничений области.',
          'Исторические диагностики известны по предыдущим попыткам; новые начальные C не участвуют в обучении. Нет заявления о слепой внешней валидации, глобальном оптимуме или превосходстве sqrt над всеми возможными преобразованиями.',
          'Полный аналитический событийный решатель, неустойчивые циклы, SIC, NEST и ускорение не проверялись.',
          '', 'Методы и команды: docs/calcium_coordinates.md. Все варианты сохранены в models/calcium_coordinates/{identity,sqrt,cbrt,log}; selected_AM.json и selected_FM.json содержат sqrt. Предыдущие модели не изменены.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(OUT/'REPORT.md')
