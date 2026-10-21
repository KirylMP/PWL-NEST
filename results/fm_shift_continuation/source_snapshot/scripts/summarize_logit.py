"""Compare physical PWA, calcium-matched logit control and refitted logit PWA."""
import json,os
os.environ.setdefault('MPLCONFIGDIR','/workspace/PWL-NEST/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/PWL-NEST/.cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pwl_nest.models import ROOT

OUT=ROOT/'results/logit_pwa'
reports={case:json.loads((OUT/f'{case}.json').read_text()) for case in ['AM','FM']}
fig,axes=plt.subplots(2,3,figsize=(15,7))
lines=['# PWA в координатах (C, logit(h), I)','',
       'Состояние хранится и интегрируется как Y=(C,z,I), h=sigmoid(z) восстанавливается при выдаче физических данных. '
       'Пороги C и число областей прежние: AM 4 / 14 параметров, FM 8 / 22 параметра.',
       '', '**Полная динамическая приёмка не пройдена.** Преобразование обеспечивает диапазон h, '
       'но само по себе не добавляет свободы кальциевой динамике при прежних C-порогах.',
       '', '## Контроль и сравнение при tau=50 с','',
       '| Модель | Вариант | RMSE C, мкМ | RMSE h |', '|---|---|---:|---:|']
for row,(case,z) in enumerate(reports.items()):
    r=next(r for r in z['results'] if r['name']=='decay_tau_50.0')
    if not r['completed']:raise AssertionError(r)
    for label,data in [('физическая PWA',r['physical_PWA_baseline']),('logit, контроль с прежним C(t)',r['calcium_matched_control']),('logit, новый подбор',r)]:
        lines.append(f"| {case} | {label} | {data['rmse'][0]:.6g} | {data['rmse'][1]:.6g} |")
    with np.load(OUT/f'{case}_decay_tau_50.0_historical_diagnostic.npz') as new, \
         np.load(ROOT/f'results/calcium_strips/{case}_decay_tau_50.0_holdout.npz') as old:
        for col,d,label in [(0,0,'C (uM)'),(1,1,'h')]:
            axes[row,col].plot(new['t'],new['lr'][:,d],label='Li-Rinzel',lw=1.4)
            axes[row,col].plot(old['t'],old['candidate'][:,d],label='physical PWA',lw=1.1,color='0.5')
            if d==1:axes[row,col].plot(new['t'],new['control'][:,d],label='logit control',lw=1,ls=':',color='C2')
            axes[row,col].plot(new['t'],new['candidate'][:,d],label='logit refit',lw=1.1,ls='--',color='C1')
            axes[row,col].set(title=f'{case}: tau=50 s, {label}',xlabel='time (s)',ylabel=label)
            axes[row,col].legend(fontsize=7)
    byname={p['name']:p for p in z['training_protocols']+z['historical_diagnostics']}
    fixed=sorted([r for r in z['results'] if r['completed'] and r['name'].startswith('fixed')],key=lambda r:byname[r['name']]['baseline'])
    inputs=[byname[r['name']]['baseline'] for r in fixed]
    for label,key,color in [('Li-Rinzel','reference','C0'),('physical PWA','physical_PWA_baseline','0.5'),('logit refit','candidate','C1')]:
        periods=[]
        for r in fixed:
            p=r[key]['candidate']['period'] if key=='physical_PWA_baseline' else r[key]['period']
            periods.append(p if p is not None else np.nan)
        axes[row,2].plot(inputs,periods,'o-',label=label,color=color,ms=4)
    axes[row,2].set(title=f'{case}: sampled periods',xlabel='IP3 (uM)',ylabel='period (s)');axes[row,2].legend(fontsize=7)
fig.tight_layout();fig.savefig(OUT/'logit_summary.png',dpi=160)
lines += ['','## Итог','',
          '- На всех 18 проверенных протоколах каждой новой модели C,h оставались в заданных физических диапазонах. Для C это наблюдение, не доказательство глобальной инвариантности. Ограничение 0<h<1 обеспечивается обратной логистикой при конечном z; численного насыщения до 0/1 не наблюдалось.',
          '- AM: ошибка C при tau=50 почти прежняя; ошибка h выросла. Сохранились потеря цикла в протоколе I=.638 и лишний пик на периодическом входе.',
          '- FM: часть автономных периодов стала ближе (I=.7: LR 22.6 с, physical PWA 38.2 с, logit PWA 30.4 с). Динамический ответ при tau=50 остаётся плохим: 3 пика вместо 5 и RMSE C выросла.',
          '- FM: RMSE h на tau=50 уменьшилась; контроль показывает, что почти такое же улучшение h получается уже от нового readout без изменения кальциевой динамики.',
          '- Новые начальные h=.65,.9 не использовались для подбора. Ошибки количества пиков сохранились. Однозначного выигрыша на этих диагностических протоколах нет.',
          '', '## Почему это не расширило класс C(t)','',
          'Внутри полосы Cdot=a_j*C+B*z+cI*I+b_j, zdot=d_j*C+D*z+dI*I+e_j. '
          'При B!=0 исключение z даёт Cddot=(a_j+D)Cdot-(a_j*D-B*d_j)C+'
          '(B*e_j-D*b_j)+(B*dI-D*cI)I+cI*Idot. Это тот же класс замкнутой кальциевой динамики, '
          'что в физической PWA. Разные результаты подбора отражают другой компромисс ошибок и локальные оптимумы, '
          'а не появление новых степеней свободы для C.',
          'Контроль сохранён отдельно: при начальном h=.8 совпадают замкнутые коэффициенты и начальное Cdot, '
          'поэтому совпадает C(t), включая экспоненциальный вход и IP3-скачки. '
          'Он не является точным нелинейным push-forward старой PWA: это контроль с новой физической интерпретацией второй координаты.',
          '', '## Проверки и ограничения','',
          '- Исходное LR-поле преобразовано по правилу цепочки: zdot=fh/[h(1-h)]. Простая замена имени h на z не использовалась для нового field-fit.',
          '- Непрерывность, региональная матричная экспонента, точная релаксация I, round-trip координат, source-Jacobian similarity и IP3-скачки проверены.',
          '- Внутренний DOP853 решает систему в Y. Обратная логистика применяется при выдаче h. C и I не требуют обратного преобразования.',
          '- Физическое поле после pullback не PWA. Это отдельная, явно выбранная пользователем ветка blueprint.',
          '- Полный аналитический событийный решатель, неустойчивые циклы, SIC и NEST не реализованы. Ускорение не измерялось.',
          '- Исторические диагностики многократно использовались в предыдущих экспериментах; новые initial-h протоколы не участвовали в текущем подборе. Нет заявления о внешней слепой проверке или глобальном оптимуме.',
          '- Преобразование не определено при h=0,1. Входные значения не обрезаются, такие начальные состояния отвергаются.',
          '', 'Команды и уравнения: docs/logit_pwa.md. Модели: models/logit_pwa. Старые модели не изменены.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(OUT/'REPORT.md')
