"""Summarize saved two-stage coordinate search and exact impulsive IP3."""
import json,os
os.environ.setdefault('MPLCONFIGDIR','/workspace/PWL-NEST/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/PWL-NEST/.cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pwl_nest.models import ROOT
from pwl_nest.impulsive_ip3 import exponential_ip3

OUT=ROOT/'results/coordinate_families'
KINDS=['sqrt','exp','log','inverse','inverse_sqrt']
LABELS={'sqrt':'sqrt(C)','exp':'exp(C)','log':'ln(C)','inverse':'-1/C','inverse_sqrt':'-1/sqrt(C)'}


def main():
    fig,axes=plt.subplots(2,3,figsize=(15,7));allresults={}
    lines=['# Другие координаты C и h; импульсная запись IP3','',
        'AM: 4 области/14 параметров; FM: 8/22. Физические спектральные C-пороги прежние. '
        'Пять C-преобразований при logit(h), затем log/exp(h) при лучшем C: семь моделей на case. '
        'Это последовательный пилот, не полная сетка C×h.',
        '', '**Лучший C по обучению — sqrt у обеих моделей. Для h выбраны log у AM и exp у FM. '
        'Ни один вариант не прошёл полную приёмку.**',
        '', '## Этап C: общий обучающий data-loss','',
        'Без регуляризации; меньше лучше. Для каждой строки те же протоколы, физические веса и максимальные бюджеты.',
        '', '| Модель | √C | exp(C) | ln(C) | −1/C | −1/√C |','|---|---:|---:|---:|---:|---:|']
    selections={case:json.loads((OUT/f'{case}_selection.json').read_text()) for case in ['AM','FM']}
    for case,s in selections.items():lines.append('| '+case+' | '+' | '.join(f"{s['training_scores'][k]:.6g}" for k in KINDS)+' |')
    lines+=['','## Этап h при выбранном √C','',
        '| Модель | logit(h) | ln(h) | exp(h) | Выбор по обучению |','|---|---:|---:|---:|---|']
    for row,(case,s) in enumerate(selections.items()):
        scores=s['training_scores'];ck=s['C_stage_selected'];hvariants=[ck,f'{ck}_h_log',f'{ck}_h_exp']
        lines.append('| '+case+' | '+' | '.join(f'{scores[k]:.6g}' for k in hvariants)+f" | {s['selected']} |")
        axes[row,0].bar(range(5),[scores[k] for k in KINDS],color=['C1','C2','C3','C4','C5'])
        axes[row,0].set_xticks(range(5),[LABELS[k] for k in KINDS],rotation=20,fontsize=8)
        axes[row,0].set(yscale='log',ylabel='training data-loss (log scale)',title=f'{case}: C transform, logit(h)')
        axes[row,1].bar(range(3),[scores[k] for k in hvariants],color=['C1','C2','C3'])
        axes[row,1].set_xticks(range(3),['logit(h)','ln(h)','exp(h)'])
        axes[row,1].set(ylabel='training data-loss',title=f'{case}: h transform, {LABELS[ck]}')
        variants=list(dict.fromkeys([ck,s['selected']]))
        for index,variant in enumerate(variants):
            with np.load(OUT/variant/f'{case}_decay_tau_50.0_historical_diagnostic.npz') as tr:
                if index==0:axes[row,2].plot(tr['t'],tr['lr'][:,0],label='Li-Rinzel',color='C0',lw=1.5)
                axes[row,2].plot(tr['t'],tr['candidate'][:,0],label=variant,color=f'C{index+1}',lw=1.1)
        with np.load(OUT/f'{case}_incumbent_tau50.npz') as tr:
            axes[row,2].plot(tr['t'],tr['candidate'][:,0],label='previous sqrt',color='0.4',lw=.9,ls='--')
        axes[row,2].set(title=f'{case}: exponential IP3, tau=50 s',xlabel='time (s)',ylabel='C (uM)');axes[row,2].legend(fontsize=7)
        allresults[case]={v:json.loads((OUT/v/f'{case}.json').read_text()) for v in s['variants']}
    fig.tight_layout();fig.savefig(OUT/'coordinate_families_summary.png',dpi=160)
    lines+=['','## Общие траектории: tau=50 и все 20 протоколов','',
        '| Модель | Вариант | RMSE C, мкМ | Пики LR/PWA | Завершено | В физических диапазонах | Режим покой/цикл совпал на фиксированных I |',
        '|---|---|---:|---:|---:|---:|---:|']
    for case,data in allresults.items():
        for variant,z in data.items():
            rr=z['results'];r=next(r for r in rr if r['name']=='decay_tau_50.0');p=r['peaks']
            fixed=[r for r in rr if 'candidate' in r]
            matches=sum(r['candidate']['observed_status']==r['reference']['observed_status'] for r in fixed)
            lines.append(f"| {case} | {variant} | {r['rmse'][0]:.6g} | {p['reference']}/{p['candidate']} | {sum(r['completed'] for r in rr)}/20 | {sum(r.get('physical',False) for r in rr)}/20 | {matches}/{len(fixed)} |")
    old=json.loads((OUT/'incumbents.json').read_text())
    lines+=['','## Сравнение с прежним сохранённым sqrt','',
        '| Модель | Прежний reported train-loss | Прежний RMSE C tau50 | Новый выбранный RMSE C tau50 |',
        '|---|---:|---:|---:|']
    for case,z in old.items():
        previous=next(r for r in z['results'] if r['name']=='decay_tau_50.0')
        selected=allresults[case][selections[case]['selected']]
        current=next(r for r in selected['results'] if r['name']=='decay_tau_50.0')
        lines.append(f"| {case} | {z['reported_previous_training_loss']:.6g} | {previous['rmse'][0]:.6g} | {current['rmse'][0]:.6g} |")
    lines+=['','Прежние модели не перезаписаны. Их выбранные траектории повторены текущим кодом (incumbents.json); старый train-loss взят из предыдущего fit, не пересчитан. '
        'FM exp(h) улучшает новый контроль, но проигрывает сохранённому предыдущему sqrt на tau50 и по reported train-loss. '
        'Старый sqrt тоже имеет неверные режимы. Новый контроль не воспроизвёл старый минимум: локальная оптимизация чувствительна к малым численным изменениям. '
        'Текущие selected_*.json означают выбор внутри этого семивариантного пилота, не лучший вариант во всей истории.',
        '', '## Конкретные выводы и дефекты','',
        '- Другие C-преобразования не превзошли sqrt по обучению. Обратные степени существенно усиливают чувствительность обратного вывода при приближении к их верхней латентной границе.',
        '- AM sqrt+ln(h): loss 1.834 против 2.181; все 20 протоколов завершены в физических диапазонах. При h0=.9 Cmax=1.298 мкМ вместо 1.778 у нового logit-контроля. На tau50 ошибка h снизилась с .01297 до .00683, а C почти не изменилась (.02300→.02281).',
        '- AM sqrt+ln(h) всё ещё даёт 10 пиков вместо 9 на tau50. При I=.638 размах всего .0213 мкМ против .2253 у LR; изменение h этот дефект не исправило.',
        '- FM sqrt+exp(h): loss 136.6 против 157.1; все 20 траекторий в физических диапазонах. Но на tau50 6 пиков вместо 5, при I=1.13 сохраняется большой цикл вместо покоя.',
        '- FM inverse выходит выше Cmax в 10 протоколах, FM inverse_sqrt достигает 2.737 мкМ при h0=.9. AM inverse достигает 3.149, AM log — 2.717.',
        '- AM inverse_sqrt при h0=.9 достигает защитного C=20 мкМ около 1.38516 с. Это единственный незавершённый прогон нового набора; перепроверен строгими настройками в failure_checks.json.',
        '- Ни одна log/exp(h)-модель этого набора не пересекла h=0 или h=1 в проверенных протоколах. Guards обеих границ дополнительно проверены искусственными тестами; глобальная инвариантность не доказана.',
        '', '## Численные проверки','',
        'Тестируются обратимость и производные всех новых координат, точное LR chain rule и source-Jacobian similarity, непрерывность и исходные региональные типы сохранённых моделей, exp(A*t) против DOP853 в области, точные I-скачки, старые JSON и native API. structure.json содержит фактические ошибки.',
        'У FM inverse, tau100 ускоритель RK4/.025 имел ошибку C .00712 мкМ против DOP853. Дополнительное уменьшение шага до .001 и .0002 дало .0000131 и .00000265 мкМ (RK_refinement.json). Достоверность этого грубого training helper ограничена; итоговые метрики взяты из DOP853. Остальные ошибки ускорителя C на train меньше .00045 мкМ.',
        'Число совпавших режимов — наблюдение на конечном окне последних 400 с, не продолжение циклов и не сертификация бифуркаций. Диагностики известны по прежним опытам, внешнего слепого теста нет. Bounds, нормировка и локальный бюджет ограничивают выводы. Полный аналитический событийный solver, неустойчивые циклы, SIC, NEST и ускорение не выполнены.',
        '', '## I через производную','',
        'Точно: J=Idot+(I−Ib)/tau=Σak δ(t−tk). J хранится как события, I — экспоненциальный след. На событии I+=ak, C,h сохраняются. '
        'Обычная координата V=−(I−Ib)/tau остаётся экспоненциальной и не добавляет свободы. '
        'Оператор, выделяющий δ, не является обратимой координатой состояния: между событиями J=0 не хранит память о прошлом входе.',
        'Точная суперпозиция и оба предела скачков сверены с 28 сохранёнными протоколами (impulsive_ip3.json). В тестах также проверена распределительная формула в слабой форме и матричный поток после замены I на V. График импульсов показывает их площади, а не конечную высоту δ.',
        '', 'Методы: [coordinate_families.md](../../docs/coordinate_families.md), [impulsive_ip3.md](../../docs/impulsive_ip3.md). '
        'Модели: models/coordinate_families/{variant}/{AM,FM}.json, selected_AM.json=√C+ln(h), selected_FM.json=√C+exp(h).']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    impulse_plot()
    print(OUT/'REPORT.md')


def impulse_plot():
    events=[(10.,.3),(30.,.3),(50.,.3),(70.,.3),(90.,.3)]
    grid=np.unique(np.r_[np.linspace(0,180,1801),[t for t,a in events]])
    values=exponential_ip3(grid,.3,20.,.3,events)
    fig,ax=plt.subplots(2,1,figsize=(10,5),sharex=True)
    ax[0].plot(grid,values,color='C0');ax[0].axhline(.3,color='0.6',ls='--',lw=.8)
    ax[0].set(ylabel='I (uM)',title='Exact exponential trace and its Dirac input, tau=20 s')
    ax[1].stem([t for t,a in events],[a for t,a in events],linefmt='C1-',markerfmt='C1o',basefmt='k-')
    ax[1].set(xlabel='time (s)',ylabel='impulse area (uM)',ylim=(0,.4),title='J = dI/dt + (I - Ib)/tau; stems mark delta areas')
    fig.tight_layout();fig.savefig(OUT/'impulsive_ip3.png',dpi=160)


if __name__=='__main__':main()
