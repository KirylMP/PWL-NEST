"""Render the saved matched fixed/movable-threshold continuation experiment."""
import json,os
os.environ.setdefault('MPLCONFIGDIR','/workspace/PWL-NEST/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/PWL-NEST/.cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pwl_nest.models import ROOT

OUT=ROOT/'results/movable_thresholds';MODELS=ROOT/'models/movable_thresholds'
VARIANTS=['baseline','fixed_refit','moving_joint']
COLORS={'baseline':'0.5','fixed_refit':'C2','moving_joint':'C1'}


def main():
    fig,axes=plt.subplots(2,3,figsize=(15,7));data={}
    lines=['# Подбор C-порогов вокруг спектральной эвристики','',
        'Раньше пороги фиксировались по исходным сменам типа равновесия LR. В этом опыте они дополнительно подбираются, '
        'порядок и число областей сохраняются. AM: 4 области, коэффициенты 14 + пороги 3 = 17 параметров; '
        'FM: 8 областей, 22 + 7 = 29. Вход I остаётся точным, региональные матрицы постоянны.',
        '', 'AM использует √C+ln(h), FM — прежний сохранённый √C+logit(h). '
        'Baseline не меняется; fixed_refit уточняет коэффициенты с фиксированными порогами; moving_joint уточняет коэффициенты и пороги. '
        'Две ветви имеют одинаковый максимальный бюджет, два старта и общий residual. '
        'Регуляризация исключена из таблицы и выбора; прежние мягкие спектральные I-якоря входят в data-loss.',
        '', '**Совместный подбор выиграл по обучению у обоих контролей, но полная приёмка всё ещё не пройдена.**',
        '', '## Обучение','',
        '| Модель | Baseline | Повторный fit, пороги фиксированы | Коэффициенты + пороги | Улучшение относительно fixed_refit |',
        '|---|---:|---:|---:|---:|']
    for row,case in enumerate(['AM','FM']):
        selection=json.loads((OUT/f'{case}_selection.json').read_text());s=selection['training_scores']
        improvement=100*(1-s['moving_joint']/s['fixed_refit'])
        lines.append(f"| {case} | {s['baseline']:.6g} | {s['fixed_refit']:.6g} | {s['moving_joint']:.6g} | {improvement:.1f}% |")
        data[case]={v:json.loads((OUT/v/f'{case}.json').read_text()) for v in VARIANTS}
        moved=json.loads((MODELS/'moving_joint'/f'{case}.json').read_text())
        ref=np.array(moved['metadata']['threshold_reference_C']);fit=np.array(moved['thresholds_C']);indices=np.arange(1,len(ref)+1)
        axes[row,0].plot(indices,ref,'o--',color='0.5',label='spectral seed',ms=4)
        axes[row,0].plot(indices,fit,'o-',color='C1',label='fitted',ms=4)
        axes[row,0].set(title=f'{case}: physical C thresholds',xlabel='threshold index',ylabel='C (uM)',xticks=indices)
        axes[row,0].legend(fontsize=8)
        for col,tau,split in [(1,50,'historical_diagnostic'),(2,100,'train')]:
            for variant in VARIANTS:
                with np.load(OUT/variant/f'{case}_decay_tau_{tau}.0_{split}.npz') as tr:
                    if variant=='baseline':axes[row,col].plot(tr['t'],tr['lr'][:,0],label='Li-Rinzel',color='C0',lw=1.5)
                    axes[row,col].plot(tr['t'],tr['candidate'][:,0],label=variant,color=COLORS[variant],lw=1.0,ls='--' if variant=='baseline' else '-')
            axes[row,col].set(title=f'{case}: exponential IP3, tau={tau} s',xlabel='time (s)',ylabel='C (uM)')
            axes[row,col].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'movable_thresholds_summary.png',dpi=160)
    lines+=['','## Физические C-пороги, мкМ','',
        '| Модель | Порог | Спектральный старт | После подбора | Сдвиг |','|---|---:|---:|---:|---:|']
    for case in ['AM','FM']:
        m=json.loads((MODELS/'moving_joint'/f'{case}.json').read_text())
        for j,(old,new) in enumerate(zip(m['metadata']['threshold_reference_C'],m['thresholds_C']),1):
            lines.append(f'| {case} | {j} | {old:.9f} | {new:.9f} | {new-old:+.9f} |')
    lines+=['','## Экспоненциальные спады и физические диапазоны','',
        '| Модель | Вариант | RMSE C tau50 | Пики tau50 LR/PWA | RMSE C tau100 | Пики tau100 LR/PWA | В диапазонах / всего |',
        '|---|---|---:|---:|---:|---:|---:|']
    for case,variants in data.items():
        for variant,z in variants.items():
            by={r['name']:r for r in z['results']};a=by['decay_tau_50.0'];b=by['decay_tau_100.0']
            lines.append(f"| {case} | {variant} | {a['rmse'][0]:.6g} | {a['peaks']['reference']}/{a['peaks']['candidate']} | {b['rmse'][0]:.6g} | {b['peaks']['reference']}/{b['peaks']['candidate']} | {sum(r.get('physical',False) for r in by.values())}/20 |")
    lines+=['','## Что улучшилось','',
        '- AM: loss 1.397 против fixed_refit 1.647 (−15.2%). На tau50 RMSE C .01944 мкМ против .02189 у fixed_refit и .02281 у baseline. На tau20 и tau100 ошибка C тоже ниже.',
        '- AM при I=.638: размах вырос до .04694 мкМ против .00235 у fixed_refit и .02133 у baseline, но LR даёт .22534. Период 11.3 с против LR 10.6. Цикл всё ещё сильно неточен.',
        '- FM: loss 71.14 против fixed_refit 107.51 (−33.8%). На tau50 RMSE C .07892 против .20089 у fixed_refit и .10974 у baseline. Пиков 5, как у LR.',
        '- FM на tau100 даёт все 8 пиков вместо прежних 6 и 5 у fixed_refit; RMSE C .19549 против .27222 baseline и .26921 fixed_refit. Максимальный сдвиг пиков остаётся 3.4 с.',
        '', '## Дефекты и ограничения','',
        '- AM по-прежнему даёт 10 пиков вместо 9 на tau50. Рост размаха у границы I=.638 не означает, что бифуркация воспроизведена.',
        '- FM moving_joint при I=1.13 продолжает колебаться вместо покоя LR. Fixed_refit здесь приходит к покою. Улучшение по train и по некоторым спадам ухудшает отдельный режим.',
        '- FM moving_joint при h0=.9 достигает C=1.8572 мкМ выше Cmax≈1.6878; fixed_refit — 1.7462, baseline в пределах. Поэтому новая FM не заменяет полностью все прежние кандидаты.',
        '- Все 120 прогонов завершены. AM: 20/20 в физических диапазонах для каждого варианта; FM: baseline 20/20, fixed_refit и moving_joint по 19/20. Обнаруженные превышения не обрезались.',
        '- Пороги теперь подбираемые: исходная спектральная эвристика задаёт старт и мягкие I-якоря. Утверждения о точном совпадении новых границ со сменами типа исходного равновесия больше нет.',
        '- Это локальный поиск с дополнительными 3/7 параметрами. Отдельные ранние остановки и конечный бюджет не исключают других решений. Повторный fixed fit и joint fit могут попадать в разные бассейны; глобальное превосходство класса не доказано.',
        '', '## Проверки','',
        'Baseline score пересчитан текущим common residual и совпал с прежним: AM 1.834470, FM 120.559670. Диагностики не входили в подбор, но они известны по прежним опытам, слепого теста нет.',
        'Новый mapping использует относительные положительные промежутки, log-множители ±ln2 и фиксированные endpoints Ceq(I=0),Ceq(I=1.3). Нулевой сдвиг точно возвращает исходные пороги, порядок не нарушается. Подробнее в docs/movable_thresholds.md.',
        'Проверены число параметров, фактические пороги JSON, непрерывность, региональные спектральные типы и exp(A*t) внутри области против DOP853. Параметризация порогов и неизменность региональных спектров при их движении покрыты тестами. Метрики сохранены в structure.json.',
        'Полные траектории проверены независимым DOP853. Ускоритель RK4/.025 перепроверен шагом .005; максимальная ошибка C против DOP менее .00094 мкМ (у FM moving_joint). Это численная проверка, не сертификация первого выхода или глобальной инвариантности.',
        'Аналитичность регионального движения при экспоненциальном I сохранена. Полный аналитический событийный решатель, неустойчивые циклы, SIC, NEST и ускорение ещё не реализованы.',
        '', 'Модели: models/movable_thresholds/{baseline,fixed_refit,moving_joint}; selected_AM.json и selected_FM.json выбирают moving_joint по обучению. Старые кандидаты сохранены. '
        'Метод и команды: [docs/movable_thresholds.md](../../docs/movable_thresholds.md).']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(OUT/'REPORT.md')


if __name__=='__main__':main()
