"""Render the saved common-span power experiment without fitting or selection."""
import json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/workspace/PWL-NEST/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/PWL-NEST/.cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pwl_nest.models import ROOT

OUT=ROOT/'results/calcium_powers';KINDS=['sqrt','square','cube']
LABELS={'sqrt':'C^0.5','square':'C^2','cube':'C^3'}
COLORS={'sqrt':'C1','square':'C2','cube':'C3'}


def main():
    fig,axes=plt.subplots(2,3,figsize=(15,7));allresults={}
    lines=['# Квадрат и куб C: сравнение с заново подобранным sqrt','',
        'Сохраняются logit(h) и исходные спектральные C-пороги. AM: 4 области/14 параметров, FM: 8/22. '
        'Ни дополнительных порогов, ни подбираемой нормировки нет. Для всех степеней phi(Cref)=0, phi(Cmax)=Cmax.',
        '', '**По обучению выбран sqrt для AM и FM. Все кандидаты остаются неприемлемыми как окончательные модели.**',
        '', '## Обучающий критерий','',
        'Сумма квадратов физических residual с прежними весами, без регуляризации; меньше лучше. '
        'Выбор сделан до диагностик. Числа AM и FM между собой не сравниваются.',
        '', '| Модель | √C | C² | C³ | Выбрано |','|---|---:|---:|---:|---|']
    for case in ['AM','FM']:
        selection=json.loads((OUT/f'{case}_selection.json').read_text());s=selection['training_scores']
        lines.append(f"| {case} | {s['sqrt']:.6g} | {s['square']:.6g} | {s['cube']:.6g} | {selection['selected']} |")
    for row,case in enumerate(['AM','FM']):
        data={kind:json.loads((OUT/kind/f'{case}.json').read_text()) for kind in KINDS};allresults[case]=data
        for kind,z in data.items():
            with np.load(OUT/kind/f'{case}_decay_tau_50.0_historical_diagnostic.npz') as trace:
                if kind=='sqrt':axes[row,0].plot(trace['t'],trace['lr'][:,0],label='Li-Rinzel',lw=1.5,color='C0')
                axes[row,0].plot(trace['t'],trace['candidate'][:,0],label=LABELS[kind],lw=1.1,color=COLORS[kind])
            byname={p['name']:p for p in z['training_protocols']+z['historical_diagnostics']}
            fixed=sorted([r for r in z['results'] if r['completed'] and r['name'].startswith('fixed')],key=lambda r:byname[r['name']]['baseline'])
            inputs=[byname[r['name']]['baseline'] for r in fixed]
            for col,metric in [(1,'period'),(2,'amplitude')]:
                if kind=='sqrt':axes[row,col].plot(inputs,[r['reference'][metric] if r['reference'][metric] is not None else np.nan for r in fixed],'o-',label='Li-Rinzel',lw=1.3,color='C0',ms=3)
                axes[row,col].plot(inputs,[r['candidate'][metric] if r['candidate'][metric] is not None else np.nan for r in fixed],'o-',label=LABELS[kind],lw=1.1,color=COLORS[kind],ms=3)
        axes[row,0].set(title=f'{case}: exponential IP3, tau=50 s',xlabel='time (s)',ylabel='C (uM)')
        axes[row,1].set(title=f'{case}: sampled periods',xlabel='IP3 (uM)',ylabel='period (s)')
        axes[row,2].set(title=f'{case}: sampled amplitudes',xlabel='IP3 (uM)',ylabel='C amplitude (uM)')
        for ax in axes[row]:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'calcium_powers_summary.png',dpi=160)
    lines+=['','## Экспоненциальный спад tau=50 с, диагностика','',
        '| Модель | Степень | RMSE C, мкМ | Пики LR/PWA | Завершено | В физических диапазонах |',
        '|---|---|---:|---:|---:|---:|']
    for case,data in allresults.items():
        for kind,z in data.items():
            rr=z['results'];r=next(r for r in rr if r['name']=='decay_tau_50.0');p=r['peaks']
            lines.append(f"| {case} | {LABELS[kind]} | {r['rmse'][0]:.6g} | {p['reference']}/{p['candidate']} | {sum(r['completed'] for r in rr)}/20 | {sum(r.get('physical',False) for r in rr)}/20 |")
    lines+=['','## Что получилось','',
        '- Квадрат и куб проиграли sqrt и на обучении, и на показанном спаде tau=50. Куб в значительной части тестов теряет нужные колебания.',
        '- AM при I=.638: квадрат увеличил размах до .0788 мкМ (sqrt .0343), но LR даёт .2253; период квадрата 13.6 с вместо 10.6. Это частичное улучшение амплитуды ценой периода, не приемлемый цикл.',
        '- FM при I=.7: LR период 22.6 с, sqrt 23.6, квадрат 59.0; куб приходит к покою.',
        '- Новый FM sqrt на tau=50 даёт все 5 пиков, RMSE C=.1097 мкМ. Старый sqrt давал 3 пика и RMSE .2753. Но новый подбор потерял правильный покой при I=1.13, а при I=.48 тоже наблюдаются лишние колебания. Утверждения о безусловном улучшении модели нет.',
        '- AM sqrt при h0=.9 достигает C=1.7757 мкМ выше Cmax≈1.6878. Квадрат в этом протоколе покидает положительную ветвь около 7.0888 с.',
        '- FM куб покидает положительную ветвь при tau=2 около 1.06167 с и в обоих протоколах скачков около .100783 с, ещё до первого скачка. Эти три прогона не считаются завершёнными.',
        '', '## Достоверность сравнения и ограничения','',
        'На вариант выполнено 20 общих протоколов: 7 train и 13 диагностических. Диагностики не входят в подбор, но они известны по прежнему эксперименту и не являются слепой внешней проверкой. Неудачные прогоны включены в знаменатель; значения не обрезались.',
        'Все степени получили одинаковый максимальный бюджет: три старта, уточнение двух, Powell. Это локальный поиск с конечными bounds. Нормировка задаёт общий диапазон, но не делает численную оптимизацию инвариантной к координатам. Не доказана невозможность других хороших C²/C³-моделей.',
        'Старые sqrt/cbrt/log-модели не перезаписаны. Новый sqrt подбирался при другой постоянной нормировке и служит контролем именно этого опыта. Улучшение относительно старого sqrt нельзя приписывать новой степени.',
        'PWA и аналитический региональный поток остаются в (u,v,I); восстановленное физическое поле не PWA. Для C²/C³ допустима только положительная обратная ветвь. Вблизи C=0 обратное отображение плохо обусловлено; выход за ветвь — реальная проблема этих кандидатов, а не разрешение продолжать через отрицательный кальций.',
        'Проверены сохранённые коэффициенты: соответствие старым C-порогам, непрерывность, региональные спектральные типы, exp(A*t) против DOP853 внутри региона, точность экспоненты I и source-Jacobian similarity. Метрики в structure.json. Модульные тесты включают новые степени и нормировку, обратимость и домен, chain rule, скачки, сериализацию и старые JSON без scale.',
        'RK4/.025 перепроверен с .005 и независимым DOP853: максимальная ошибка C относительно DOP менее .00031 мкМ на обучающих спадах, существенно меньше модельных ошибок. Выходы ветви перепроверены меньшим шагом в failure_checks.json.',
        'Глобальная инвариантность, бифуркационная сертификация и глобальная оптимальность не установлены. Полный событийный решатель, неустойчивые циклы, SIC, NEST и ускорение не проверялись.',
        '', 'Методы и команды: [docs/calcium_powers.md](../../docs/calcium_powers.md). Модели: models/calcium_powers/{sqrt,square,cube}; выбранные selected_AM.json и selected_FM.json используют sqrt.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(OUT/'REPORT.md')


if __name__=='__main__':main()
