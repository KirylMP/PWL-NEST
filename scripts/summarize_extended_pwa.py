"""Summarize all standalone pilots, including numerical training audit."""
import json,os
os.environ.setdefault('MPLCONFIGDIR','/workspace/PWL-NEST/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/PWL-NEST/.cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA

OUT=ROOT/'results/extended_pwa';MODELS=ROOT/'models/extended_pwa'
LABELS={'baseline':'prior','control':'refit','sqrt_cube':'sqrt+h3','log_cube':'log+h3','hill_cube':'Hill+h3','input_one':'I:1','input_two':'I:2','tilted':'tilted'}


def main():
    fig,axes=plt.subplots(2,3,figsize=(16,7));allresults={};selections={}
    lines=['# Все предложенные направления: h³, I-пороги, наклонные границы','',
        'Проверены 8 вариантов AM и 8 FM, по 20 общих протоколов: всего 320 прогонов. '
        'Это отдельные ветви, не полная комбинационная сетка. Baseline сохраняет лучший по обучению кандидат предыдущего опыта с подвижными C-порогами. '
        'Control заново подбирает коэффициенты при той же фиксированной геометрии. Все новые ветви получили одинаковые максимальные бюджеты.',
        '', '**По независимо пересчитанному обучающему критерию выбраны input_two у AM и sqrt_cube у FM. '
        'Все модели остаются исследовательскими: полная приёмка не пройдена.**',
        '', '## Обучение после численной проверки','',
        'Общий residual: физические траектории/статистики и прежние мягкие source-I-якоря; регуляризация исключена. '
        'Динамическая часть пересчитана по независимому DOP853, при необходимости с более строгой опорной траекторией. '
        'Выбор не использует диагностические протоколы.',
        '', '| Модель | Baseline | Control | √C+h³ | ln(C)+h³ | Hill(C)+h³ | Один I-порог | Два I-порога | Наклонные |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row,case in enumerate(['AM','FM']):
        selection=json.loads((OUT/f'{case}_selection.json').read_text());selections[case]=selection
        variants=selection['variants'];scores=selection['verified_training_scores']
        lines.append('| '+case+' | '+' | '.join(f'{scores[v]:.6g}' for v in variants)+' |')
        allresults[case]={v:json.loads((OUT/v/f'{case}.json').read_text()) for v in variants}
        axes[row,0].bar(range(len(variants)),[scores[v] for v in variants],color=['0.6','C2','C1','C3','C4','C5','C6','C7'])
        axes[row,0].set_xticks(range(len(variants)),[LABELS[v] for v in variants],rotation=35,fontsize=8)
        axes[row,0].set(yscale='log',ylabel='verified training data-loss (log)',title=f'{case}: standalone pilots')
        shown=list(dict.fromkeys(['baseline','control',selection['selected']]))
        for col,tau,split in [(1,50,'historical_diagnostic'),(2,100,'train')]:
            for index,v in enumerate(shown):
                with np.load(OUT/v/f'{case}_decay_tau_{tau}.0_{split}.npz') as trace:
                    if index==0:axes[row,col].plot(trace['t'],trace['lr'][:,0],label='Li-Rinzel',color='C0',lw=1.5)
                    color={'baseline':'0.5','control':'C2'}.get(v,'C1')
                    axes[row,col].plot(trace['t'],trace['candidate'][:,0],label=v,color=color,lw=1.1,ls='--' if v=='baseline' else '-')
            axes[row,col].set(title=f'{case}: exponential IP3, tau={tau} s',xlabel='time (s)',ylabel='C (uM)');axes[row,col].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'extended_pwa_summary.png',dpi=160)
    lines+=['','## Геометрия и число активных параметров','',
        '| Вариант | Активные параметры AM/FM | Области AM/FM | Что меняется |','|---|---:|---:|---|',
        '| baseline | 0 в этом опыте | 4/8 | сохранённые ранее коэффициенты и C-пороги |',
        '| control, sqrt_cube, log_cube, hill_cube | 14/22 | 4/8 | коэффициенты при фиксированных физических C-порогах |',
        '| input_one | 16/24 | 8/16 | два новых коэффициента входного hinge |',
        '| input_two | 18/26 | 12/24 | четыре новых коэффициента двух input hinges |',
        '| tilted | 17/29 | 4/8 | 3/7 параметров упорядоченных наклонных C-границ |',
        '', 'Положения I-узлов фиксированы: AM .35,.64 мкМ, FM .53,1.10; в input_one используется верхний. '
        'Раньше baseline имел 17/29 подбираемых параметров с учётом C-порогов; в этой стадии они held geometry. '
        'Наклонные границы сохраняют порядок на I∈[0,1.3].',
        '', '## Независимые траектории','',
        '| Модель | Вариант | RMSE C tau50 | Пики tau50 LR/PWA | RMSE C tau100 | Пики tau100 LR/PWA | Завершено | В диапазонах | Совпали режимы на фиксированных I |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for case,data in allresults.items():
        for variant,z in data.items():
            rr=z['results'];by={r['name']:r for r in rr};a=by['decay_tau_50.0'];b=by['decay_tau_100.0'];fixed=[r for r in rr if 'candidate' in r]
            matches=sum(r['candidate']['observed_status']==r['reference']['observed_status'] for r in fixed)
            lines.append(f"| {case} | {variant} | {a['rmse'][0]:.6g} | {a['peaks']['reference']}/{a['peaks']['candidate']} | {b['rmse'][0]:.6g} | {b['peaks']['reference']}/{b['peaks']['candidate']} | {sum(r['completed'] for r in rr)}/20 | {sum(r.get('physical',False) for r in rr)}/20 | {matches}/{len(fixed)} |")
    lines+=['','## Выводы по направлениям','',
        '- h³: полезнее всего у FM с √C. Верифицированный train-loss ≈50.32 против control≈64.12 (около −21.5%). При I=1.13 — покой, как у LR; сохранённый baseline здесь колеблется. Это исправление даёт и новый control, но h³ дополнительно улучшает общий критерий.',
        '- FM √C+h³: на tau50 5 пиков, на tau100 8, как у LR. Максимальные сдвиги пиков .7 и 2.6 с. RMSE tau100=.1714 лучше baseline=.1955, но tau50=.1048 хуже baseline=.0789. Это не безусловное улучшение каждого протокола.',
        '- FM √C+h³ при h0=.9 снижает Cmax до 1.7287 мкМ против 1.8572 baseline, но физический Cmax≈1.6878 всё ещё превышен. Неверных ярлыков режима среди десяти фиксированных I в этой конечной проверке нет; это не сертификация бифуркаций.',
        '- У AM замена ln(h) на h³ при √C не улучшила train-loss и ухудшила ошибку h. При I=.638 размах .0560 вместо .0469 baseline, всё ещё далеко от .2253 у LR.',
        '- ln(C)+h³ проигрывает √C. У AM при h0=.9 C достигает 2.441 мкМ; у FM — 1.962. Само устранение h³ из формулы кальциевого тока не гарантирует удобное полное поле.',
        '- Hill(C)+h³ — самый слабый кандидат: большинство требуемых колебаний теряются. FM при h0=.9 достигает защитного C=20 мкМ около .258147 с; прогон незавершён. У остальных прошедших диапазоны траекторий плохая динамика всё равно не считается успехом.',
        '- I-пороги: у AM два узла дают лучший train-loss ≈1.111 против control≈1.391 (около −20.1%). На tau50 исчез лишний пик: 9 вместо 10. Но RMSE C немного вырос (.02017 против .01945), а нужный цикл при I=.638 исчез. Поэтому выбранный по train AM ещё нельзя принимать.',
        '- У FM один/два I-порога не обошли новый control по train-loss. Дополнительные параметры не дали самостоятельного выигрыша при выбранных фиксированных узлах и локальном бюджете.',
        '- Наклонные C-границы улучшили train у обоих case. У AM tau50 RMSE C=.01894 — немного лучше baseline/control, но лишний пик и малая амплитуда у границы остались. У FM tau100 RMSE=.1551 лучше остальных показанных вариантов, зато tau50=.1826 хуже baseline и √C+h³. Все эти модели сохранены для дальнейшего сравнения.',
        '- У FM наклоны очень малы: C-пороги между I=0 и 1.3 меняются максимум на .0000122 мкМ. При удалении новой геометрии с замороженными коэффициентами исходный helper-loss меняется 51.13→141.12; это чувствительный локальный результат, а не доказательство большого устойчивого выигрыша от наклонов. У AM соответствующая ablation даёт 1.225→12.35. Расчёт в geometry_ablation.json использует прежний RK4, не повторный fit и не независимую проверку аблированных траекторий; в выборе не участвует.',
        '', '## Численная достоверность','',
        'Сохранены прямое hinge-поле и постоянные региональные матрицы. Проверены непрерывность по I и наклонным C-границам, точная постоянная-I редукция для native statistics, source-Jacobian similarity, exp(A*t) внутри каждого типа области, порядок границ, положительные восстанавливающие I-наклоны и явные времена I-crossings. structure.json содержит фактические ошибки.',
        'У FM √C+h³ на train tau100 исходный RK4/.025 расходился с DOP853 до .05668 мкМ. Поэтому первоначальный loss 49.9895 не использован как окончательная численная оценка: train пересчитан по независимому DOP. Получено около 50.32, выбор остался √C+h³. Диагностические данные в этот пересчёт не входят.',
        'Для всех helpers с ошибкой более .001 мкМ уменьшен шаг до .001/.0002/.00005. При необходимости DOP повторён с rtol=2e-11,atol=2e-13,max_step=.03. Согласование уточнённых численных решений проверено до .0001 мкМ; точные результаты и чувствительность опоры в RK_refinement.json. Это не заявление о точности грубого ускорителя для всех параметров.',
        'Проверенные training scores записаны отдельно от optimizer_training_scores в *_selection.json и verified_training.json. Уточнённые опорные train-траектории сохранены отдельно; исходные сырые проверки не скрыты. Подбор по новым уточнённым траекториям заново не выполнялся.',
        '319 из 320 прогонов завершены; один runaway Hill/FM подтверждён более строгими настройками (failure_checks.json). Выходы за физический Cmax не обрезались. Предельные h-ветви и C>0 проверяются независимо от inverse/readout.',
        'Это локальный поиск с конечными bounds и исторически известными диагностическими протоколами. Совпадение десяти режимов — наблюдение на последних 400 с, не продолжение циклов и не доказательство отсутствия других аттракторов. Общая приёмка, глобальная инвариантность и оптимальность не доказаны.',
        '', '## Что остаётся аналитическим','',
        'В Y=(phi(C),psi(h),I) внутри области Ydot=A_jY+b_j с постоянными коэффициентами и точным экспоненциальным I. На чистой I-границе непрерывность допускает изменение I-столбца и смещения; C,h-столбцы совпадают. Наклонная граница меняет C- и I-столбцы согласованно и перемещается с I.',
        'I-crossings находятся через логарифм; пересечения состояния с наклонной или обычной C-границей требуют локализации первого корня. Полный аналитический событийный solver, неустойчивые циклы, SIC, NEST и измерение ускорения не реализованы.',
        '', 'Методы и воспроизведение: [docs/extended_pwa.md](../../docs/extended_pwa.md). '
        'Модели models/extended_pwa/{variant}/{AM,FM}.json; selected_AM.json=input_two, selected_FM.json=sqrt_cube по проверенному train-критерию. '
        'Все прежние кандидаты сохранены; эти selected-файлы не означают полной приёмки.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');geometry_plot();print(OUT/'REPORT.md')


def geometry_plot():
    fig,axes=plt.subplots(2,2,figsize=(12,7));grid=np.linspace(0,1.3,301)
    for row,case in enumerate(['AM','FM']):
        tilted=CalciumTransformPWA.load(MODELS/'tilted'/f'{case}.json')
        prior=CalciumTransformPWA.load(MODELS/'baseline'/f'{case}.json')
        for j in range(len(tilted.latent.thresholds)):
            vals=tilted.coordinate.inverse(tilted.latent.thresholds[j]+tilted.threshold_slopes[j]*(grid-.65))
            axes[row,0].plot(grid,vals,color=f'C{j}',label=f'C{j+1}')
            axes[row,0].axhline(prior.metadata['physical_thresholds_C'][j],ls='--',lw=.7,color=f'C{j}',alpha=.6)
        axes[row,0].set(title=f'{case}: tilted C thresholds (dashes: prior)',xlabel='I (uM)',ylabel='C threshold (uM)');axes[row,0].legend(fontsize=7,ncol=2)
        for variant in ['control','input_one','input_two']:
            model=CalciumTransformPWA.load(MODELS/variant/f'{case}.json');B,D=model.latent.matrix[:,2]
            qb=model.latent.spectral_coefficients()[3];dq=B*model.input_weights[1]-D*model.input_weights[0]
            restoring=qb*grid+np.maximum(0,grid[:,None]-model.input_thresholds)@dq
            axes[row,1].plot(grid,restoring/restoring[-1],label=variant)
        axes[row,1].set(title=f'{case}: normalized restoring input',xlabel='I (uM)',ylabel='R(I)/R(1.3)');axes[row,1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'geometry_and_inputs.png',dpi=160)


if __name__=='__main__':main()
