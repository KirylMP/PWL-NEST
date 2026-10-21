"""Summarize verified experiments, keeping failed candidates visible."""
import os,json
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path('/workspace/PWL-NEST/.cache/matplotlib')))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX
from fit_positive_coordinates import OUT,VARIANTS


def load(case,variant):return json.loads((OUT/variant/f'{case}.json').read_text())


def record(data,name):return next(r for r in data['results'] if r['name']==name)


def number(x):return 'failed' if x is None else f'{x:.6g}'


def run():
    choices={case:json.loads((OUT/f'{case}_selection.json').read_text()) for case in ['AM','FM']}
    data={case:{v:load(case,v) for v in ['baseline']+VARIANTS} for case in choices}
    lines=['# Повторный подбор: положительность, шестые степени и граничные режимы','',
        '**Ни один selected-файл не означает полной приёмки.** Сравнение выполнено в преобразованных координатах; C не обрезается.',
        '',f'Физическая верхняя граница Cmax={C_MAX:.9g} мкМ. Чётная степень используется на положительной обратимой ветви. bounded_C задаёт C=Cmax·sigmoid(affine(u)), поэтому C ограничен самим readout.',
        '', '## Независимо пересчитанная калибровка','',
        'Добавлены проблемные фиксированные I и начальное h=.9. Вес входной waveform-ошибки уменьшен вдвое, штраф числа пиков усилен. Старые и новые значения loss между экспериментами напрямую не сравниваются.',
        '', '| Case | Вариант | Проверенный loss | Завершены | Физические | Известные фиксированные статусы совпали |',
        '|---|---|---:|---:|---:|---:|']
    totals=[0,0,0]
    for case,variants in data.items():
        for v,d in variants.items():
            rs=d['results'];complete=sum(r['completed'] for r in rs);physical=sum(r.get('physical',False) for r in rs)
            known=[r for r in rs if r.get('reference',{}).get('observed_status') in ['cycling','rest']]
            matched=sum(r['candidate']['observed_status']==r['reference']['observed_status'] for r in known)
            lines.append(f'| {case} | {v} | {number(d["verified_calibration_loss"])} | {complete}/{len(rs)} | {physical}/{len(rs)} | {matched}/{len(known)} |')
            totals=np.array(totals)+[len(rs),complete,physical]
    for case in choices:
        lines.extend(['',f'Выбор {case}: **{choices[case]["selected"]}**, только по независимо проверенной калибровке и структурным проверкам. Свежие тесты не участвовали.',''])
    lines.extend([f'Всего попыток: {totals[0]}, завершены {totals[1]}, в физических диапазонах {totals[2]}. Доля совпадений статусов учитывает лишь завершённые траектории с известным исходным статусом; незавершённые видны в отдельных столбцах.',
        '', '## Проверки вне калибровки','',
        '| Case | Вариант | tau50 RMSE C | tau50 пики LR/PWA | tau100 RMSE C (калибровка) | tau100 пики | h0=.9 Cmax (калибровка) |',
        '|---|---|---:|---|---:|---|---:|'])
    for case,variants in data.items():
        for v,d in variants.items():
            a,b,h=(record(d,k) for k in ['decay_tau_50.0','decay_tau_100.0','novel_initial_h_0.9'])
            def rmse(r):return number(r['rmse'][0]) if r['completed'] else 'failed'
            def peaks(r):return f'{r["peaks"]["reference"]}/{r["peaks"]["candidate"]}' if r['completed'] else 'failed'
            high=number(h['C_range'][1]) if h['completed'] else 'failed'
            lines.append(f'| {case} | {v} | {rmse(a)} | {peaks(a)} | {rmse(b)} | {peaks(b)} | {high} |')
    lines.extend(['','## Свежие проверки','',
        'Шесть новых протоколов на case: три фиксированных I, tau=7 и 65 с другими C0,h0 и новый нерегулярный вход. Стартовые модели и веса не подбирались по этим результатам.',
        '', '| Case | Вариант | Завершены | Физические | tau65 RMSE C | tau65 пики LR/PWA |',
        '|---|---|---:|---:|---:|---|'])
    for case,variants in data.items():
        for v,d in variants.items():
            fresh=[r for r in d['results'] if r['split']=='fresh_validation'];r=record(d,'fresh_tau_65.0')
            lines.append(f'| {case} | {v} | {sum(r["completed"] for r in fresh)}/6 | {sum(r.get("physical",False) for r in fresh)}/6 | {number(r["rmse"][0]) if r["completed"] else "failed"} | {str(r["peaks"]["reference"])+"/"+str(r["peaks"]["candidate"]) if r["completed"] else "failed"} |')
    lines.extend(['','## Численная и научная область выводов','',
        'Все калибровочные траектории, включая фиксированные I, пересчитаны DOP853 с координатными абсолютными допусками. Loss при незавершённой калибровке помечен failed, без подстановки успешного результата fitting helper. Подробные RK/.005 и RK/.001 расхождения сохранены в JSON каждого кандидата.',
        '',f'Дополнительное уточнение selected: AM score={choices["AM"].get("refined_selected_calibration_loss",float("nan")):.9g}, FM score={choices["FM"].get("refined_selected_calibration_loss",float("nan")):.9g}. У чувствительного FM tau100 уточнён DOP и RK до .00005; максимальное расхождение C около 1.9e-7 мкМ. Выбор не изменился. Полные данные: selected_RK_refinement.json.',
        '', 'C^6 сильно сжимает малые концентрации в латентной координате; исходные абсолютные допуски для неё недостаточны. Неудача этой ветви относится к данному локальному подбору и его численной реализации, а не доказывает невозможность модели при другой параметризации.',
        '', 'На I=.64 у исходной AM ответ в последнем окне ещё unresolved. Ему не приписана жёсткая метка покоя или цикла. Наблюдаемые режимы на конечных траекториях не сертифицируют бифуркации и отсутствие других аттракторов.',
        '', 'Аналитичность относится к постоянным региональным матрицам в латентных координатах. Полного событийного решателя, SIC/NEST и измерения ускорения нет. Новый локальный минимум и selected не означают выполнения всех требований blueprint.',
        '', 'Первый пилот с меньшим штрафом пиков и промежуточный AM-подбор сохранены в results/positive_coordinates_pilot и models/positive_coordinates_pilot. Их частичная проверка не входит в приведённые итоги.',
        '', '[Метод, формулы и команды](../../docs/positive_coordinates.md). [Графики](positive_coordinates_summary.png).'])
    am=data['AM'][choices['AM']['selected']];fm=data['FM'][choices['FM']['selected']]
    ac=record(am,'fixed_0.638');ah=record(am,'fresh_tau_65.0');fh=record(fm,'novel_initial_h_0.9')
    fb=record(data['FM']['baseline'],'novel_initial_h_0.9')
    conclusions=['','## Практический итог','',
        'Шестая степень C, шестая степень h и шестой корень C не обошли повторный подбор прежних координат по проверенному критерию этого локального опыта.',
        '',f'FM sqrt_tails: все 26 траекторий завершены в физических диапазонах на сетке выдачи. При h0=.9 максимум C уменьшился с {fb["C_range"][1]:.6g} до {fh["C_range"][1]:.6g} мкМ, ниже Cmax. Исходный LR даёт около 1.209 мкМ: ошибка высокого первого пика остаётся. Числа пиков на tau50/tau100 — 5/8, как у LR. Результат достигается при прежних 8 PWA-областях, без clipping и без дополнительных переключений на стыках readout.',
        '',f'AM {choices["AM"]["selected"]}: цикл при I=.638 восстановлен, размах {ac["candidate"]["amplitude"]:.6g} против {ac["reference"]["amplitude"]:.6g} мкМ у LR. Это ещё неточная амплитуда. Ошибка входных траекторий выросла; на новом tau65 — {ah["peaks"]["candidate"]} пиков против {ah["peaks"]["reference"]} у LR. Выбор по калибровке не превратился в универсальное улучшение. Другие кандидаты сохранены.',
        '', 'Физические диапазоны в таблицах, кроме аналитической гарантии C у ограниченных readout, проверены на сетке выдачи: .2 с для фиксированного I, .25 с для входных протоколов. Это не доказательство инвариантности полного физического состояния между отсчётами.']
    lines.extend(conclusions)
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(4,2,figsize=(13,15),constrained_layout=True)
    labels=['baseline']+VARIANTS
    for col,case in enumerate(['AM','FM']):
        ax=axes[0,col];scores=choices[case]['verified_calibration_scores']
        for i,v in enumerate(labels):
            if scores.get(v) is None:ax.text(i,.04,'failed',transform=ax.get_xaxis_transform(),rotation=90,ha='center',va='bottom',color='red')
            else:ax.bar(i,scores[v],color='tab:green' if v==choices[case]['selected'] else 'tab:blue')
        ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right');ax.set_yscale('log');ax.set_title(f'{case}: verified calibration loss')
        names=['fixed_0.638' if case=='AM' else 'novel_initial_h_0.9','decay_tau_100.0','fresh_tau_65.0']
        for row,name in enumerate(names,1):
            ax=axes[row,col];plotted=False
            variants=list(dict.fromkeys(['baseline',choices[case]['selected'],'bounded_C']))
            for v in variants:
                path=OUT/v/f'{case}_{name}.npz'
                if not path.exists():continue
                with np.load(path) as tr:
                    mask=(tr['t']>=1500) if 'fixed' in name else (tr['t']<=35 if row==1 else tr['t']>=0)
                    if not plotted:ax.plot(tr['t'][mask],tr['lr'][mask,0],color='black',lw=1.2,label='Li–Rinzel');plotted=True
                    ax.plot(tr['t'][mask],tr['candidate'][mask,0],lw=.85,label=v,alpha=.85)
            ax.set(title=f'{case}: {name}',xlabel='time (s)',ylabel='C (uM)')
            if case=='FM' and row==1:ax.axhline(C_MAX,color='red',ls='--',lw=.8,label='Cmax')
            ax.legend(fontsize=8)
    fig.savefig(OUT/'positive_coordinates_summary.png',dpi=150);plt.close(fig)
    print('Saved',OUT/'REPORT.md')


if __name__=='__main__':run()
