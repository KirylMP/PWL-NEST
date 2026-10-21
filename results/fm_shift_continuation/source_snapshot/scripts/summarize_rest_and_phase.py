"""Plots and explicit limits of phase-aligned response comparisons."""
import json,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/workspace/PWL-NEST/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/PWL-NEST/.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_calcium_coordinates import dump
from check_rest_and_phase import OUT,MODES


def find(data,mode,name):return next(r for r in data['results'] if r['mode']==mode and r['protocol']==name)


def plot_case(case,data):
    name='decay_tau_100.0';fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for col,mode in enumerate(['original','own_rest']):
        r=find(data,mode,name);g=r['global_shift']
        with np.load(OUT/f'{case}_{mode}_{name}.npz') as trace:
            t,lr,pwa=trace['t'],trace['lr'][:,0],trace['pwa'][:,0]
        axes[0,col].plot(t,lr,'k',lw=1.2,label='Li–Rinzel')
        axes[0,col].plot(t,pwa,lw=.9,label='sqrt_tails')
        axes[0,col].set(title=f'{case}: {mode}, unchanged input times',ylabel='C (uM)')
        axes[0,col].legend(fontsize=9)
        mask=(t>=g['fit_window'][0])&(t<=g['fit_window'][1])
        axes[1,col].plot(t[mask],lr[mask],'k',lw=1.2,label='Li–Rinzel')
        axes[1,col].plot(t[mask],np.interp(t[mask]+g['shift_s'],t,pwa),lw=.9,label='PWA(t+delta)')
        axes[1,col].set(title=f'Diagnostic translation delta={g["shift_s"]:.2f} s; RMSE {g["raw_eval_rmse"]:.3f} → {g["aligned_eval_rmse"]:.3f}',ylabel='C (uM)')
        axes[1,col].legend(fontsize=9)
    for mode in MODES:
        p=find(data,mode,name)['peaks']
        if 'lags_s' in p:axes[2,0].plot(p['LR_times'],p['lags_s'],'o-',ms=4,label=mode)
    axes[2,0].axhline(0,color='black',lw=.5);axes[2,0].set(title='Input-locked peak timing',ylabel='PWA peak time − LR peak time (s)')
    axes[2,0].legend(fontsize=9)
    p=find(data,'own_rest',name)['peaks']
    for label in ['LR','PWA']:
        widths=p[f'{label}_widths_half_prominence_s'];axes[2,1].plot(np.arange(1,len(widths)+1),widths,'o-',label=label)
    axes[2,1].set(title='Own rest: width at half prominence',ylabel='peak width (s)',xlabel='peak number')
    axes[2,1].legend(fontsize=9)
    for ax in axes[:2].ravel():ax.set_xlabel('time after input onset (s)')
    axes[2,0].set_xlabel('LR peak time (s)')
    fig.savefig(OUT/f'{case}_phase_comparison.png',dpi=150);plt.close(fig)


def run():
    both={case:json.loads((OUT/f'{case}.json').read_text()) for case in ['AM','FM']}
    lines=['# Инициализация из покоя и фазовая ошибка sqrt_tails','',
        'Модели и пороги заморожены. Начальные состояния и параметры не оптимизировались. У каждого case выполнены 18 парных сравнений: шесть одинаковых входов × три способа инициализации.',
        '', '## Инициализация','',
        '- original: прежние одинаковые физические C0,h0 из протокола;',
        '- own_rest: каждая модель стартует из собственного устойчивого равновесия при фоновом I;',
        '- common_LR_rest: обе стартуют из равновесия LR при фоновом I. Для PWA это контроль ошибки положения покоя.',
        '', 'Равновесия решены из уравнений и подтверждены численным выходом из прежнего начального состояния за 1200 с. Проверены невязки и отрицательные действительные части собственных значений. Скачок входа меняет только I; C,h не сбрасываются.',
        '', '| Case | Фоновый I | C покоя LR/PWA | h покоя LR/PWA |',
        '|---|---:|---|---|']
    for case,d in both.items():
        a,b=d['equilibrium_preparation']['LR'],d['equilibrium_preparation']['PWA']
        lines.append(f'| {case} | {a[2]} | {a[0]:.7g} / {b[0]:.7g} | {a[1]:.7g} / {b[1]:.7g} |')
    lines.extend(['','## Как разделена ошибка','',
        'Исходные задержки и RMSE считаются при одинаковых физических временах входа. Затем диагностически подбирается единственный сдвиг delta в пределах ±5 с: PWA(t+delta) сравнивается с LR(t). Положительная delta означает, что PWA запаздывает.',
        '', 'Все delta сравниваются на одном фиксированном окне: от onset+5 до конца−5 с. Поэтому нельзя улучшить score простым удалением плохого участка при изменении delta. Это окно исключает ранний первый пик в части протоколов; полный RMSE и задержка первого пика сохранены отдельно. Амплитуда и масштаб времени не подгоняются.',
        '', 'Дополнительно delta подбирается только по первой половине фиксированного окна и без изменения применяется ко второй. Для периодических входов во второй половине может быть преимущественно покой — это видно по самому протоколу, и такой результат не доказывает точность спайков.',
        '', 'Отдельная морфологическая диагностика центрирует каждый соответствующий пик в окне ±3 с без масштабирования амплитуды или времени. Это помогает оценить форму одного импульса, но **не заменяет проверку задержек и причинного ответа на вход**. При разных числах пиков соответствие по порядку не навязывается.',
        '', '## Все протоколы','',
        '| Case | Старт | Вход | Пики LR/PWA | Полный RMSE C | RMSE общего окна | После сдвига | delta, с | RMSE позднего окна до/после раннего сдвига |',
        '|---|---|---|---|---:|---:|---:|---:|---|'])
    for case,d in both.items():
        for r in d['results']:
            p,g,e=r['peaks'],r['global_shift'],r['early_shift_late_test']
            flag='*' if g['at_search_boundary'] else ''
            lines.append(f'| {case} | {r["mode"]} | {r["protocol"]} | {p["counts"][0]}/{p["counts"][1]} | {r["raw_full_RMSE"][0]:.5g} | {g["raw_eval_rmse"]:.5g} | {g["aligned_eval_rmse"]:.5g} | {g["shift_s"]:.3f}{flag} | {e["raw_eval_rmse"]:.5g} / {e["aligned_eval_rmse"]:.5g} |')
    lines.extend(['','Звёздочка у delta: минимум упёрся в заранее заданный предел ±5 с. Это ограниченный поиск сдвига, а не доказательство глобального оптимума.'])
    fm=find(both['FM'],'own_rest','decay_tau_100.0');old=find(both['FM'],'original','decay_tau_100.0')
    p,g=fm['peaks'],fm['global_shift'];wlr=np.median(p['LR_widths_half_prominence_s'][1:]);wpwa=np.median(p['PWA_widths_half_prominence_s'][1:])
    periodic=find(both['FM'],'own_rest','periodic_jumps_then_stop')['global_shift']
    lines.extend(['','## Вывод по FM','',
        f'После старта из собственного покоя на tau100 обе модели дают {p["counts"][0]} пиков. Сдвиг delta={g["shift_s"]:.4g} с уменьшает RMSE общего окна с {g["raw_eval_rmse"]:.5g} до {g["aligned_eval_rmse"]:.5g} мкМ. Фазовый компонент ошибки существенен.',
        '',f'Но задержки отдельных пиков идут от {min(p["lags_s"]):.3g} до {max(p["lags_s"]):.3g} с: это не постоянное рассогласование часов. У прежнего старта один сдвиг уменьшает RMSE лишь с {old["global_shift"]["raw_eval_rmse"]:.5g} до {old["global_shift"]["aligned_eval_rmse"]:.5g}. Подготовка покоя сама по себе не устраняет расхождение.',
        '',f'Высоты последующих пиков близки: средняя абсолютная ошибка без первого — {p["peak_amplitude_MAE_after_first"]:.5g} мкМ. При этом медианная ширина без первого пика: LR {wlr:.4g} с, PWA {wpwa:.4g} с ({100*(1-wpwa/wlr):.1f}% уже). Даже после индивидуальной центровки импульсов RMSE в ±3 с около пика — {p["peak_centered_shape"]["RMSE"]:.5g} мкМ. Остаётся ошибка формы, не только положения пиков.',
        '',f'Для периодических входов оптимальная delta имеет другой знак: {periodic["shift_s"]:.4g} с. Один постоянный временной сдвиг для всех входов не подходит. Следующий подбор разумно оценивать отдельно по высотам, ширинам, интервалам и задержкам, сохраняя проверку полного ответа.',
        '', '## Область вывода','',
        'Это диагностика фиксированных sqrt_tails AM/FM, а не новый подбор или изменение критериев приёмки. Совпадение числа пиков и улучшение после выравнивания не означает выполнения blueprint. Аналитическое региональное поле не изменено. Полный событийный решатель и NEST не исследовались.',
        '', '[График FM](FM_phase_comparison.png). [График AM](AM_phase_comparison.png). Полные peak times, высоты, ширины, SHA256 моделей и неизменённые траектории находятся рядом в JSON/NPZ.',
        '', 'Воспроизведение:', '', '```bash',
        'OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_rest_and_phase.py',
        'OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/summarize_rest_and_phase.py',
        '.venv/bin/python -m unittest discover -s tests -q','```'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for case,d in both.items():plot_case(case,d)
    print('Saved',OUT/'REPORT.md')


if __name__=='__main__':run()
