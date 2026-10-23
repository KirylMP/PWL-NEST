"""Select by the original train objective, then diagnose delivered SIC in NEST.

The evaluation plan is frozen before fitting. It never enters the optimizer.
Both original AM candidates and their continuations are kept for review.
"""
import copy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark_nest import network_run, protocols as nest_stress, initial_state, compare
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.rest_protocols import stable_rest
from refine_server_candidates import source_rest

OUT=ROOT/'results/am_baseline_refit'


def main():
    plan=json.loads((OUT/'evaluation_plan.json').read_text())
    branches={}
    candidates={}
    for gate in ['cube','log']:
        attempts=[]
        for directory in [OUT/gate,OUT/'refined'/gate,OUT/'flow'/gate]:
            summary=json.loads((directory/'summary.json').read_text())[0]
            attempts.append((directory,summary))
        directory,summary=min(attempts,key=lambda item:item[1]['selected_train_loss'])
        name='AM_sqrt_tails__'+gate
        original=ROOT/'models/server_candidates'/f'{name}.json'
        selected=directory/summary['selected_model']
        branches[gate]=dict(baseline_train_loss=summary['baseline_train_loss'],
            selected_train_loss=summary['selected_train_loss'],selected_trial=summary['selected_trial'],
            model=str(selected.relative_to(ROOT)),trials=[dict(run=str(d.relative_to(OUT)),**r)
                for d,s in attempts for r in s['trials']],
            runs=[dict(directory=str(d.relative_to(OUT)),baseline_train_loss=s['baseline_train_loss'],
                       selected_train_loss=s['selected_train_loss'],
                       config=json.loads((d/'config.json').read_text())) for d,s in attempts],
            previous_tests=summary['selected_previous_tests']['aggregate'],
            historical_diagnostics=summary['selected_new_tests']['aggregate'])
        candidates[gate+'_baseline']=CalciumTransformPWA.load(original)
        candidates[gate+'_refit']=CalciumTransformPWA.load(selected)
    winner=min(branches,key=lambda k:branches[k]['selected_train_loss'])
    chosen=copy.deepcopy(candidates[winner+'_refit'])
    chosen.metadata.update(experiment_variant='am_baseline_refit_'+winner,
        verified_current_train_loss=branches[winner]['selected_train_loss'],
        current_objective='original RestObjective unchanged; no extra absolute timing or SIC terms',
        selected_by='minimum independently verified original training loss across both branches and their frozen incumbents',
        accepted=False,current_validation='results/am_baseline_refit/REPORT.md; all older inherited loss fields are historical')
    model_path=ROOT/'models/am_baseline_refit/selected_AM.json'
    model_path.parent.mkdir(parents=True,exist_ok=True)
    chosen.save(model_path)
    report=dict(selection=plan['selection'],branches=branches,selected_gate=winner,
        selected_model=str(model_path.relative_to(ROOT)),
        selected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        network_diagnostics={},diagnostic_aggregates={},implementation_check={})
    target=OUT/'nest';target.mkdir(exist_ok=True)
    figures={}
    for group,specs in [('new_own_rest',plan['new_diagnostics']),('known_NEST_stress',nest_stress('AM'))]:
        records=[];figures[group]={}
        for p in specs:
            # Standard methodology uses each candidate's own equilibrium.
            # Known NEST stress retains the previous common-LR-state setup.
            lr_initial=source_rest('AM',p['baseline'])
            if group=='known_NEST_stress':lr_initial=initial_state('AM',p['baseline'])
            truth,lr_timing=network_run('AM',chosen,lr_initial,p,'lr',10.,record=True)
            paths={'lr':truth};metrics={}
            for name,model in candidates.items():
                start=stable_rest(model,p['baseline'],lr_initial)[0] if group=='new_own_rest' else lr_initial
                data,timing=network_run('AM',model,start,p,'pwa',10.,record=True)
                np.testing.assert_allclose(data['t'],truth['t'],rtol=0,atol=1e-12)
                result=compare(truth,data,.01)
                result.update(initial=list(map(float,start)),timing=timing)
                metrics[name]=result;paths[name]=data
                np.savez_compressed(target/f'{group}_{p["name"]}_{name}.npz',**data)
            np.savez_compressed(target/f'{group}_{p["name"]}_lr.npz',**truth)
            records.append(dict(protocol=p,LR_initial=lr_initial,LR_timing=lr_timing,candidates=metrics))
            figures[group][p['name']]=paths
            print(json.dumps(dict(group=group,protocol=p['name'],
                SIC_RMSE={k:v['I_SIC']['rmse'] for k,v in metrics.items()})),flush=True)
        report['network_diagnostics'][group]=records
        report['diagnostic_aggregates'][group]={name:dict(
            mean_SIC_RMSE=float(np.mean([r['candidates'][name]['I_SIC']['rmse'] for r in records])),
            mean_C_RMSE=float(np.mean([r['candidates'][name]['C']['rmse'] for r in records])),
            SIC_peak_count_matches=sum(r['candidates'][name]['lr']['sic_peak_count']==r['candidates'][name]['pwa']['sic_peak_count'] for r in records))
            for name in candidates}
        fig,axes=plt.subplots(2,3,figsize=(13,6),sharex='col')
        names=[specs[k]['name'] for k in [0,2,4]]
        for col,name in enumerate(names):
            paths=figures[group][name]
            for label,key,color in [('LR','lr','black'),('NEST AM before','cube_baseline','tab:blue'),
                                    ('Selected gate before',winner+'_baseline','tab:green'),
                                    ('Selected by train',winner+'_refit','tab:orange')]:
                data=paths[key];s=slice(None,None,5)
                axes[0,col].plot(data['t'][s],data['C'][s],lw=1,color=color,label=label)
                axes[1,col].plot(data['t'][s],data['I_SIC'][s],lw=1,color=color,label=label)
            axes[0,col].set_title(name);axes[1,col].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Ca (µM)');axes[1,0].set_ylabel('Delivered SIC (pA)')
        axes[0,0].legend(fontsize=8);fig.suptitle('AM baseline method: '+group)
        fig.tight_layout();fig.savefig(OUT/f'{group}.png',dpi=160);fig.savefig(OUT/f'{group}.pdf');plt.close(fig)
    # Numerical implementation check for the selected model, not just the old seeds.
    from pwl_nest.calcium_coordinates import simulate_coordinates
    p=plan['new_diagnostics'][0];base=source_rest('AM',p['baseline'])
    rest=stable_rest(chosen,p['baseline'],base)[0]
    data,_=network_run('AM',chosen,rest,p,'pwa',10.,record=True)
    tr,xr,_=simulate_coordinates(chosen,rest,**{k:v for k,v in p.items() if k!='name'},dt=.01,
        rtol=2e-11,atol=2e-13,max_step=.01)
    reference=np.column_stack([np.interp(data['t'],tr,xr[:,d]) for d in range(3)])
    x=np.column_stack([data[k] for k in ['C','h','IP3']])
    errors=np.max(np.abs(x[:,:2]-reference[:,:2]),axis=0)
    np.testing.assert_allclose(x[:,:2],reference[:,:2],rtol=0,atol=3e-5)
    report['implementation_check']=dict(max_abs_C=float(errors[0]),max_abs_h=float(errors[1]),
        reference='independent high-accuracy DOP853 on first fresh input')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_report(report)


def write_report(report):
    branches=report['branches'];winner=report['selected_gate']
    model_path=ROOT/report['selected_model']
    errors=[report['implementation_check']['max_abs_C'],report['implementation_check']['max_abs_h']]
    for gate,record in branches.items():
        model=CalciumTransformPWA.load(ROOT/record['model'])
        model.metadata.update(verified_current_train_loss=record['selected_train_loss'],accepted=False,
            current_objective='original RestObjective unchanged; no extra absolute timing or SIC terms',
            current_validation='results/am_baseline_refit/REPORT.md; all older inherited loss fields are historical')
        path=model_path.parent/f'AM_sqrt_tails__{gate}.json';model.save(path)
        record['review_model']=str(path.relative_to(ROOT))
        record['review_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# AM: продолжение базового подбора','',
        'Критерий RestObjective сохранён: статические режимы/спектральные якоря,',
        'собственный покой, высоты/ширины/интервалы/число пиков, первая задержка,',
        'слабый поточечный остаток. Новые штрафы по SIC или абсолютным временам',
        'пиков не добавлялись. 21 параметр и 12 областей; пять перезапусков на ветвь,',
        'по 1500 вызовов остатка на каждую из двух LS-стадий и Powell.','',
        '| Координата h | До | После, независимый DOP train-loss | Улучшение |',
        '|---|---:|---:|---:|']
    for gate,r in branches.items():
        improvement=100*(1-r['selected_train_loss']/r['baseline_train_loss'])
        lines.append(f"| {gate} | {r['baseline_train_loss']:.6f} | {r['selected_train_loss']:.6f} | {improvement:.2f}% |")
    lines+=['',f'По обучению выбран `{winner}`: [{model_path.name}](../../{report["selected_model"]}).',
        'Диагностические входы не участвовали в выборе. Исходные модели и прежний',
        'default NEST-кандидат не перезаписаны. Унаследованные старые loss-поля',
        'в JSON исторические; текущие значения находятся в этой таблице и summary.json.','',
        '## Результат по выходу SIC','']
    for group in ['new_own_rest','known_NEST_stress']:
        before=report['diagnostic_aggregates'][group][winner+'_baseline']
        after=report['diagnostic_aggregates'][group][winner+'_refit']
        lines += [f"- {group}: средняя RMSE SIC {before['mean_SIC_RMSE']:.5f} → {after['mean_SIC_RMSE']:.5f} pA; совпадение числа SIC-пиков {before['SIC_peak_count_matches']}/5 → {after['SIC_peak_count_matches']}/5."]
    changes=[(values[winner+'_baseline']['mean_SIC_RMSE'],values[winner+'_refit']['mean_SIC_RMSE'])
             for values in report['diagnostic_aggregates'].values()]
    lines += ['', 'Уменьшение базового train-loss '+('сопровождается снижением средней SIC RMSE в обоих наборах.'
        if all(after<before for before,after in changes) else 'не дало общего улучшения SIC выбранной ветви.')]
    repaired=[r['protocol']['name'] for r in report['network_diagnostics']['known_NEST_stress']
        if r['candidates']['cube_refit']['lr']['sic_peak_count']==r['candidates']['cube_refit']['pwa']['sic_peak_count']
        and r['candidates']['cube_baseline']['lr']['sic_peak_count']!=r['candidates']['cube_baseline']['pwa']['sic_peak_count']]
    if repaired:lines += ['Ветка h³ восстановила правильное число SIC-пиков на: '+', '.join(repaired)+'.']
    remaining=5-report['diagnostic_aggregates']['new_own_rest'][winner+'_refit']['SIC_peak_count_matches']
    lines += [f'На {remaining} из пяти новых входов число SIC-пиков не совпадает.',
        'Общая приёмка не пройдена.','',
        '## Численное уточнение без изменения критерия','',
        'Первые два продолжения на ветвь использовали прежний RK4/.005 и',
        'глобальные bounds Powell. У cube оптимизатор нашёл точку, где',
        'ошибка C порядка 2e-7 μM меняла классификацию слабого пика на пороге',
        'prominence=.02; DOP853 отверг этот результат. Проверка сохранена в',
        'initial_numerical_audit.json. Затем проведены ещё два продолжения',
        'с RK4/.001 и локальным Powell: ±.01*max(1,abs(parameter)) внутри',
        'прежних допустимых bounds. Остаток, пороги обнаружения пиков,',
        'обучающие входы и число параметров не менялись. Все попытки',
        'выбираются только после полного DOP replay обучения.','',
        'Затем быстрые автономные attractor-статистики были заменены полными',
        'траекториями того же фиксированного временного окна. Для ускорения',
        'использован кешированный аналитический поток NEST-решателя; остаток',
        'не менялся. Проведено ещё по одному продолжению. На исходных моделях',
        'этот helper воспроизводит независимый train-loss с разницей около 1e-7',
        '(flow_helper_audit.json). Независимый DOP по-прежнему определяет выбор.','',
        '## SIC в NEST','',
        'Новые входы используют базовую методику: фон I=.3, скачок .35,',
        'собственный покой каждого кандидата и задержка входа 20 с. Набор',
        'заранее зафиксирован в evaluation_plan.json. Известный стресс-набор',
        'повторяет прежний benchmark: фон .2, скачок .7, общее начальное',
        'состояние LR. Он не был добавлен в обучение.','',
        '| Набор | Вход | SIC RMSE: прежний cube | SIC RMSE: выбранная координата до | SIC RMSE: после | Пики SIC LR/до/после |',
        '|---|---|---:|---:|---:|---:|']
    for group,records in report['network_diagnostics'].items():
        for row in records:
            a=row['candidates']['cube_baseline'];before=row['candidates'][winner+'_baseline'];b=row['candidates'][winner+'_refit']
            lines.append(f"| {group} | {row['protocol']['name']} | {a['I_SIC']['rmse']:.4f} | {before['I_SIC']['rmse']:.4f} | {b['I_SIC']['rmse']:.4f} | {b['lr']['sic_peak_count']}/{before['pwa']['sic_peak_count']}/{b['pwa']['sic_peak_count']} |")
    lines+=['','![Новые входы, собственный покой](new_own_rest.png)','',
        '![Прежние стресс-входы NEST](known_NEST_stress.png)','',
        f'Проверка выбранной реализации против собственного DOP853: max ошибка C={errors[0]:.3g} μM, h={errors[1]:.3g}.',
        'Скорость заново не измерялась. Уменьшение train-loss не означает',
        'автоматического прохождения всех SIC-протоколов или полной приёмки.','',
        '## Воспроизведение','',
        '```bash',
        'export PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1',
        'export MPLCONFIGDIR=/tmp/nest-matplotlib XDG_CACHE_HOME=/tmp',
        '.venv/bin/python scripts/refine_server_candidates.py --base-objective \\',
        '  --models AM_sqrt_tails__cube --output results/am_baseline_refit/cube \\',
        '  --budget 1500 --restarts 2 --seed 20261024',
        '# Аналогично для AM_sqrt_tails__log с отдельным output.',
        '# Дополнительные продолжения описаны в docs/am_baseline_refit.md.',
        '.venv/bin/python scripts/evaluate_am_baseline_refit.py',
        '```','',
        'Для нового запуска укажите новый каталог output; существующие',
        'результаты скрипт не перезаписывает.',
        'Полные попытки, checkpoints и независимые replay-метрики сохранены',
        'в cube/ и log/. Сырые NEST-траектории — в nest/.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(OUT/'REPORT.md',flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--report-only']:
        write_report(json.loads((OUT/'summary.json').read_text()))
    else:main()
