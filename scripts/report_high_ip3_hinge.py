"""Render saved metrics and timing without rerunning the experiment."""
import json
import csv
import hashlib
import platform
import shutil
import subprocess
import numpy as np
from pwl_nest.models import ROOT

OUT=ROOT/'results/high_ip3_hinge'


def main():
    d=json.loads((OUT/'evaluation.json').read_text())
    checks={}
    for name in ['full_AM','AM_2C','AM_2C_2I']:
        a=np.load(OUT/'traces'/f'default_jump_{name}.npz')
        b=np.load(OUT/'traces'/f'default_jump_{name}_upper.npz')
        checks[name]=all(np.array_equal(a[k],b[k]) for k in a.files)
        assert checks[name],name
    d['low_input_exact_trajectory_preservation']=checks
    # Paired resampling by repetition; descriptive uncertainty on this host only.
    rng=np.random.default_rng(20261031)
    lr=np.array([r['timing']['run_seconds'] for r in d['timing']['LR']['samples']])
    indices=rng.integers(0,len(lr),size=(5000,len(lr)))
    for name,r in d['timing'].items():
        assert len(r['samples'])==7
        assert all(s['topology_sha256']==r['warmup']['topology_sha256'] for s in r['samples'])
        assert all(s['terminal_C']==r['samples'][0]['terminal_C'] for s in r['samples'])
        x=np.array([s['timing']['run_seconds'] for s in r['samples']])
        boot=np.median(lr[indices],axis=1)/np.median(x[indices],axis=1)
        r['speedup_bootstrap_95_interval']=np.quantile(boot,[.025,.975]).tolist()
    with (OUT/'timings.csv').open('w') as f:
        fields=['model','repeat','build_seconds','prepare_seconds','run_seconds','cleanup_seconds']
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for name,r in d['timing'].items():
            for s in r['samples']:
                writer.writerow(dict(model=name,repeat=s['repeat'],**s['timing']))
    for p in list(d['source_sha256'])+['scripts/report_high_ip3_hinge.py','tests/test_nest_adapter.py']:
        d['source_sha256'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        dest=OUT/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/p,dest)
    cpu=next((s.split(':',1)[1].strip() for s in open('/proc/cpuinfo') if s.startswith('model name')),'unknown')
    d['environment']=dict(python=platform.python_version(),numpy=np.__version__,cpu=cpu,
         base_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         module_sha256=hashlib.sha256((ROOT/'.build/pwl_astro_module.so').read_bytes()).hexdigest())
    (OUT/'evaluation.json').write_text(json.dumps(d,indent=2)+'\n')
    rows=d['diagnostics']['original_input']['results']
    lines=['# Один дополнительный порог IP3: качество и время сети', '',
      'Непрерывная поправка `w*(IP3-k)+` добавлена к трём AM-моделям. '
      'Перебирался k в [1.3, 1.9]; во всех случаях выбран k=1.3. '
      'Ниже порога исходное поле сохраняется точно. '
      'Старые модели и значения по умолчанию не изменены.', '',
      '## Подбор и независимая сетевая проверка', '',
      'Только два веса и положение порога подбирались по автономным '
      'равновесиям LR при IP3 = 2, 4, 8, 16, 32, 48 мкМ. '
      'Минимизировалась ошибка равновесий в латентных координатах. '
      'Сетевые траектории, спайки и времена не входили в подбор. '
      'Исходная матрица динамики в каждой C-области не менялась. '
      'Метод не оптимизирует переходные процессы и глобальная '
      'оптимальность модели не заявляется.', '',
      'Сеть — официальный небольшой feedforward-пример NEST 3.10: '
      '10 исходных и 10 целевых нейронов, 5 астроцитов. '
      'Проверены исходный скачок IP3=0.2 мкМ и стандартный 0.0002 мкМ. '
      'Оба опыта длятся 30 с при шаге 0.1 мс, один поток. '
      'Для моделей совпадают топология и пресинаптические спайки. '
      'Все модели стартуют из одного физического состояния. '
      'Переобучения после сетевой проверки не было.', '',
      '| Модель | C в конце, мкМ | SIC в конце, пА | RMSE SIC, пА | Спайков | Сохранённый эффект SIC |',
      '|---|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        lines.append(f"| {name} | {r['astro_Ca_astro_tail_mean']:.4f} | "
          f"{r['post_I_SIC_tail_mean']:.3f} | {r['post_I_SIC_RMSE']:.3f} | "
          f"{r['post_spikes']} | {r['effect_fraction_vs_LR']:.1%} |")
    lines += ['', 'Конечный уровень — среднее последних 5 с. '
      'Эффект SIC — число дополнительных спайков относительно no_SIC, '
      'делённое на соответствующую прибавку у LR. RMSE рассчитана по '
      'доставленному суммарному SIC всех целевых нейронов за весь опыт.', '',
      'У полной AM RMSE SIC уменьшилась с 26.53 до 6.24 пА, '
      'у AM 2C+2I с 33.00 до 10.59 пА. У AM 2C ошибка, наоборот, '
      'выросла с 8.92 до 15.90 пА: лучшее плато не гарантирует лучшего '
      'переходного процесса. Все три исправленных варианта остаются '
      'в физических диапазонах C,h на записанных сетевых траекториях. '
      'Это наблюдение для данных запусков, не гарантия для любых входов.', '',
      'При малом входе для каждой исходной/исправленной пары все '
      'сохранённые массивы траекторий и спайков совпадают побитово.', '',
      '![Исходный вход](original_input.png)', '',
      '![Малый вход](default_jump.png)', '',
      '## Время симуляции', '',
      'Тот же исходный вход, 30 с модельного времени. Все регистраторы '
      'отключены. Для каждой модели исключён один полный прогрев; '
      'затем семь повторов в перемешанном порядке. Измеряется nest.Run. '
      'Построение, Prepare и Cleanup записаны отдельно и не включены. '
      'RNG и топология фиксированы. Это время всей небольшой сети, '
      'а не изолированного астроцитарного решателя.', '',
      '| Модель | Медиана, с | Минимум–максимум, с | Ускорение к LR |',
      '|---|---:|---:|---:|']
    for name,r in d['timing'].items():
        lines.append(f"| {name} | {r['median_seconds']:.3f} | "
                     f"{r['min_seconds']:.3f}–{r['max_seconds']:.3f} | {r['speedup_vs_LR']:.2f}× |")
    lines += ['', 'Сырые повторы и описательные bootstrap-интервалы ускорения '
      'сохранены в JSON. Они характеризуют разброс времени на этой машине, '
      'а не разброс поведения по сетям. Небольшие различия между вариантами '
      'нужно оценивать по диапазонам повторов.', '',
      '## Сохранённые данные', '',
      '- [evaluation.json](evaluation.json): параметры, метрики, временные повторы и хеши.',
      '- [metrics.csv](metrics.csv): скалярные метрики по всем моделям и двум входам.',
      '- [timings.csv](timings.csv): 56 измеренных запусков с раздельными стадиями.',
      '- [fit.json](fit.json): обучающие равновесия, веса и выбранный порог.',
      '- `traces/*.npz`: полные записанные траектории и спайки.',
      '- Новые модели: `models/high_ip3_hinge/`.', '',
      'Дополнительно сохраняются интеграл SIC, его максимумы и времена пиков, '
      'ошибки C/h, CV ISI, ошибка популяционной частоты в окнах 0.1 и 1 с, '
      'проверка физических границ и эффект относительно no_SIC. '
      'Один seed в полностью связанной feedforward-сети не доказывает '
      'сохранение синхронизации или других режимов рекуррентной сети Brunel.', '',
      '## Воспроизведение', '', '```bash',
      '.venv/bin/python scripts/fit_high_ip3_hinge.py',
      '# Собрать NEST-модуль по docs/nest_module.md',
      'MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/evaluate_high_ip3_hinge.py',
      'MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/report_high_ip3_hinge.py',
      '```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
