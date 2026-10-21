# Текущий FM-кандидат: уточнение времени пиков

Сохранены 8 областей, 27 параметров и точный покой. На шести новых входах
средняя ошибка времени пиков уменьшилась с 2.14 до 1.52 с, C RMSE —
с .138 до .118 мкМ. Число пиков верное на 6/6; форма проходит на 5/6:
на нерегулярном входе остаётся ошибка ширины около 20%.
[Модель, графики и воспроизведение](results/fm_phase_continuation/README.md).

Последнее уточнение широкого пика: четыре запуска и 169 точек сетки
не дали нового допустимого кандидата; прежняя модель сохранена.
[Отрицательный результат и проверки](results/fm_width_continuation/README.md).

Предыдущий раунд со сдвигами и исправлением лишнего пика tau=48:
[результаты](results/fm_shift_continuation/README.md).

# PWL-NEST: компактные AM/FM при экспоненциальном I

На предыдущем этапе получен **FM-кандидат с точным покоем LR и ограничениями формы пиков**.
На шести новых входах средняя RMSE C уменьшилась с .165 до .136 мкМ;
число и форма пиков проходят на 5/6 входов. Остался лишний пик при tau=48.
Достаточно прежних восьми областей и 27 параметров: малый наклон удалось убрать.
[Модель, результаты и графики](results/fm_anchored_shapes/README.md),
[условия и воспроизведение](docs/anchored_shapes.md).

Проверены **наклонные и квадратичные границы C–h**, выпрямленные в координату U.
Остались восемь областей и постоянные региональные матрицы. На шести новых входах
числа пиков совпали, но общая ошибка C выросла; исходный FM сохранён.
[Результаты и геометрия границ](results/fm_mixed_coordinates/README.md),
[преобразование и запуск](docs/mixed_coordinates.md).

Проверен **один подбираемый I-порог для FM**: найден I≈.548 мкМ.
Число пиков совпало на всех шести новых входах, но общая ошибка C выросла;
исходная модель сохранена. [Результаты и графики](results/fm_one_input_threshold/README.md),
[уравнения, бюджеты и запуск](docs/fm_one_input_threshold.md).
Исправлена утечка test-точек в полевой старт; новый опыт использует train-only
инициализацию, а исторические grid-таблицы считаются диагностическими.

Получены **результаты лабораторного перебора**: 156 ячеек, 2447 проверенных
перезапусков. [Полные C × h таблицы](results/server_coordinate_grid/README.md)
и [три сохранённых кандидата с независимым аудитом](results/server_candidates/README.md).
Для проверки накопления фазовой ошибки добавлен отдельный
[подбор с абсолютными временами пиков](docs/server_timing_refinement.md).
У FM ошибка времени уменьшилась, но появились лишние полные пики —
[исходный кандидат сохранён](results/server_timing_refinement/README.md).
У AM [новые входы](results/server_am_fresh_validation/README.md) выявили
чувствительность подсчёта последних слабых колебаний к порогу prominence.

Готов **ночной перебор C × h**: 14×6 преобразований отдельно для AM/FM,
с собственным устойчивым покоем, задержкой входа 20 с и независимым DOP-аудитом.
Сохраняются все попытки, лучшие train-кандидаты, причинные/фазовые ошибки,
CSV и PNG/PDF таблицы. Проверена малая сетка 2×2 (AM/FM, 8 попыток).
[Запуск и продолжение](docs/overnight_coordinate_grid.md),
[пример чисел](results/grid_smoke_v2/TABLE.md),
[пример таблицы FM](results/grid_smoke_v2/FM_coordinate_grid.png).

Отдельно проверены **инициализация из покоя и фазовая ошибка sqrt_tails**.
На FM tau100 число пиков совпадает, общий временной сдвиг уменьшает RMSE,
но задержки меняются по ходу ответа, а пики примерно на 16% уже LR.
Модели и начальные условия при этой проверке не подбирались.
[Отчёт и графики](results/rest_and_phase/REPORT.md).

Последний опыт проверяет **C⁶, C¹ᐟ⁶, h⁶ и ограниченные C-координаты**,
с повторным подбором коэффициентов и самих C-порогов. Добавлены проблемные
режимы в обучение и шесть новых проверочных входов на case. C не обрезается.

У FM полезнее оказались **плавные ограниченные хвосты прежнего √C-readout**:
все 26 проверок завершились в физических диапазонах на сетке выдачи,
сохранились 5/8 пиков при tau50/tau100. При h0=.9 максимум C снизился
с 1.727 до 1.548 мкМ, ниже физического Cmax≈1.688. Рабочая часть координаты
и прежние 8 PWA-областей сохранены; высокий пик всё ещё отличается от LR.
Шестые степени не улучшили повторный подбор прежних координат.

У AM выбран bounded_C: нужный цикл при I=.638 восстановлен, но амплитуда
и часть входных откликов ухудшились. **Общая приёмка не пройдена.** Выбор
`selected` — минимум нового калибровочного критерия, не одобрение модели.
Все сравнения и неудачные пилоты сохранены.

- [Отчёт, успехи и ухудшения](results/positive_coordinates/REPORT.md).
- [Графики](results/positive_coordinates/positive_coordinates_summary.png).
- [Формулы, ветви обратимости, критерий и команды](docs/positive_coordinates.md).
- Модели: `models/positive_coordinates/selected_AM.json` — bounded_C,
  `selected_FM.json` — sqrt_tails.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/fit_positive_coordinates.py --budget 1200
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/validate_positive_coordinates.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_positive_numerics.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/summarize_positive_coordinates.py
.venv/bin/python -m unittest discover -s tests -q
```

## Предыдущий эксперимент: h³, I-пороги и наклонные границы

Проверены **h³, один/два I-порога и наклонные C-границы**, а также
ln(C)+h³ и насыщение насоса+h³. На AM и FM по 8 кандидатов, по 20 протоколов.
В каждом регионе сохраняются постоянные матрицы и точная экспонента I.

По независимо проверенному train-критерию выбраны два I-порога у AM
(1.111 против control 1.391) и √C+h³ у FM (50.32 против 64.12).
**Приёмка не пройдена:** AM с I-слоями теряет нужный цикл при I=.638,
FM с h³ ещё превышает Cmax при h0=.9. ln(C) и насыщение насоса не улучшили
подбор. Старые кандидаты сохранены, выигрыши и ухудшения показаны вместе.

- [Полное сравнение](results/extended_pwa/REPORT.md).
- [Траектории и критерии](results/extended_pwa/extended_pwa_summary.png).
- [Наклонные пороги и входные зависимости](results/extended_pwa/geometry_and_inputs.png).
- [Формулы, число областей, проверки и команды](docs/extended_pwa.md).
- Модели: `models/extended_pwa/{variant}/{AM,FM}.json`;
  `selected_AM.json` — input_two, `selected_FM.json` — sqrt_cube по training.

При I-слоях непрерывность разрешает менять входной столбец и смещение;
два I-порога дают 12 областей у AM или 24 у FM. Наклонные C-границы
сохраняют 4/8 областей и меняют положение границ с I. В текущем опыте
активно подбираются 14/22 коэффициента и новые 2/4 либо 3/7 параметров.
Положения прежних C-порогов удерживаются фиксированными, кроме tilted.

```bash
bash scripts/setup.sh
.venv/bin/python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_extended_pwa.py
```

Грубый fitting RK4 у FM h³ заметно расходился с DOP; критерий пересчитан
по независимым training-траекториям, выбор сохранился. Численный аудит
сохранён. Полный аналитический событийный solver, SIC, NEST и ускорение
по-прежнему не реализованы.

## Предыдущий эксперимент: подвижные C-пороги

Последний опыт **подбирает сами C-пороги** вокруг спектрального старта.
Число областей сохранено: AM 4, FM 8; добавлены 3/7 подбираемых положений,
итого **17/29 параметров**. Порядок порогов, непрерывность и региональная
аналитичность сохраняются. Порогов по I нет.

Совместный подбор коэффициентов и порогов снизил обучающий критерий
на 15.2% у AM и 33.8% у FM относительно повторного подбора только
коэффициентов при том же максимальном бюджете. FM на спаде tau100
воспроизводит все 8 пиков вместо прежних 6; AM тоже улучшает отклики.
**Приёмка не пройдена:** неверный цикл FM при I=1.13, превышение Cmax
при h0=.9 и ошибка амплитуды AM около границы колебаний остаются.

- [Отчёт и таблица сдвигов порогов](results/movable_thresholds/REPORT.md).
- [Графики](results/movable_thresholds/movable_thresholds_summary.png).
- [Метод и воспроизведение](docs/movable_thresholds.md).
- Модели: `models/movable_thresholds/{baseline,fixed_refit,moving_joint}`;
  `selected_AM.json` и `selected_FM.json` — moving_joint по обучению.

```bash
bash scripts/setup.sh
.venv/bin/python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_calcium_coordinates.py --experiment movable_thresholds
```

## Предыдущий эксперимент: другие координаты C и h

Последний опыт сравнивает exp(C), ln(C), C⁻¹ и C⁻¹ᐟ² с √C, затем log/exp
для h. **AM: 4 области / 14 параметров. FM: 8 / 22.** Физические пороги
исходной спектральной эвристики сохранены. PWA и региональная аналитика
определены в преобразованных координатах, обратные функции дают C,h.

Среди новых C-координат √C сохранил лучший обучающий критерий. Для AM
добавление ln(h) снизило loss 2.181→1.834 и сохранило физические диапазоны
на всех 20 протоколах; точность около границы колебаний ещё недостаточна.
Для FM exp(h) улучшил новый контроль, но прежний sqrt из calcium_powers
лучше воспроизводит спад tau=50. Все кандидаты пока **не прошли приёмку**.

- [Отчёт о C/h-преобразованиях](results/coordinate_families/REPORT.md).
- [Графики сравнений](results/coordinate_families/coordinate_families_summary.png).
- [Методы, области определения и команды](docs/coordinate_families.md).
- [I через производную и δ-импульсы](docs/impulsive_ip3.md),
  [график](results/coordinate_families/impulsive_ip3.png).
- Текущий пилот: `models/coordinate_families/selected_AM.json` — √C+ln(h),
  `selected_FM.json` — √C+exp(h). Выбор внутри пилота по обучению;
  прежние модели сохранены, включая FM `models/calcium_powers/sqrt/FM.json`.

Точно: J=Idot+(I−Ib)/tau=Σak*δ(t−tk). J задаётся событиями,
I хранит их экспоненциальный след. На событии меняется только I.
Региональное движение остаётся аналитическим; полный событийный solver,
SIC, NEST и измерение ускорения пока не выполнены.

```bash
bash scripts/setup.sh
.venv/bin/python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_calcium_coordinates.py --experiment coordinate_families
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_impulsive_ip3.py
```

## Предыдущий эксперимент: квадрат и куб

Проверены степени **C² и C³** с заново подобранным контролем **√C**.
Сохраняются logit(h), исходные спектральные C-пороги и размеры моделей:
**AM: 4 области / 14 параметров. FM: 8 / 22.** Все три координаты приведены
к одному диапазону постоянной нормировкой, дополнительных параметров нет.

По общему обучающему критерию выбран √C для обеих моделей. На диагностическом
спаде I с tau=50 ошибка C у AM: .0230 / .1023 / .1136 мкМ для √C / C² / C³;
у FM: .1097 / .2780 / .3565 мкМ. Квадрат и куб в этом подборе проиграли.
Есть выходы из положительной ветви и неверные режимы колебаний.
**Полная приёмка не пройдена ни одним вариантом.**

- [Отчёт о степенях](results/calcium_powers/REPORT.md).
- [Графики](results/calcium_powers/calcium_powers_summary.png).
- [Преобразования, аналитичность и воспроизведение](docs/calcium_powers.md).
- Модели: `models/calcium_powers/{sqrt,square,cube}/{AM,FM}.json`;
  `selected_AM.json` и `selected_FM.json` — √C, выбраны только по обучению.

PWA хранится в преобразованных координатах; региональное движение при
экспоненциальном I аналитично через матричную экспоненту. Полные траектории
проверены DOP853, событийный аналитический решатель пока не реализован.
Старые модели и результаты сохранены.

```bash
bash scripts/setup.sh
.venv/bin/python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_calcium_coordinates.py --experiment calcium_powers
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_calcium_power_failures.py
```

## Предыдущий эксперимент: sqrt/cbrt/log

Сравнены квадратный корень, кубический корень и логарифм C, сохраняя logit(h)
и исходные физические спектральные C-пороги. Состояние хранится в новых
координатах, PWA и точный региональный поток определены в них.
**AM: 4 области / 14 параметров. FM: 8 / 22.** Дополнительных порогов нет.

- [Сравнительный отчёт](results/calcium_coordinates/REPORT.md).
- [Графики всех вариантов](results/calcium_coordinates/calcium_coordinates_summary.png).
- [Координаты, domains, методы и воспроизведение](docs/calcium_coordinates.md).
- Выбраны по обучению: `models/calcium_coordinates/selected_AM.json`,
  `selected_FM.json` — оба используют sqrt. Identity-контроль подбирался заново
  с той же процедурой; предыдущие модели сохранены.

sqrt качественно исправил отдельные режимы: AM при I=.638 колеблется,
но с сильно заниженной амплитудой; FM при I=1.13 приходит к покою.
**Приёмка ещё не пройдена**: есть лишние/пропущенные
пики и выход C за верхнюю физическую границу у AM при h0=.9. Корни проверяют
допустимую положительную ветвь; отрицательные координаты не сворачиваются
квадратом. Log может давать большой положительный runaway C.

```bash
# Из /workspace/PWL-NEST:
bash scripts/setup.sh
.venv/bin/python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_calcium_coordinates.py
```

Полный аналитический событийный решатель, SIC и NEST не реализованы. Полные
траектории проверяются DOP853 в новых координатах; региональная матричная
экспонента проверена отдельно. Ниже сохранены предыдущие baselines.

## Предыдущий эксперимент: только logit(h)

Текущий эксперимент: внутреннее состояние `(C,z=logit(h),I)`, непрерывная
PWA в этих координатах и обратная логистика только при выдаче физического h.
**AM: 4 области / 14 параметров; FM: 8 / 22.** Исходные спектральные C-пороги
сохранены, новых порогов и полиномов нет.

- [Сравнение физической PWA, контроля и нового подбора](results/logit_pwa/REPORT.md).
- [Графики](results/logit_pwa/logit_summary.png).
- [Преобразование, ограничения и воспроизведение](docs/logit_pwa.md).
- Модели: `models/logit_pwa/AM.json`, `models/logit_pwa/FM.json`.

Ветка преобразованных координат разрешена пользователем отдельно: физическое
поле после обратного преобразования обычно не PWA. Все 18 проверенных
протоколов каждой модели остались в физических диапазонах, однако точность
динамики недостаточна: **приёмка не пройдена**. AM почти не улучшилась;
у FM некоторые периоды и h стали ближе, но ответ C на медленный спад IP3
остаётся плохим. Контроль показал, что замена только h не расширяет класс C(t).

```bash
# Из /workspace/PWL-NEST:
bash scripts/setup.sh
.venv/bin/python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_logit.py
```

Полный событийный решатель, SIC и NEST пока не реализованы. Аналитический
региональный поток проверен отдельно; полные траектории проверяются DOP853
в преобразованных координатах. Предыдущие модели и результаты сохранены ниже.

## Предыдущая компактная PWA в физических координатах

Текущий эксперимент сохраняет исходную эвристику legacy: найти смену типа
равновесия Li–Rinzel по спектру якобиана и поставить постоянный порог по C.
**AM: 3 порога / 4 области / 14 параметров. FM: 7 порогов / 8 областей / 22 параметра.**
Поле непрерывно, внутри полос аффинно в физических `(C,h,I)`;
`Idot=-(I-I_b)/tau` точно. Дополнительных h/I-порогов и полиномов нет.

- [Результаты, включая неудачные проверки](results/calcium_strips/REPORT.md).
- [Графики](results/calcium_strips/compact_summary.png).
- [Эвристика, уравнения и воспроизведение](docs/calcium_strips.md).
- Модели: `models/calcium_strips/AM.json`, `models/calcium_strips/FM.json`.

Обе модели пока **не прошли приёмку**. AM воспроизводит основные колебания,
но теряет цикл в протоколе I=.638 и даёт лишний пик на периодическом входе.
FM сохраняет региональные типы, но имеет большие ошибки периода/отклика и
нарушения физических диапазонов. Это результаты локального подбора,
а не доказательство невозможности улучшить этот компактный класс.

Из `/workspace/PWL-NEST`:

```bash
bash scripts/setup.sh
.venv/bin/python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_strips.py
# Повторный подбор перезапишет текущие результаты:
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/refine_strips.py
.venv/bin/python scripts/summarize_strips.py
```

```python
from pwl_nest.calcium_strips import CalciumStrips
from pwl_nest.protocols import simulate
model = CalciumStrips.load('models/calcium_strips/AM.json')
t, x, diagnostics = simulate('AM', [.08, .8, .65], 180., tau=50., baseline=.3,
                             model=model, enforce_domain=False)
# Диагностический режим: явно проверить C,h, а не обрезать нарушения.
```

Полные траектории проверяются DOP853. Аналитический региональный поток
проверен отдельно; общий событийный решатель для динамического I ещё впереди.
Предыдущие плотные модели и legacy сохранены ниже как отдельные baselines.

## Предыдущий плотный 3D baseline

Текущая работа — строгая идентификация полного поля в физических `(C,h,I)`
по [блюпринту](docs/research_blueprint.md), с точной экспоненциальной релаксацией IP3.
Сохранены первые подобранные модели, аффинные внутри тетраэдров и непрерывные
на общих гранях. Полиномиальные вложения не используются.

- [Эвристика, ограничения класса и протокол подбора](docs/strict_pwa_identification.md).
- [Отчёт первого эксперимента](results/strict_pwa/REPORT.md).
- [Графики динамических ответов и сложности](results/strict_pwa/strict_pwa_summary.png).
- Модели: `models/strict_pwa/AM.npz`, `models/strict_pwa/FM.npz`.

Это плотные стартовые кандидаты: 209664 области AM и 408000 FM.
Выявлены ошибки около верхней границы AM и в фазе FM при медленном спаде IP3;
полная динамическая приёмка не пройдена. NEST, SIC и ускорение ещё не проверены.

Из `/workspace/PWL-NEST`:

```bash
bash scripts/setup.sh
.venv/bin/python -m unittest discover -s tests -v
# Проверить уже подобранные модели:
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_strict.py
# Полностью повторить подбор (перезапишет результаты):
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/fit_strict.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_strict_convergence.py
.venv/bin/python scripts/summarize_strict.py
```

Использование модели:

```python
from pwl_nest.strict_pwa import StrictPWA, regional_flow
from pwl_nest.protocols import simulate
model = StrictPWA.load('models/strict_pwa/AM.npz')
t, state, info = simulate('AM', [.08, .8, .65], 180., tau=50., baseline=.3, model=model)
A, b, tetrahedron = model.region([.08, .8, .65], tau=50., baseline=.3)
# Аналитический поток только до первого выхода из данного тетраэдра:
x_short = regional_flow(A, b, [.08, .8, .65], 1e-6)
```

Полные траектории пока проверяются DOP853; общий аналитический событийный
решатель ещё не реализован. `regional_flow` не ищет первый выход из региона.

## Исторические legacy baselines

Первый воспроизводимый baseline по двум legacy-архивам. Код для NEST пока не реализован.

- **AM**: `models/AM.json`, `AM_baseline4_F` из архива от 18 октября: **4 порога, 5 областей**.
  В legacy отмечен как validated reference. AM2 не выбран: полный протокол не пройден.
- **FM**: `models/FM.json`, fitted coefficients из `li_rinzel_AM_FM_affine_fit.zip`:
  **7 порогов, 8 областей**. Поле разрывно на границах; это исходный baseline для дальнейшего улучшения.
- **Li–Rinzel**: физические координаты `(C,h)`, `K_ER=0.1` (AM), `0.05` (FM).
  Параметры и единицы соответствуют legacy: C и IP3 в мкМ, время в секундах.

## Запуск

Python >=3.11, g++, Linux. Команды из `/workspace/PWL-NEST`:

```bash
bash scripts/setup.sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate.py
```

Установка использует доступные системные пакеты, проверяет закреплённые версии
numpy/scipy/matplotlib и пересобирает native engines из исходников.
Архивный `engine.so` не используется. Новые серверы и ключи доступа не требуются.

```python
from pwl_nest import trajectory
from pwl_nest.native import observables
t, state = trajectory('AM', .5, kind='pwa')
print(observables('FM', .8))
```

## Уравнения и ограничения

IP3 здесь **фиксирован** в диапазоне [0,1.3]. На участке j:

- AM: `Cdot=w+mu_j*C+g_j`, `wdot=q_j-k_j*C`.
  `g` и силы согласованы на границах: поле непрерывно в `(C,w)`.
  В native engine используется эквивалентная при постоянном IP3 координата `v=Cdot`.
- FM: `Cdot=v`, `vdot=-k_j*C+q_j+mu_j*v`.
  Здесь `q_j=k_j*center_j`, а `center_j` аффинно зависит от IP3.
  Непрерывность полного поля не обеспечена.

Ни w, ни v не являются h. Коэффициенты, зависящие от IP3, не делают полное
трёхмерное поле аффинным: возникают произведения состояния и IP3.
AM также имеет квадратичный вход и гладкий rest guard. Это пока не конечные
3D PWA-модели из [блюпринта](docs/research_blueprint.md).

`trajectory` — независимый численный DOP853, а `observables` — legacy C++
с аналитическим региональным потоком и локализацией границ. Python-решатель
не используется для заявлений об ускорении. Native FM engine сохраняет
legacy ограничения (включая предел числа шагов); полного общего событийного
решателя и доказательства положительности пока нет.

## Валидация

`scripts/validate.py` заново выполняет 7 фиксированных входов для каждой модели,
сохраняет метрики и доступные LR-reference значения в `results/validation.json`.
Старые PWS-колонки CSV не считаются эталоном текущей модели.
При AM I=.5 и FM I=.8 сравниваются native и независимые численные траектории;
допуски: 0.0002 мкМ для экстремумов, 0.12 с для периода.
Сохраняются полные траектории NPZ и `results/starting_models.png`.

Инициализация соответствует large-excursion протоколу: C0=.6 (AM), 1.3 (FM),
Cdot0=0 для PWA, h0=.8 для LR. Это сравнение привлекающих ветвей,
а не согласованных физических начальных состояний. На рисунке фазы не выровнены.
Период измеряется в последние 400 секунд; отсутствие периода обозначается null.
Проверки не заменяют полный acceptance protocol, бифуркационный анализ,
валидацию динамического IP3, SIC и benchmark NEST.

`models/provenance.json` содержит происхождение и SHA256 исходных моделей.
[Полиномиальное вложение](docs/polynomial_extension.md) сохранено как отдельная
исследовательская идея; непроверенные пробные модели не приняты.
