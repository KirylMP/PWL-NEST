# PWL-NEST: компактная AM/FM PWA на спектральных C-порогах

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
