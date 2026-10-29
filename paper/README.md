# Статья для Chaos, Solitons & Fractals

Работаем в ветке `paper`. Основной файл статьи — `main.tex`.
Сейчас это заготовка: название, авторов, аннотацию и текст заполним при написании.

## Файлы

- `main.tex` — рукопись на английском языке, формат `preprint`, 12 pt.
- `references.bib` — база литературы.
- `figure1.tex` — подпись и подключение фигуры 1.
- `figures/` — фигура 1 в PDF, SVG и PNG, численные данные и описание расчёта.
- `data/astrocyte_neuron_ratio/` — таблица и исходная сводка замеров ускорения NEST при изменении числа астроцитов на нейрон.
- `data/two_threshold_models/` — сводные ошибки циклов и замеры NEST для двухпороговых моделей, исходные данные и описание протоколов.
- `highlights.txt` — отдельная заготовка highlights.
- `template/elsarticle-template-num.tex` — неизменённый исходный шаблон Elsevier
  с числовыми ссылками.
- `template/elsarticle.cls`, `template/elsarticle-num.bst` — класс и стиль
  библиографии, используемые при сборке.
- `template/` также содержит исходники класса, лицензию LPPL, README разработчика
  и `UPSTREAM.json` с Git SHA файлов для проверки происхождения.

## Сборка

Нужны GNU Make, `latexmk`, pdfLaTeX и BibTeX из TeX Live или MiKTeX
с пакетами `natbib`, `amsmath`, `amssymb`, `booktabs`, `hyperref` и зависимостями
`elsarticle`. Класс `elsarticle` и его стиль библиографии включены в репозиторий.

Из корня репозитория:

```bash
make -C paper
```

Результат — `paper/build/main.pdf`. Служебные файлы и PDF не попадают в Git.
Очистка: `make -C paper clean`.

Для Overleaf загрузить содержимое `paper/`, переместить `elsarticle.cls`
и `elsarticle-num.bst` из `template/` в корень проекта и выбрать
`main.tex` основным документом, компилятор pdfLaTeX.

## Источники шаблона

Использован пакет `elsarticle` версии 3.4c (2025/01/11) из
[репозитория разработчика STM Document Engineering](https://github.com/STM-Document-Engineering/elsarticle).
Файлы скопированы без изменений; размер и Git SHA каждого файла проверены.
Это общий шаблон Elsevier, настроенный на название журнала
`Chaos, Solitons & Fractals`, а не отдельный класс журнала.

- [Инструкции Elsevier для LaTeX](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions).
- [Guide for Authors журнала](https://www.sciencedirect.com/journal/chaos-solitons-and-fractals/publish/guide-for-authors).

Страница Guide for Authors при подготовке вернула HTTP 403, поэтому актуальные
специальные требования журнала здесь не заявлены как проверенные. Перед подачей
нужно сверить их, заполнить заявления авторов и заменить все заготовки.
Для подачи через Editorial Manager исходники, рисунки и стили следует собрать
в один каталог, как указано в инструкции Elsevier.

Для содержательной работы: `../docs/article_claims.md` содержит подтверждённые
формулировки по результатам проекта; переносим их в статью с проверкой источников.
