.PHONY: all ru english clean

# Use the bundled, unmodified Elsevier class and bibliography style.
# A trailing colon preserves the TeX distribution's default search paths.
export TEXINPUTS := ./template//:$(TEXINPUTS):
export BSTINPUTS := ./template//:$(BSTINPUTS):

all: ru

ru:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main_ru.tex

english:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

clean:
	latexmk -C -outdir=build main_ru.tex main.tex
