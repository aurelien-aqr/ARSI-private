# ARSI-VLM manuscript. `make` builds main.pdf; `make clean` removes aux files.
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all clean watch wordcount

all: main.pdf

main.pdf: main.tex preamble.tex metadata.tex references.bib $(wildcard sections/*.tex) $(wildcard figures/*)
	$(LATEXMK) main.tex

watch:
	$(LATEXMK) -pvc main.tex

# Rough body word count, excluding LaTeX markup and the bibliography.
wordcount:
	@cat sections/*.tex | sed 's/%.*//' | detex 2>/dev/null | wc -w || \
	 cat sections/*.tex | sed 's/%.*//' | wc -w

clean:
	latexmk -C
	rm -f main.bbl main.blg *.aux sections/*.aux
