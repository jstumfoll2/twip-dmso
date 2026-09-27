# Build the thesis.  Requires a TeX distribution with pdflatex and bibtex.
#   make          -> main.pdf
#   make clean    -> remove build artifacts
#   make watch    -> continuous rebuild (requires latexmk)

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex frontmatter.tex $(wildcard chapters/*.tex) refs.bib
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

watch:
	latexmk -pdf -pvc $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk
	rm -f chapters/*.aux

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: all watch clean distclean
