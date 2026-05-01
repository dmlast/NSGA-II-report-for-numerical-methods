.PHONY: all assets notebook slides clean

all: assets notebook slides

assets:
	python3 scripts/generate_assets.py

slides:
	latexmk -xelatex -interaction=nonstopmode -file-line-error tex/slides.tex

clean:
	latexmk -C tex/slides.tex
