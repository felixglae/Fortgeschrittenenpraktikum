all: build/template.pdf

TeXOptions = -lualatex \
				 -interaction=nonstopmode \
				 -halt-on-error \
				 -output-directory=build

build/template.pdf: FORCE | build
	latexmk $(TexOptions) content/template.tex

light: content/template.tex
	lualatex -interaction=nonstopmode \
	-halt-on-error \
	-output-directory=build \
	content/template.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
