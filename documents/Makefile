FILES = fempy.tex
BASE = fempy

all: full

quick:  $(FILES)
	pdflatex -synctex=1 $<
	if test $(OSTYPE) == darwin; then open -a Skim $(BASE).pdf; fi

full:   $(FILES)
	pdflatex -synctex=1 $<
	makeindex $(basename $<).nlo -s nomencl.ist -o $(basename $<).nls
#	bibtex $(basename $<)
	pdflatex -synctex=1 $<
	pdflatex -synctex=1 $<
	@if test $(OSTYPE) == darwin; then open -a Skim $(BASE).pdf; fi

realfull: $(FILES)
	make clean
	$(HOME)/Local/darwin/bin/close Skim
	pdflatex -synctex=1 $<
	makeindex dissertation.nlo -s nomencl.ist -o dissertation.nls
	bibtex $(basename $<)
	pdflatex -synctex=1 $<
	pdflatex -synctex=1 $<
	if test $(OSTYPE) == darwin; then open -a Skim $(BASE).pdf; fi

clean:
	@rm -f $(addsuffix .aux, $(BASE) ) $(addsuffix .bbl, $(BASE) )
	@rm -f $(addsuffix .blg, $(BASE) ) $(addsuffix .lof, $(BASE) )
	@rm -f $(addsuffix .log, $(BASE) ) $(addsuffix .lot, $(BASE) )
	@rm -f $(addsuffix .glo, $(BASE) ) $(addsuffix .gls, $(BASE) )
	@rm -f $(addsuffix .ilg, $(BASE) ) $(addsuffix .nlo, $(BASE) )
	@rm -f $(addsuffix .nls, $(BASE) ) $(addsuffix .toc, $(BASE) )
	@rm -f $(addsuffix .synctex.gz, $(BASE) )
	@rm -rf auto
	@rm -f texput.log
