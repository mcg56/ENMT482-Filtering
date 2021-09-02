all: assignment-guide.pdf

assignment-guide.pdf: figs/IRsensor-error-histogram.pgf figs/IRsensor-error.pgf figs/IRsensor-linearised.pgf figs/IRsensor-scatter.pgf

%.pdf: %.tex
	pdflatex -shell-escape $*
	pdflatex -shell-escape $*

%.pgf: %.py
	cd $(dir $@); python3 $(notdir $<)

clean:
	-rm *.aux *.log
