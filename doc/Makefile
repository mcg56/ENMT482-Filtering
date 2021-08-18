all: assignment_brief.pdf report-template.pdf report-template.docx assignment_guide.pdf

%.pdf: %.md
	pandoc -f markdown -t latex $< -o $@ -V papersize:a4 -V fontsize=12pt  -V geometry:margin=1in

%.docx: %.tex
	-pandoc -t docx -N -o $@ $<

%.pdf: %.tex
	pdflatex $*
	pdflatex $*

assignment_guide.pdf: figs/IRsensor-error-histogram.pgf figs/IRsensor-error.pgf figs/IRsensor-linearised.pgf figs/IRsensor-scatter.pgf

%.pgf: %.py
	cd $(dir $@); python3 $(notdir $<)

clean:
	-rm *.aux *.log
