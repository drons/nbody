
TARGET = $(patsubst %.tex, %.png,$(wildcard *.tex)) $(patsubst %.tex, %.svg,$(wildcard *.tex))

all: $(TARGET)

%.png: %.svg
	convert -density 150 $^ $@

%.svg: %.pdf
	pdf2svg $^ $@

%.pdf: %.tex
	pdflatex $^
	pdflatex $^

clean:
	rm -rf *.pdf *.svg *.png *.aux *.log
