
CC=g++
# Not exactly sure why, but sdl2-config only works with $$, not with $
CFLAGS=-g -fopenmp -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $$(sdl2-config --cflags) -I. -I./sdl
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf -lfftw3

.PHONY: clean all

all: main.exe

clean:
	rm -f main.exe gaussian.o

gaussian.o: math/image/gaussian.cpp math/image/gaussian.h
	rm -f $@
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

dft.o: math/fouriertransform/dft.cpp math/fouriertransform/dft.h
	rm -f $@
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

matrix.o: math/matrix/matrixInvertGaussJacobi.cpp math/matrix/matrixInvertGaussJacobi.h
	rm -f $@
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

main.exe: main.cpp gaussian.o dft.o matrix.o \
	      sdl/sdlplot.h sdl/sdlplot.cpp sdl/sdlcommon.cpp sdl/sdlcommon.h \
		  math/fouriertransform/dcft.h math/fouriertransform/dcft.cpp \
		  tests/testDcft.cpp tests/testDft.cpp tests/testFftw.cpp \
		  tests/testGaussian.cpp tests/testGaussian2d.cpp tests/testSdlPlot.cpp
	rm -f $@
	$(CC) $(CFLAGS) $< $(LIBS) gaussian.o dft.o matrix.o -o $@
