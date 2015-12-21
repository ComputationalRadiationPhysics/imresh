
CC=g++-4.9
# Not exactly sure why, but sdl2-config only works with $$, not with $
INC=-I.
WARNINGS=-Wall -Wextra -Wshadow -Wno-unused-parameter
NVCCFLAGS=-std=c++11 --compiler-bindir $$(which $(CC)) -arch=sm_20 \
		  $$(sdl2-config --cflags) $(INC) --compiler-options -fopenmp \
		  --compiler-options -Wall,-Wextra,-Wshadow,-Wno-unused-parameter
CFLAGS=-g -fopenmp -std=c++11 $$(sdl2-config --cflags) $(INC) $(WARNINGS)
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf -lfftw3

.PHONY: clean all

all: main.exe

clean:
	rm -f main.exe *.o

gaussian.o: math/image/gaussian.cpp math/image/gaussian.h
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

sdlcommon.o: sdl/sdlcommon.cpp sdl/sdlcommon.h sdl/sdlcommon.tpp
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

sdlplot.o: sdl/sdlplot.cpp sdl/sdlplot.h sdl/sdlplot.tpp
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

cudacommon.o: cudacommon.cu cudacommon.h
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@

cudaGaussian.o: math/image/cudaGaussian.cu math/image/cudaGaussian.h cudacommon.cu
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@

dft.o: math/fouriertransform/dft.cpp math/fouriertransform/dft.h
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

matrix.o: math/matrix/matrixInvertGaussJacobi.cpp math/matrix/matrixInvertGaussJacobi.h
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

vectorReduce.o: math/vector/vectorReduce.cpp math/vector/vectorReduce.h
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

testGaussian2d.o: tests/testGaussian2d.cu tests/testGaussian2d.h cudacommon.cu
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@

main.exe: main.cpp gaussian.o dft.o matrix.o vectorReduce.o cudaGaussian.o \
	      testGaussian2d.o sdlplot.o sdlcommon.o cudacommon.o \
	      sdl/sdlplot.h sdl/sdlplot.cpp sdl/sdlcommon.h \
		  math/fouriertransform/dcft.h math/fouriertransform/dcft.cpp \
		  tests/testDcft.cpp tests/testDft.cpp tests/testFftw.cpp \
		  tests/testGaussian.cpp tests/testSdlPlot.cpp
	rm -f $@
	nvcc $(NVCCFLAGS) $< $(LIBS) gaussian.o dft.o matrix.o vectorReduce.o \
	testGaussian2d.o cudaGaussian.o sdlplot.o sdlcommon.o cudacommon.o -o $@
