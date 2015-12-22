
CC=g++-4.9
# Not exactly sure why, but sdl2-config only works with $$, not with $
INC=-I.
NVCCFLAGS=-std=c++11 --compiler-bindir $$(which $(CC)) -arch=sm_20 \
		  $$(sdl2-config --cflags) $(INC) --compiler-options -fopenmp \
		  --compiler-options -Wall,-Wextra,-Wshadow,-Wno-unused-parameter

SDLLIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf
WARNINGS=-Wall -Wextra -Wshadow -Wno-unused-parameter
CFLAGS=-g -fopenmp -std=c++11 $$(sdl2-config --cflags) $(INC) $(WARNINGS)

.PHONY: clean all

all: mainTestAll.exe mainUnderstandShrinkWrap.exe

clean:
	rm -f main.exe *.o

gaussian.o: math/image/gaussian.cpp math/image/gaussian.h
	$(CC) -c $(CFLAGS) $< -o $@

sdlcommon.o: sdl/sdlcommon.cpp sdl/sdlcommon.h sdl/sdlcommon.tpp
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

sdlplot.o: sdl/sdlplot.cpp sdl/sdlplot.h sdl/sdlplot.tpp
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

cudacommon.o: cudacommon.cu cudacommon.h
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@

cudaGaussian.o: math/image/cudaGaussian.cu math/image/cudaGaussian.h cudacommon.h
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@

dft.o: math/fouriertransform/dft.cpp math/fouriertransform/dft.h
	$(CC) -c $(CFLAGS) $< -o $@

matrix.o: math/matrix/matrixInvertGaussJacobi.cpp math/matrix/matrixInvertGaussJacobi.h
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

vectorReduce.o: math/vector/vectorReduce.cpp math/vector/vectorReduce.h
	$(CC) -c $(CFLAGS) $< -o $@

testGaussian.o: tests/testGaussian.cpp tests/testGaussian.h sdlplot.o
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

testGaussian2d.o: tests/testGaussian2d.cu tests/testGaussian2d.h cudacommon.h sdlplot.o
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@

testDft.o: tests/testDft.cpp tests/testDft.h sdlplot.o
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

testFftw.o: tests/testFftw.cpp tests/testFftw.h sdlplot.o
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -lfftw3 -o $@

mainTestAll.exe: mainTestAll.cpp gaussian.o dft.o matrix.o vectorReduce.o cudaGaussian.o \
	      testGaussian2d.o sdlplot.o sdlcommon.o cudacommon.o \
		  math/fouriertransform/dcft.h math/fouriertransform/dcft.cpp \
		  tests/testDcft.cpp testDft.o testFftw.o \
		  testGaussian.o tests/testSdlPlot.cpp
	rm -f $@
	nvcc $(NVCCFLAGS) $< $(SDLLIBS) -lfftw3 gaussian.o dft.o matrix.o vectorReduce.o \
	testGaussian.o testDft.o testFftw.o \
	testGaussian2d.o cudaGaussian.o sdlplot.o sdlcommon.o cudacommon.o -o $@

mainUnderstandShrinkWrap.exe: mainUnderstandShrinkWrap.cpp \
		gaussian.o sdlplot.o sdlcommon.o cudacommon.o vectorReduce.o \
		../examples/createAtomCluster.cpp ../examples/createSlit.cpp
	rm -f $@
	nvcc $(NVCCFLAGS) $< $(SDLLIBS) -lfftw3 gaussian.o sdlplot.o sdlcommon.o \
	cudacommon.o vectorReduce.o -o $@
