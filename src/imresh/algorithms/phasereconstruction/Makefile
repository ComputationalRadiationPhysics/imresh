
CC=g++-4.9
INC=-I.
NVCCFLAGS=-std=c++11 --compiler-bindir $$(which $(CC)) -arch=sm_20 \
		  $$(sdl2-config --cflags) $(INC) --compiler-options -fopenmp \
		  --compiler-options -Wall,-Wextra,-Wshadow,-Wno-unused-parameter

SDLLIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf
WARNINGS=-Wall -Wextra -Wshadow -Wno-unused-parameter
CFLAGS=-g -fopenmp -std=c++11 $$(sdl2-config --cflags) $(INC) $(WARNINGS)



.PHONY: clean all

all: phasereconstruction.a

clean:
	rm -f main.exe *.o *.a


hybridInputOutput.o: hybridInputOutput.cpp hybridInputOutput.h
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -lfftw3 -lfftw3f -o $@

#sdlplot.o: sdl/sdlplot.cpp sdl/sdlplot.h sdl/sdlplot.tpp sdlcommon.o colors/colors.a
#	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@
#
#complexPlot.o: sdl/complexPlot.cpp sdl/complexPlot.h sdlplot.o colors/colors.a
#	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -lfftw3 -o $@
#
#cudacommon.o: cudacommon.cu cudacommon.h
#	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@
#
#cudaGaussian.o: math/image/cudaGaussian.cu math/image/cudaGaussian.h cudacommon.h
#	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@
#
#dft.o: math/fouriertransform/dft.cpp math/fouriertransform/dft.h
#	$(CC) -c $(CFLAGS) $< -o $@
#
#matrix.o: math/matrix/matrixInvertGaussJacobi.cpp math/matrix/matrixInvertGaussJacobi.h
#	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@
#
#vectorReduce.o: math/vector/vectorReduce.cpp math/vector/vectorReduce.h
#	$(CC) -c $(CFLAGS) $< -o $@
#
#
#
#cudaBasics.a: cudaGaussian.o cudacommon.o
#	nvcc $(NVCCFLAGS) $(SDLLIBS) -lib $^ -o $@

phasereconstruction.a: hybridInputOutput.o
	ar rcsT $@ $^
