
CC=g++-4.9
INC=-I. -I../..
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
shrinkWrap.o: shrinkWrap.cpp shrinkWrap.h
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -lfftw3 -lfftw3f -o $@

#cudacommon.o: cudacommon.cu cudacommon.h
#	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@
#
#cudaBasics.a: cudaGaussian.o cudacommon.o
#	nvcc $(NVCCFLAGS) $(SDLLIBS) -lib $^ -o $@

phasereconstruction.a: hybridInputOutput.o shrinkWrap.o
	ar rcsT $@ $^
