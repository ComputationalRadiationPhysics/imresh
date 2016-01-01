
CC=g++-4.9
INC=-I.
NVCCFLAGS=-std=c++11 --compiler-bindir $$(which $(CC)) -arch=sm_20 \
		  $$(sdl2-config --cflags) $(INC) --compiler-options -fopenmp \
		  --compiler-options -Wall,-Wextra,-Wshadow,-Wno-unused-parameter

WARNINGS=-Wall -Wextra -Wshadow -Wno-unused-parameter
CFLAGS=-g -fopenmp -std=c++11 $(INC) $(WARNINGS)



.PHONY: clean all colors.a phasereconstruction/phasereconstruction.a \
vectorIndex.o vectorReduce.o vectorElementwise.o diffractionIntensity.o \
gaussian.o

all: algorithms.a

clean:
	rm -f main.exe *.o *.a
	make -C phasereconstruction clean


phasereconstruction/phasereconstruction.a:
	make -C phasereconstruction


vectorIndex.o: vectorIndex.cpp vectorIndex.h
	$(CC) -c $(CFLAGS) $< -o $@
vectorReduce.o: vectorReduce.cpp vectorReduce.h
	$(CC) -c $(CFLAGS) $< -o $@
vectorElementwise.o: vectorElementwise.cpp vectorElementwise.h
	$(CC) -c $(CFLAGS) $< -o $@
diffractionIntensity.o: diffractionIntensity.cpp diffractionIntensity.h
	$(CC) -c $(CFLAGS) $< -lfftw3 -lfftw3f -o $@
gaussian.o: gaussian.cpp gaussian.h
	$(CC) -c $(CFLAGS) $< -o $@

#cudacommon.o: cudacommon.cu cudacommon.h
#	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@
#
#cudaBasics.a: cudaGaussian.o cudacommon.o
#	nvcc $(NVCCFLAGS) $(SDLLIBS) -lib $^ -o $@

algorithms.a: phasereconstruction/phasereconstruction.a vectorIndex.o vectorReduce.o vectorElementwise.o diffractionIntensity.o gaussian.o
	rm -f $@
	ar rcsT $@ $^
