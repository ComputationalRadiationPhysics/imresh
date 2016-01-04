
CC=g++
INC=-I. -I../libs
WARNINGS=-Wall -Wextra -Wshadow -Wno-unused-parameter
CFLAGS=-g -fopenmp -std=c++11 $(INC) $(WARNINGS)


.PHONY: clean all colors.a phasereconstruction/phasereconstruction.a \
vectorReduce.o vectorElementwise.o hybridInputOutput.o shrinkWrap.o

all: imresh.a cuda/cuda.a

clean:
	rm -f main.exe *.o *.a
	make -C cuda clean

cuda/cuda.a:
	make -C cuda

vectorReduce.o: vectorReduce.cpp vectorReduce.h
	$(CC) -c $(CFLAGS) $< -o $@
vectorElementwise.o: vectorElementwise.cpp vectorElementwise.h
	$(CC) -c $(CFLAGS) $< -o $@
hybridInputOutput.o: hybridInputOutput.cpp hybridInputOutput.h
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -lfftw3 -lfftw3f -o $@
shrinkWrap.o: shrinkWrap.cpp shrinkWrap.h
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -lfftw3 -lfftw3f -o $@


imresh.a: vectorReduce.o vectorElementwise.o hybridInputOutput.o shrinkWrap.o
	rm -f $@
	ar rcsT $@ $^
