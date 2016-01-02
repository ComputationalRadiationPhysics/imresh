
CC=g++-4.9
INC=-I.
WARNINGS=-Wall -Wextra -Wshadow -Wno-unused-parameter
CFLAGS=-g -fopenmp -std=c++11 $(INC) $(WARNINGS)


.PHONY: clean all colors.a phasereconstruction/phasereconstruction.a \
vectorIndex.o vectorReduce.o vectorElementwise.o diffractionIntensity.o \
gaussian.o cuda/cuda.a

all: algorithms.a cuda/cuda.a

clean:
	rm -f main.exe *.o *.a
	make -C phasereconstruction clean
	make -C cuda clean


phasereconstruction/phasereconstruction.a:
	make -C phasereconstruction
cuda/cuda.a:
	make -C cuda

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

algorithms.a: phasereconstruction/phasereconstruction.a vectorIndex.o vectorReduce.o vectorElementwise.o diffractionIntensity.o gaussian.o
	rm -f $@
	ar rcsT $@ $^
