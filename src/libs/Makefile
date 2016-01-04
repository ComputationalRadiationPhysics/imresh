
CC=g++
INC=-I.
WARNINGS=-Wall -Wextra -Wshadow -Wno-unused-parameter
CFLAGS=-g -fopenmp -std=c++11 $(INC) $(WARNINGS)


.PHONY: clean all libs.a vectorIndex.o diffractionIntensity.o gaussian.o

all: libs.a

clean:
	rm -f main.exe *.o *.a
	
vectorIndex.o: vectorIndex.cpp vectorIndex.h
	$(CC) -c $(CFLAGS) $< -o $@
diffractionIntensity.o: diffractionIntensity.cpp diffractionIntensity.h
	$(CC) -c $(CFLAGS) $< -lfftw3 -lfftw3f -o $@
gaussian.o: gaussian.cpp gaussian.h
	$(CC) -c $(CFLAGS) $< -o $@

libs.a: vectorIndex.o diffractionIntensity.o gaussian.o
	rm -f $@
	ar rcsT $@ $^
