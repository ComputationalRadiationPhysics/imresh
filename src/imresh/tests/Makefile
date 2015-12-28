
CC=g++
CFLAGS=-g -fopenmp -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $$(sdl2-config --cflags) -I.. -I./basics
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf -lfftw3 -lfftw3f


.PHONY: clean all vectorIndex.o sdlcommon.a diffractionIntensity.o

all: mainTestAll.exe

clean:
	rm -f *.exe *.o *.a


vectorIndex.o:
	make -C ../algorithms vectorIndex.o

diffractionIntensity.o:
	make -C ../algorithms diffractionIntensity.o

sdlcommon.a:
	make -C ../sdlcommon sdlcommon.a


testVectorIndex.o: testVectorIndex.cpp testVectorIndex.h vectorIndex.o
	$(CC) -c $(CFLAGS) $< -o $@

testDiffractionIntensity.o: testDiffractionIntensity.cpp testDiffractionIntensity.h diffractionIntensity.o
	$(CC) -c $(CFLAGS) $< -o $@


mainTestAll.exe: mainTestAll.cpp testVectorIndex.o testDiffractionIntensity.o
	rm -f $@
	$(CC) $(CFLAGS) $< testVectorIndex.o ../algorithms/vectorIndex.o \
	testDiffractionIntensity.o ../algorithms/diffractionIntensity.o \
	 -lfftw3 -lfftw3f -o $@
