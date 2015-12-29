
CC=g++
SDLLIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf
CFLAGS=-g -fopenmp -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $$(sdl2-config --cflags) -I.. -I./basics
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf -lfftw3 -lfftw3f


.PHONY: clean all vectorIndex.o sdlcommon.a diffractionIntensity.o examples.a

all: mainTestAll.exe

clean:
	rm -f *.exe *.o *.a


vectorIndex.o:
	make -C ../algorithms $@
diffractionIntensity.o:
	make -C ../algorithms $@
hybridInputOutput.o:
	make -C ../algorithms/phasereconstruction $@

sdlcommon.a:
	make -C ../sdlcommon $@
examples.a:
	make -C ../examples $@


testVectorIndex.o: testVectorIndex.cpp testVectorIndex.h vectorIndex.o
	$(CC) -c $(CFLAGS) $< -o $@
testDiffractionIntensity.o: testDiffractionIntensity.cpp testDiffractionIntensity.h diffractionIntensity.o
	$(CC) -c $(CFLAGS) $< -o $@
testHybridInputOutput.o: testHybridInputOutput.cpp testHybridInputOutput.h diffractionIntensity.o examples.a
	$(CC) -c $(CFLAGS) $< -o $@


mainTestAll.exe: mainTestAll.cpp testVectorIndex.o testDiffractionIntensity.o sdlcommon.a testHybridInputOutput.o hybridInputOutput.o
	rm -f $@
	$(CC) $(CFLAGS) $< testVectorIndex.o ../algorithms/vectorIndex.o \
	testDiffractionIntensity.o ../algorithms/diffractionIntensity.o \
	testHybridInputOutput.o ../examples/examples.a \
	../algorithms/phasereconstruction/hybridInputOutput.o \
	../sdlcommon/sdlcommon.a ../colors/colors.a $(SDLLIBS) \
	 -lfftw3 -lfftw3f -o $@
