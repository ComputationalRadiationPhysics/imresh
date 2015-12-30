
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
shrinkWrap.o:
	make -C ../algorithms/phasereconstruction $@
gaussian.o:
	make -C ../algorithms $@

sdlcommon.a:
	make -C ../sdlcommon $@
examples.a:
	make -C ../examples $@


testVectorIndex.o: testVectorIndex.cpp testVectorIndex.h vectorIndex.o
	$(CC) -c $(CFLAGS) $< -o $@
testDiffractionIntensity.o: testDiffractionIntensity.cpp testDiffractionIntensity.h diffractionIntensity.o
	$(CC) -c $(CFLAGS) $< -o $@
testHybridInputOutput.o: testHybridInputOutput.cpp testHybridInputOutput.h diffractionIntensity.o examples.a hybridInputOutput.o
	$(CC) -c $(CFLAGS) $< -o $@
testShrinkWrap.o: testShrinkWrap.cpp testShrinkWrap.h diffractionIntensity.o examples.a shrinkWrap.o
	$(CC) -c $(CFLAGS) $< -o $@
testGaussian.o: testGaussian.cpp testGaussian.h gaussian.o
	$(CC) -c $(CFLAGS) $< -o $@
#testGaussian2d.o: testGaussian2d.cpp testGaussian2d.h gaussian.o
#	$(CC) -c $(CFLAGS) $< -o $@


mainTestAll.exe: mainTestAll.cpp testVectorIndex.o testDiffractionIntensity.o sdlcommon.a testHybridInputOutput.o testShrinkWrap.o testGaussian.o
	rm -f $@
	$(CC) $(CFLAGS) $< testVectorIndex.o ../algorithms/vectorIndex.o \
	testDiffractionIntensity.o ../algorithms/diffractionIntensity.o \
	testHybridInputOutput.o  \
	testShrinkWrap.o ../algorithms/phasereconstruction/shrinkWrap.o \
	../algorithms/phasereconstruction/hybridInputOutput.o \
	testGaussian.o \
	../examples/examples.a \
	../algorithms/gaussian.o \
	../sdlcommon/sdlcommon.a ../colors/colors.a $(SDLLIBS) \
	 -lfftw3 -lfftw3f -o $@
