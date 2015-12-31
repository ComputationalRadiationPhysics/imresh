
CC=g++
SDLLIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf
CFLAGS=-g -fopenmp -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $$(sdl2-config --cflags) -I.. -I./basics
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf -lfftw3 -lfftw3f


.PHONY: clean all algorithms.a sdlcommon.a examples.a

all: mainTestAll.exe

clean:
	rm -f *.exe *.o *.a


algorithms.a:
	make -C ../algorithms $@
sdlcommon.a:
	make -C ../sdlcommon $@
examples.a:
	make -C ../examples $@


testVectorIndex.o: testVectorIndex.cpp testVectorIndex.h
	$(CC) -c $(CFLAGS) $< -o $@
testDiffractionIntensity.o: testDiffractionIntensity.cpp testDiffractionIntensity.h
	$(CC) -c $(CFLAGS) $< -o $@
testHybridInputOutput.o: testHybridInputOutput.cpp testHybridInputOutput.h
	$(CC) -c $(CFLAGS) $< -o $@
testShrinkWrap.o: testShrinkWrap.cpp testShrinkWrap.h
	$(CC) -c $(CFLAGS) $< -o $@
testGaussian.o: testGaussian.cpp testGaussian.h
	$(CC) -c $(CFLAGS) $< -o $@
#testGaussian2d.o: testGaussian2d.cpp testGaussian2d.h gaussian.o
#	$(CC) -c $(CFLAGS) $< -o $@


mainTestAll.exe: mainTestAll.cpp testVectorIndex.o testDiffractionIntensity.o sdlcommon.a testHybridInputOutput.o testShrinkWrap.o testGaussian.o algorithms.a
	rm -f $@
	$(CC) $(CFLAGS) $< testVectorIndex.o \
	testDiffractionIntensity.o \
	testHybridInputOutput.o  \
	testShrinkWrap.o  \
	testGaussian.o \
	../examples/examples.a \
	../sdlcommon/sdlcommon.a ../colors/colors.a $(SDLLIBS) \
	../algorithms/phasereconstruction/phasereconstruction.a \
	../algorithms/algorithms.a  -lfftw3 -lfftw3f -o $@
