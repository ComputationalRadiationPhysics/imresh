
CC=g++-4.9
# Not exactly sure why, but sdl2-config only works with $$, not with $
INC=-I.
NVCCFLAGS=-std=c++11 --compiler-bindir $$(which $(CC)) -arch=sm_20 \
		  $$(sdl2-config --cflags) $(INC) --compiler-options -fopenmp \
		  --compiler-options -Wall,-Wextra,-Wshadow,-Wno-unused-parameter

SDLLIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf
WARNINGS=-Wall -Wextra -Wshadow -Wno-unused-parameter
CFLAGS=-g -fopenmp -std=c++11 $$(sdl2-config --cflags) $(INC) $(WARNINGS)



.PHONY: clean all colors/colors.a

all: basics.a cudaBasics.a mainTestAll.exe mainUnderstandShrinkWrap.exe

clean:
	rm -f main.exe *.o *.a
	make -C colors clean



colors/colors.a:
	make -C colors all

gaussian.o: math/image/gaussian.cpp math/image/gaussian.h
	$(CC) -c $(CFLAGS) $< -o $@

sdlcommon.o: sdl/sdlcommon.cpp sdl/sdlcommon.h sdl/sdlcommon.tpp
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

sdlplot.o: sdl/sdlplot.cpp sdl/sdlplot.h sdl/sdlplot.tpp sdlcommon.o colors/colors.a
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

complexPlot.o: sdl/complexPlot.cpp sdl/complexPlot.h sdlplot.o colors/colors.a
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -lfftw3 -o $@

cudacommon.o: cudacommon.cu cudacommon.h
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@

cudaGaussian.o: math/image/cudaGaussian.cu math/image/cudaGaussian.h cudacommon.h
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@

dft.o: math/fouriertransform/dft.cpp math/fouriertransform/dft.h
	$(CC) -c $(CFLAGS) $< -o $@

matrix.o: math/matrix/matrixInvertGaussJacobi.cpp math/matrix/matrixInvertGaussJacobi.h
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

vectorReduce.o: math/vector/vectorReduce.cpp math/vector/vectorReduce.h
	$(CC) -c $(CFLAGS) $< -o $@



cudaBasics.a: cudaGaussian.o cudacommon.o
	nvcc $(NVCCFLAGS) $(SDLLIBS) -lib $^ -o $@

basics.a: gaussian.o complexPlot.o colors/colors.a sdlplot.o sdlcommon.o \
 dft.o matrix.o vectorReduce.o
	ar rcsT $@ $^



testGaussian.o: tests/testGaussian.cpp tests/testGaussian.h sdlplot.o gaussian.o
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

testGaussian2d.o: tests/testGaussian2d.cu tests/testGaussian2d.h cudacommon.h sdlplot.o cudaGaussian.o
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@

testDft.o: tests/testDft.cpp tests/testDft.h sdlplot.o dft.o
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

testFftw.o: tests/testFftw.cpp tests/testFftw.h sdlplot.o
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -lfftw3 -o $@

testColors.o: tests/testColors.cpp tests/testColors.h sdlcommon.o colors/colors.a
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -lfftw3 -o $@



mainTestAll.exe: mainTestAll.cpp basics.a cudaBasics.a \
		  math/fouriertransform/dcft.h math/fouriertransform/dcft.cpp \
		  tests/testDcft.cpp tests/testSdlPlot.cpp \
		  testDft.o testFftw.o testGaussian.o testGaussian2d.o
	rm -f $@
	nvcc $(NVCCFLAGS) $< $(SDLLIBS) -lfftw3 \
	testGaussian.o testDft.o testFftw.o testGaussian2d.o cudaBasics.a basics.a -o $@

mainUnderstandShrinkWrap.exe: mainUnderstandShrinkWrap.cpp \
		gaussian.o sdlplot.o sdlcommon.o vectorReduce.o \
		../examples/createAtomCluster.cpp ../examples/createSlit.cpp \
		colors/colors.a testColors.o complexPlot.o
	rm -f $@
	$(CC) $(CFLAGS) $< $(SDLLIBS) -lfftw3 gaussian.o sdlplot.o sdlcommon.o \
	vectorReduce.o testColors.o colors/colors.a -o $@

mainUnderstandHybridInputOutput.exe: mainUnderstandHybridInputOutput.cpp \
		../examples/createAtomCluster.cpp ../examples/createSlit.cpp basics.a
	rm -f $@
	$(CC) $(CFLAGS) $< $(SDLLIBS) -lfftw3 basics.a -o $@
