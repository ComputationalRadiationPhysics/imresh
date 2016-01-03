
CC=g++
SDLLIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf
INC=-I..
CFLAGS=-g -fopenmp -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $$(sdl2-config --cflags) $(INC)
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf -lfftw3 -lfftw3f
NVCCFLAGS=-std=c++11 --compiler-bindir $$(which $(CC)) -arch=sm_20        \
		  $$(sdl2-config --cflags) $(INC) --compiler-options -fopenmp     \
		  --compiler-options -Wall,-Wextra,-Wshadow,-Wno-unused-parameter


.PHONY: clean all algorithms.a sdlcommon.a examples.a cuda.a testVectorIndex.o testDiffractionIntensity.o testHybridInputOutput.o testShrinkWrap.o testGaussian.o testGaussian2d.o testCudaShrinkWrap.o

all: mainTestAll.exe

clean:
	rm -f *.exe *.o *.a


algorithms.a:
	make -C ../algorithms $@
sdlcommon.a:
	make -C ../sdlcommon $@
examples.a:
	make -C ../examples $@
cuda.a:
	make -C ../algorithms/cuda $@


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

testGaussian2d.o: testGaussian2d.cu testGaussian2d.h
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@
testCudaShrinkWrap.o: testCudaShrinkWrap.cpp testCudaShrinkWrap.h
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@

mainTestAll.exe: mainTestAll.cpp testVectorIndex.o testDiffractionIntensity.o sdlcommon.a testHybridInputOutput.o testShrinkWrap.o testGaussian.o algorithms.a
	rm -f $@
	$(CC) $(CFLAGS) $<            \
	testVectorIndex.o             \
	testDiffractionIntensity.o    \
	testHybridInputOutput.o       \
	testShrinkWrap.o              \
	testGaussian.o                \
	../sdlcommon/sdlcommon.a      \
		../colors/colors.a        \
		$(SDLLIBS)                \
	../algorithms/algorithms.a    \
		-lfftw3 -lfftw3f          \
	../examples/examples.a        \
	-o $@

mainTestAllCuda.exe: mainTestAllCuda.cpp sdlcommon.a algorithms.a testGaussian2d.o testCudaShrinkWrap.o cuda.a
	rm -f $@
	nvcc $(NVCCFLAGS) $<          \
	testGaussian2d.o              \
	testCudaShrinkWrap.o          \
	../examples/examples.a        \
	../algorithms/algorithms.a    \
		-lfftw3 -lfftw3f          \
	../sdlcommon/sdlcommon.a      \
		../colors/colors.a        \
		$(SDLLIBS)                \
	../algorithms/cuda/cuda.a     \
	-lcufft                       \
	 -o $@
