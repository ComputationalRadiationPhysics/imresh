
CC=g++-4.9
INC=-I. -I..
NVCCFLAGS=-std=c++11 --compiler-bindir $$(which $(CC)) -arch=sm_20 \
		  $$(sdl2-config --cflags) $(INC) --compiler-options -fopenmp \
		  --compiler-options -Wall,-Wextra,-Wshadow,-Wno-unused-parameter

.PHONY: clean all cudaGaussian.o cudacommon.o
all: cuda.a
clean:
	rm -f main.exe *.o *.a

cudacommon.o: cudacommon.cu cudacommon.h
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@
cudaGaussian.o: cudaGaussian.cu cudaGaussian.h
	nvcc $(NVCCFLAGS) -dc -x cu $< -o $@

cuda.a: cudaGaussian.o cudacommon.o
	nvcc $(NVCCFLAGS) $(SDLLIBS) -lib $^ -o $@
