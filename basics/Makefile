
CC=g++
# Not exactly sure why, but sdl2-config only works with $$, not with $
CFLAGS=-g -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $$(sdl2-config --cflags) -I. -I./sdl
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf -lfftw3

.PHONY: clean all

all: main.exe

clean:
	rm -f main.exe gaussian.o

gaussian.o: gaussian.cpp gaussian.h
	rm -f $@
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

main.exe: main.cpp gaussian.o sdl/sdlplot.h sdl/sdlplot.cpp sdl/sdlcommon.cpp sdl/sdlcommon.h dcft.h dcft.cpp dft.h dft.cpp tests/testDcft.cpp tests/testDft.cpp tests/testFftw.cpp tests/testGaussian.cpp tests/testGaussian2d.cpp tests/testSdlPlot.cpp
	rm -f $@
	$(CC) $(CFLAGS) $< $(LIBS) gaussian.o -o $@
