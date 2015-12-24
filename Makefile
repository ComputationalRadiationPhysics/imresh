
CC=g++
CFLAGS=-g -fopenmp -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $$(sdl2-config --cflags) -I. -I./basics
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf -lfftw3

.PHONY: clean all

all: main.exe

clean:
	rm -f main.exe

main.exe: main.cpp basics/gaussian.o basics/sdlcommon.o basics/sdlplot.o
	make -C basics
	rm -f $@
	$(CC) $(CFLAGS) $< $(LIBS) basics/gaussian.o basics/sdlcommon.o basics/sdlplot.o -o $@
