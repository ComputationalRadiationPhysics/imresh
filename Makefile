
CC=g++
CFLAGS=-g -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $$(sdl2-config --cflags) -I. -I./basics
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf -lfftw3

.PHONY: clean all

all: main.exe

clean:
	rm -f main.exe

gaussian.o: basics/math/image/gaussian.cpp basics/math/image/gaussian.h
	rm -f $@
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

main.exe: main.cpp gaussian.o
	make -C basics
	rm -f $@
	$(CC) $(CFLAGS) $< $(LIBS) gaussian.o -o $@
