
CC=g++
CFLAGS=-g -fopenmp -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $$(sdl2-config --cflags) -I. -I./imresh -I./libs -I./examples
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf -lfftw3 -lfftw3f

.PHONY: clean all

all: main.exe

clean:
	rm -f *.exe *.o *.a

main.exe: main.cpp
	make -C libs all
	make -C imresh imresh.a
	make -C examples all
	rm -f $@
	$(CC) $(CFLAGS) $< $(LIBS) imresh/imresh.a examples/examples.a libs/libs.a -lfftw3 -lfftw3f -o $@
