
CC=g++
CFLAGS=-g -fopenmp -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $$(sdl2-config --cflags) -I.. -I./basics
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf -lfftw3 -lfftw3f


.PHONY: clean all

all: examples.a

clean:
	rm -f *.exe *.o *.a


#createAtomCluster.o: createAtomCluster.cpp createAtomCluster.h
#	$(CC) -c $(CFLAGS) $< -o $@
createSlit.o: createSlit.cpp createSlit.h
	$(CC) -c $(CFLAGS) $< -o $@


examples.a: createSlit.o
	ar rcsT $@ $^
