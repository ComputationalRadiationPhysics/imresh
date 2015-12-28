
CC=g++-4.9
INC=-I. -I..
SDLLIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf
WARNINGS=-Wall -Wextra -Wshadow -Wno-unused-parameter
CFLAGS=-g -fopenmp -std=c++11 $$(sdl2-config --cflags) $(INC) $(WARNINGS)



.PHONY: clean all colors.a

all: sdlcommons.a

clean:
	rm -f *.exe *.o *.a


colors.a:
	make -C ../colors all


sdlcommon.o: sdlcommon.cpp sdlcommon.h
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

sdlplot.o: sdlplot.cpp sdlplot.h sdlplot.tpp sdlcommon.o
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -o $@

complexPlot.o: complexPlot.cpp complexPlot.h sdlplot.o colors.a
	$(CC) -c $(CFLAGS) $< $(SDLLIBS) -lfftw3 -o $@


sdlcommons.a: complexPlot.o sdlplot.o sdlcommon.o
	ar rcsT $@ $^
