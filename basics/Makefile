
CC=g++
# Not exactly sure why, but sdl2-config only works with $$, not with $
CFLAGS=-g -std=c++11 -Wall -Wextra -Wshadow -Wno-unused-parameter $$(sdl2-config --cflags)
LIBS=$$(sdl2-config --libs) -l SDL2_image -l SDL2_ttf

.PHONY: clean all

all: main.exe

clean:
	rm -f main.exe gaussian.o

gaussian.o: gaussian.cpp gaussian.h
	rm -f $@
	$(CC) -c $(CFLAGS) $< $(LIBS) -o $@

main.exe: main.cpp gaussian.o sdlplot.h sdlplot.cpp dcft.h dcft.cpp sdlcommon.cpp sdlcommon.h
	rm -f $@
	$(CC) $(CFLAGS) $< $(LIBS) gaussian.o -o $@
