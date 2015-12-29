
CC=g++-4.9
INC=-I.
WARNINGS=-Wall -Wextra -Wshadow -Wno-unused-parameter
CFLAGS=-std=c++11 $$(sdl2-config --cflags) $(INC) $(WARNINGS)

.PHONY: clean all

all: colors.a

clean:
	rm -f main.exe *.o *.a

conversions.o: conversions.cpp conversions.h
	$(CC) -c $(CFLAGS) $< -o $@

x11Colors.o: x11Colors.cpp x11Colors.h
	$(CC) -c $(CFLAGS) $< -o $@

colors.a: x11Colors.o conversions.o
	ar rcs colors.a $^
