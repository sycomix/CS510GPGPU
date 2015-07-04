
all: 	gol

gol:	gol.h gol-cpu.o gol-gpu.o gol-gpu-tests.o
	gcc -Wall -o gol gol-cpu.o gol-gpu.o -lX11

gol-cpu.o: gol-cpu.c	
	gcc -Wall -c gol-cpu.c

gol-gpu.o: gol-gpu.c	
	gcc -Wall -c gol-gpu.c

gol-gpu-tests.o: gol-gpu-tests.c
	gcc -Wall -c gol-gpu-tests.c

clean:
	rm *~ *.o
