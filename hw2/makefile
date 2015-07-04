
all: 	gol

gol:	gol.h gol-cpu.o gol-gpu.o gol-gpu-tests.o
	nvcc --compiler-options -Wall -o gol gol-cpu.o gol-gpu.o gol-gpu-tests.o -lX11

gol-cpu.o: gol-cpu.c
	nvcc --compiler-options -Wall -c gol-cpu.c

gol-gpu.o: gol-gpu.c
	nvcc --compiler-options -Wall -c gol-gpu.c

gol-gpu-tests.o: gol-gpu-tests.c
	nvcc --compiler-options -Wall -c gol-gpu-tests.c

clean:
	rm *~ *.o
