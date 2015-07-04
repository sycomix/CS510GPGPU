
all: 	gol

gol: 	gol-cpu.cu gol-gpu.cu gol-gpu-tests.cu
	nvcc -o gol gol-gpu.cu gol-gpu-tests.cu gol-cpu.cu -lX11	
# gol:	gol.h gol-cpu.o gol-gpu.o gol-gpu-tests.o
# 	nvcc -o gol gol-cpu.o gol-gpu.o gol-gpu-tests.o -lX11

# gol-cpu.o: gol-cpu.cu
# 	nvcc --compiler-options -c gol-cpu.cu

# gol-gpu.o: gol-gpu.cu
# 	nvcc --compiler-options -c gol-gpu.cu

# gol-gpu-tests.o: gol-gpu-tests.cu
# 	nvcc --compiler-options -c gol-gpu-tests.cu

clean:
	rm *~ *.o gol a.out
