all: 	vector_addition_timed

vector_addition_timed:	vector_addition_timed.cu
	nvcc -o vector_addition_timed vector_addition_timed.cu

clean:
	rm *~ *.o vector_addition_timed a.out
