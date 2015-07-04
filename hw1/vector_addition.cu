#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int are_vectors_equal(int* a, int* b, int n);

/* The old-fashioned CPU-only way to add two vectors */
void add_vectors_host(int *result, int *a, int *b, int n) {
    for (int i=0; i<n; i++)
        result[i] = a[i] + b[i];
}

/* The kernel that will execute on the GPU */
__global__ void add_vectors_kernel(int *result, int *a, int *b, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // If we have more threads than the magnitude of our vector, we need to
    // make sure that the excess threads don't try to save results into
    // unallocated memory.
    if (idx < n)
        result[idx] = a[idx] + b[idx];
}

/* This function encapsulates the process of creating and tearing down the
 * environment used to execute our vector addition kernel. The steps of the
 * process are:
 *   1. Allocate memory on the device to hold our vectors
 *   2. Copy the vectors to device memory
 *   3. Execute the kernel
 *   4. Retrieve the result vector from the device by copying it to the host
 *   5. Free memory on the device
 */
void add_vectors_dev(int *result, int *a, int *b, int n) {
    // Step 1: Allocate memory
    int *a_dev, *b_dev, *result_dev;

    // Since cudaMalloc does not return a pointer like C's traditional malloc
    // (it returns a success status instead), we provide as it's first argument
    // the address of our device pointer variable so that it can change the
    // value of our pointer to the correct device address.
    cudaMalloc((void **) &a_dev, sizeof(int) * n);
    cudaMalloc((void **) &b_dev, sizeof(int) * n);
    cudaMalloc((void **) &result_dev, sizeof(int) * n);

    // Step 2: Copy the input vectors to the device
    cudaError_t err = cudaMemcpy(a_dev, a, sizeof(int) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      printf("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    cudaMemcpy(b_dev, b, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Step 3: Invoke the kernel
    // We allocate enough blocks (each 512 threads long) in the grid to
    // accomodate all `n` elements in the vectors. The 512 long block size
    // is somewhat arbitrary, but with the constraint that we know the
    // hardware will support blocks of that size.
    dim3 dimGrid((n + 512 - 1) / 512, 1, 1);
    dim3 dimBlock(512, 1, 1);
    add_vectors_kernel<<<dimGrid, dimBlock>>>(result_dev, a_dev, b_dev, n);

    // Step 4: Retrieve the results
    cudaMemcpy(result, result_dev, sizeof(int) * n, cudaMemcpyDeviceToHost);

    // Step 5: Free device memory
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(result_dev);
}

void print_vector(int *array, int n) {
    int i;
    for (i=0; i<n; i++)
        printf("%d ", array[i]);
    printf("\n");
}

int main(void) {
    int n = 5; // Length of the arrays
    int a[] = {0, 1, 2, 3, 4};
    int b[] = {5, 6, 7, 8, 9};
    int host_result[5];
    int device_result[5];
    int l, i;
    int* rand_a, *rand_b, *rand_host_result, *rand_device_result;
    clock_t start, stop;
    double gpu_time, cpu_time;

    printf("Please enter vector length: ");
    scanf("%d", &l);

    rand_a = (int*) malloc(sizeof(int)*l);
    rand_b = (int*) malloc(sizeof(int)*l);
    rand_host_result = (int*) malloc(sizeof(int)*l);
    rand_device_result = (int*) malloc(sizeof(int)*l);

    printf("The CPU's answer: ");
    add_vectors_host(host_result, a, b, n);
    print_vector(host_result, n);
    
    printf("The GPU's answer: ");
    add_vectors_dev(device_result, a, b, n);
    print_vector(device_result, n);

    printf("Generating vectors of length %d... \n", l);

    for(i=0; i<l; ++i) {
      rand_a[i] = rand() % 10;
      rand_b[i] = rand() % 10;
      //printf("%d: %d, %d \n", i, rand_a[i], rand_b[i]);
    }
    
    start = clock();
    add_vectors_host(rand_host_result, rand_a, rand_b, l);
    stop = clock();
    cpu_time = (double) (stop-start)/CLOCKS_PER_SEC;

    start = clock();
    add_vectors_dev(rand_device_result, rand_a, rand_b, l);
    stop = clock();
    gpu_time = (double) (stop-start)/CLOCKS_PER_SEC;

    
    //print_vector(rand_host_result, l);
    printf("CPU compute time: %f", cpu_time);
    printf("\n");
    printf("GPU compute time: %f", gpu_time);
    printf("\n");
    printf("Ratio: %f", cpu_time / gpu_time);
    printf("\n");

    if(!are_vectors_equal(rand_host_result, rand_device_result, l)) {
      printf("WARNING! Host and device results do not agree");
    }
    
    free(rand_a);
    free(rand_b);
    return 0;
}


int are_vectors_equal(int* a, int* b, int n) {
  // Return 1 if vectors a and be are equal, else return 0.
  int i;
  for (i=0; i<n; ++i) {
    if (a[i] != b[i])
	  return 0;
  }
  return 1;
}
