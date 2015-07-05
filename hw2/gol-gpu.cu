/* Henry Cooney - CS510, Accel. Comp. - 4 July 2015

   Conway's game of life, computed on the GPU.
  
   Uses a tiled convolution pattern to achieve good performance
   (hopefully)
*/


#include "gol.h"

int main() {
  
  test_gol(1, HEIGHT, WIDTH);
  //  fill_board
  //gpu_compute(current, HEIGHT, WIDTH, 1);
  return 0;
}


int* gpu_compute(int* initial, int height, int width, int timesteps) {
  // Does GoL on the GPU. Initial is the starting
  // matrix (it is not modified.) The resulting matrix after timesteps
  // iterations is returned.

  printf("Launching GPU computation for %d timesteps... \n", timesteps);
  int n = width * height;
  int* result = (int*) malloc(sizeof(int) * n);
  int* current_dev,* next_dev;


  //  int tester[n];
  //int i;
  //for(i=0; i < n; ++i)
  // tester[i] = 0;
  
  // Memory transfer
  printCudaError(cudaMalloc((void**) &current_dev, sizeof(int)*n));
  printCudaError(cudaMalloc((void**) &next_dev, sizeof(int)*n));
  
  cudaThreadSynchronize(); // is this necessary? 
  printCudaError(cudaMemcpy(current_dev, initial, sizeof(int)*n, cudaMemcpyHostToDevice));
  
  // Establish dimms - these are for GTX 645

  dim3 dimBlock(TW, TW, 1);
  dim3 dimGrid(divideRoundUp(width, ETW), divideRoundUp(height, ETW), 1);

  printf("Matrix size (width x height): %d x %d\n", width, height);
  printf("Block dims (x, y, z): %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
  printf("Grid dims (x, y, z): %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
  
  printf("Starting kernel... \n");
  conway_step_kernel<<<dimGrid, dimBlock>>>(current_dev, next_dev, height, width);

  printf("Kernel done. \n");

  // Copy memory back and free GPU mem
  printCudaError(cudaMemcpy(result, next_dev, sizeof(int)*n, cudaMemcpyDeviceToHost));
  cudaFree(current_dev);
  cudaFree(next_dev);

  return result;

}


__global__ 
void conway_step_kernel(int* current_dev, int* next_dev, 
			int height, int width) {
  // Advances the game of life one timestep.
  // current_dev is the initial matrix, it is not modified. next_dev 
  // the next timestep (the result)
  
  __shared__ int dsm[TW][TW]; // Device Shared Memory
  
  // Each thread is responsbile for a. fetching one item
  // from global memory and b. writing one item to output matrix.

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Each output pixel requires knowledge of neighboring pixels.
  // Thus, each tile has 'egde pixels' which are loaded into shared mem
  // but not written to. Values must be shifted by one to compensate for this

  int row = by*ETW + ty - 1;
  int col = bx*ETW + tx - 1;

  dsm[ty][tx] = current_dev[row*width + col];

  __syncthreads();
  
  if (row >= 0 && row < height && col >= 0 && col < width) {
    // This pixel is not an edge pixel, so write it.
    next_dev[row*width + col] = dsm[ty][tx];
  }
 
  return;
}

void printCudaError(cudaError_t err) {
  // Checks the value of input error. If it does not
  // indicate success, prints an error message.
  
  if(err != cudaSuccess) {
    printf("**** CUDA ERROR: ****\n");
    printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  
}


int divideRoundUp(int a, int b) {
  // Divides a by b, but rounds the result up instead of down.
  return (a+(b-1)) / b;
}
