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
  int tw = 32; // Tile width -- 32 will mean 1024 threads/block.
  // This is the maximum t/blk for the GTX 645
  //int blockThreads = tw*tw;
  int tester[n];
  int i;
  for(i=0; i < n; ++i)
    tester[i] = 0;
  
  // Memory transfer
  printCudaError(cudaMalloc((void**) &current_dev, sizeof(int)*n));
  printCudaError(cudaMalloc((void**) &next_dev, sizeof(int)*n));
  //  printMatrix(tester, height, width);
  
  cudaThreadSynchronize();
  printCudaError(cudaMemcpy(current_dev, tester, sizeof(int)*n, cudaMemcpyHostToDevice));
  
  // Establish dimms - these are for GTX 645

  dim3 dimBlock(tw, tw, 1);
  dim3 dimGrid(divideRoundUp(width, tw), divideRoundUp(height, tw), 1);

  printf("Matrix size (width x height): %d x %d\n", width, height);
  printf("Block dims (x, y, z): %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
  printf("Grid dims (x, y, z): %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
  
  printf("Starting kernel... \n");
  conway_step_kernel<<<dimGrid, dimBlock>>>(current_dev, next_dev, height, width, tw);
  printf("Kernel done. \n");
  return NULL;
}


__global__ 
void conway_step_kernel(int* current_dev, int* next_dev, int height, int widh, int tw) {
  // Advances the game of life one timestep.
  // current_dev is the initial matrix, it is not modified. next_dev 
  // the next timestep (the result)
  
  __shared__ int dsm[tw][tw]; // Device Shared Memory
  
  // Each thread is responsbile for a. fetching one item
  // from global memory and b. writing one item to output matrix.

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by*tw + ty;
  int col = bx*tw + tx;

  dsm[ty][tx] = current_dev[row*width + col];
  if(bx < 1 && by < 1)
    printf("tx: ty: Loaded: %d\n",  )

  
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
