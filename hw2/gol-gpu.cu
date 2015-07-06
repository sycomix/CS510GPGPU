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
  cudaDeviceSynchronize();
  // Check for errors from the kernel:
  if(cudaGetLastError() != cudaSuccess) {
    printf("*** ERROR IN KERNEL *** \n");
    exit(1);
  }
  
  // Copy memory back and free GPU mem
  printCudaError(cudaMemcpy(result, next_dev, sizeof(int)*n, cudaMemcpyDeviceToHost));
  printCudaError(cudaFree(current_dev));
  printCudaError(cudaFree(next_dev));

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
  int i, ii;

  // Each output pixel requires knowledge of neighboring pixels.
  // Thus, each tile has 'egde pixels' which are loaded into shared mem
  // but not written to. Values must be shifted by one to compensate for this
  // Use a modulus to 'wrap around' grid edges
  int row = (by*ETW + ty - 1);
  int col = (bx*ETW + tx - 1);
  int this_pixel = current_dev[row%height*width + col%width];
  
  dsm[ty][tx] = this_pixel;

  __syncthreads();
  
  if (row >= 0 && row < height && col >= 0 && col < width 
      && tx > 1 && tx < 30 && ty > 1 && ty < 30) {
    // This pixel is not an edge pixel, so figure out its value
    // in the next frame, and write it.
    
    // num_neighbors is the sum of all the neighboring cells. Since
    // the loop will pass through this pixel, I negate this pixels value.
    // Thus, this pixel does not contribute to the overall sum.
    int num_neighbors = -this_pixel; 

    for(i=-1; i<2; ++i) {
	for(ii=-1; ii<2; ++ii) {
	  num_neighbors += dsm[ty+i][tx+ii];
	  if(bx == 0 && by == 0 && tx == 3 && ty == 3) 
	    printf("%d", dsm[ty+i][tx+ii]);
	}
	if(bx == 0 && by == 0 && tx == 3 && ty == 3) 
	  printf("\n");
    }


    int next;
    if(num_neighbors == 2 || num_neighbors == 3)
      next = 1;
    else if(this_pixel && num_neighbors == 3)
      next = 1;
    else
      next = 0;
    
    /*
    next_dev[row*width + col] = (((num_neighbors==2 || num_neighbors==3) \
				 || (num_neighbors==3 && this_pixel==1)) ? \
				 1 : 0); */
    next = 0;
    next_dev[row*width + col] = next;
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
