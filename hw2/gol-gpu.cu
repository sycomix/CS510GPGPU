/* Henry Cooney - CS510, Accel. Comp. - 4 July 2015

   Conway's game of life, computed on the GPU.
  
   Uses a tiled convolution pattern to achieve good performance
   (hopefully)
*/


#include "gol.h"

int main() {
  
  srand(time(NULL));
  int i;
  for(i = 1; i < 5; ++i) {
    if(!test_gol(i, HEIGHT, WIDTH)) {
          printf("**** FAILED ****\n");
	  printf("CPU and GPU results did not agree.\n");
	  exit(1);
    }
  }
   
  printf("OK \n");
  printf("All tests passed! GPU results agree with CPU results.\n");
  
  printf("Starting the Game of Life... \n");
  animate_gpu();

  
 
  return 0;
}

void animate_gpu() {
  // Does the Game of Life on the GPU, and outputs results
  // to an X window. Unfortunately, this method is inefficient,
  // since results are copied back from the GPU every timestep (rather
  // than remaining on the GPU). Oh well :(

  // Display code is from gol.c by Christopher Mitchell (chrism@lclark.edu)

  Display* display;
  display = XOpenDisplay(NULL);
  if (display == NULL) {
    fprintf(stderr, "Could not open an X display.\n");
    exit(-1);
  }
  int screen_num = DefaultScreen(display);

  int black = BlackPixel(display, screen_num);
  int white = WhitePixel(display, screen_num);

  Window win = XCreateSimpleWindow(display,
				   RootWindow(display, screen_num),
				   0, 0,
				   WIDTH, HEIGHT,
				   0,
				   black, white);
  XStoreName(display, win, "The Game of Life");

  XSelectInput(display, win, StructureNotifyMask);
  XMapWindow(display, win);
  while (1) {
    XEvent e;
    XNextEvent(display, &e);
    if (e.type == MapNotify)
      break;
  }

  GC gc = XCreateGC(display, win, 0, NULL);

  int x, y, n;
  XPoint points[WIDTH * HEIGHT];
  // Generate a random board:
  int initial[WIDTH * HEIGHT];
  int* result;

  int i;
  for(i=0; i<WIDTH*HEIGHT; ++i)
    initial[i] = rand() % 2;

  while (1) {
    XClearWindow(display, win);
    // Get the GPU GoL values:
    result = gpu_compute(initial, HEIGHT, WIDTH, 1);
    memcpy(initial, result, sizeof(int)*WIDTH*HEIGHT);
    n = 0;
    for (y=0; y<HEIGHT; y++) {
      for (x=0; x<WIDTH; x++) {
	if (result[y * WIDTH + x]) {
	  points[n].x = x;
	  points[n].y = y;
	  n++;
	}
      }
      free(result);
    }
    XDrawPoints(display, win, gc, points, n, CoordModeOrigin);
    XFlush(display);
  }

}


int* gpu_compute(int* initial, int height, int width, int timesteps) {
  // Does GoL on the GPU. Initial is the starting
  // matrix (it is not modified.) The resulting matrix after timesteps
  // iterations is returned.

  printf("Launching GPU computation for %d timesteps... \n", timesteps);
  int n = width * height;
  int* result = (int*) malloc(sizeof(int) * n);
  int* current_dev,* next_dev;
  int steps_done = 0;
  
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

  // For testing - set GPU memory to 2. If any pixel get missed it should be
  // very visible
  //  setGPUMemory<<<dim3(divideRoundUp(n,512),1,1), dim3(512, 1, 1)>>>
  // (next_dev, 2);
  
  while(steps_done < timesteps) {
    // To make things faster, current_dev and next dev are swapped back
    // and forth. 
    if(steps_done % 2 == 0)
      conway_step_kernel<<<dimGrid, dimBlock>>>
	(current_dev, next_dev, height, width);
    else
      conway_step_kernel<<<dimGrid, dimBlock>>>
	(next_dev, current_dev, height, width);

    ++steps_done;
  }

  printf("Kernel done. \n");
  cudaDeviceSynchronize(); // Necessary??
  // Check for errors from the kernel:
  if(cudaGetLastError() != cudaSuccess) {
    printf("*** ERROR IN KERNEL *** \n");
    exit(1);
  }
  
  // Copy back memory. Make sure we get the right buffer 
  // (since current and next are swapped each frame)
  if(steps_done % 2 == 1)
    printCudaError(cudaMemcpy(result, next_dev, sizeof(int)*n, cudaMemcpyDeviceToHost));
  else
    printCudaError(cudaMemcpy(result, current_dev, sizeof(int)*n, cudaMemcpyDeviceToHost));

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
  int num_neighbors;
  int next = 0;
  int this_pixel;
  

  // Each output pixel requires knowledge of neighboring pixels.
  // Thus, each tile has 'edge pixels' which are loaded into shared mem
  // but not written to. Values are shifted by one to compensate for this
  // Mod arithmetic implements wraparound for pixels that are off the board
  
  int row = (by*ETW + ty + height - 1) % height;
  int col = (bx*ETW + tx + width - 1) % width;
    
  this_pixel = current_dev[row*width + col];
  dsm[ty][tx] = this_pixel;

  __syncthreads();

  if(tx > 0 && tx <= ETW && ty > 0 && ty <= ETW) {
    // This pixel is not an edge pixel, so figure out its value
    // in the next frame, and write it.
    
    // num_neighbors is the sum of all the neighboring cells. Since
    // the loop will pass through this pixel, I negate this pixels value.
    // Thus, this pixel does not contribute to the overall sum.
    num_neighbors = -this_pixel; 
    for(i=-1; i<2; ++i) {
	for(ii=-1; ii<2; ++ii) {
	  num_neighbors += dsm[ty+i][tx+ii];
	}
    }

    if(num_neighbors == 3 || (num_neighbors == 2 && this_pixel))
      next = 1;
	
    next_dev[row*width + col] = next;
  }  
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


__global__
void setGPUMemory(int* ptr, int val) {
  // Sets memory at pointer.
  // For testing, since garbage hanging out in the GPU
  // can cause confusing results
  
  ptr[blockIdx.x*blockDim.x + threadIdx.x] = val;
 
}
