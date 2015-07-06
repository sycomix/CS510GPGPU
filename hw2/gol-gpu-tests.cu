/* Henry Cooney - CS510, Accel. Comp. - 4 July 2015

   Conway's game of life, computed on the GPU -- Tests
   
   This file includes tests functions for GoL.
*/


//#include "gol-cpu.cu"
#include "gol.h"

int test_gol(int timesteps, int height, int width) {
  // Tests the GPU GOL computation against the known good
  // CPU computation. Compares final results after timesteps iterations.
  // width and height are the dimensions of the GOL matrix.
  // 
  // Returns 1 if CPU and GPU results agree, else 0.
  

  printf("test_gol launched at %d timesteps \n", timesteps);
  int gpu_initial[width*height];
  int n = width*height;


  // Compute CPU result:
  fill_board(getCPUCurrent());
  step(); // now next has the next board...
  copy_array(gpu_initial, getCPUCurrent(), n);
  assertEqual(are_arrays_equal(getCPUCurrent(), gpu_initial, n), 1, "Testing array copy");
  
  int* result = gpu_compute(gpu_initial, height, width, timesteps);
  // assertEqual(are_arrays_equal(gpu_initial, result, n), 1, "Kernel that does nothing should return same array.");

  printf("Initial: \n");
  printMatrixWindow(gpu_initial, HEIGHT, WIDTH, 10, 10);

  printf("Result: \n");
  printMatrixWindow(result, HEIGHT, WIDTH, 10, 10);

  printf("CPU: \n");
  printMatrixWindow(getCPUNext(), HEIGHT, WIDTH, 10, 10);


  assertArraysEqual(getCPUNext(), result, height, width);
  assertArraysEqual(getCPUCurrent(), gpu_initial, height, width);
  
  return 0;
}


void copy_array(int* dest, int* src, int n) {
  // Copies identically sized arrays. Contents of src are copied
  // into dest.
  
  int i;
  for(i=0; i<n; ++i) 
    dest[i] = src[i];
}


int are_arrays_equal(int* a, int* b, int n) {
  // Returns 1 if integer arrays a and b are equal, else returns 0.
  
  int i;
  for(i=0; i<n; ++i) {
    if (a[i] != b[i])
      return 0;
    }
  
  return 1;
}


int assertEqual(int thing1, int thing2, char* message) {
  // Asserts whether thing1 and thing2 are equal. If they are not,
  // outputs message and returns 0 (if they are equal, returns 1.)
  // If you don't want to include a message, just make message null

  if(thing1 != thing2) {
    printf("ERROR: %d does not equal %d \n", thing1, thing2);
    if(message)
      printf("Message: %s \n", message);
    
    //  exit(1);
    return 0;
  }
  return 1;
}

int assertArraysEqual(int* arr1, int* arr2, int height, int width) {
  // Tests whether two arrays are equal. 
  // If they are not, prints a warning message and the row/col 
  // where the first issue was found.
  // Returns 1 if they are equal, else returns 0.

  int i, ii;

  for (i=0; i<height; ++i)
    for(ii=0; ii<width; ++ii){
      if (arr1[i*width + ii] != arr2[i*width+ii]) {
	printf("ERROR: Arrays do not match at [%d][%d]\n", ii, i);
	return 0;
      }
    }
  return 1;
}


void printMatrix(int* mat, int height, int width) {
  // Print the matrix mat.

  int i, ii;
  
  for(i=0; i<height; ++i) {
    for(ii=0; ii<width; ++ii) {
      printf("%d", mat[i*width + ii]);
    }
    printf("\n");
  }
}


void printMatrixWindow(int* mat, int height, int width, int rows, int cols){
  // Prints a submatrix ('window') of the matrix mat.
  // Prints the submatrix in the upper left corner of mats, with 
  // dimension rows x cols.
  
  int i, ii;
  for(i = 0; i < rows; ++i) {
    for(ii=0; ii < cols; ++ii) {
      printf("%d", mat[i*width + ii]);
    }
    printf("\n");
  }
}
