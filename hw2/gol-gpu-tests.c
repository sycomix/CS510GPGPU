/* Henry Cooney - CS510, Accel. Comp. - 4 July 2015

   Conway's game of life, computed on the GPU -- Tests
   
   This file includes tests functions for GoL.
*/



#include "gol.h"

int test_gol(int timesteps, int width, int height) {
  // Tests the GPU GOL computation against the known good
  // CPU computation. Compares final results after timesteps iterations.
  // width and height are the dimensions of the GOL matrix.
  // 
  // Returns 1 if CPU and GPU results agree, else 0.
  

  printf("test_gol launched at %d timesteps \n", timesteps);
  int gpu_initial[width*height];
  int n = width*height;


  // Compute CPU result:
  fill_board(current);
  copy_array(gpu_initial, current, n);
  assertEqual(are_arrays_equal(current, gpu_initial, n), 1, "Testing array copy");
  
  int* result = gpu_compute(gpu_initial, height, width, timesteps);
  
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


void assertEqual(int thing1, int thing2, char* message) {
  // Asserts whether thing1 and thing2 are equal. If they are not,
  // outputs message and exits the program.
  // If you don't want to include a message, just make message nullx

  if(thing1 != thing2) {
    printf("ERROR: %d does not equal %d \n", thing1, thing2);
    if(message)
      printf("Message: %s \n", message);
    
    exit(1);
  }
}
