/* Henry Cooney <hcooney@pdx.edu>
 * 2015-06-04
 *
 * Conway's game of life, computed on the GPU. Header file.
 */
   
#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <X11/Xlib.h> // for the graphics
#include <cuda.h>
#include <driver_types.h> // for cudaError_t and other CUDA types
#include <vector_types.h> // for dim3, etc

#define WIDTH 800
#define HEIGHT 600

#define TW 32 // Tile Width
#define MASK_WIDTH 1 // Each pixel must know about other pixels up to 1
// step away.
#define ETW 30 // The "effective tile width". This is how many pixels
// are actually written by each tile (the edge pixels are loaded into shared
// memory but not written to the result matrix.)


// Function prototypes 

int main();
void fill_board(int* board);
void step();
void animate();
int* getCPUCurrent();
int* getCPUNext();
int test_gol(int timesteps, int width, int height);
void copy_array(int* dest, int* src, int n);
int are_arrays_equal(int* a, int* b, int n);
int assertEqual(int thing1, int thing2, char* message);
int assertArraysEqual(int* arr1, int* arr2, int height, int width);
int* gpu_compute(int* initial, int height, int width, int timesteps);
void printCudaError(cudaError_t err);
void printMatrix(int* mat, int height, int width);
void printMatrixWindow(int* mat, int height, int width, int rows, int cols);
int divideRoundUp(int a, int b);
__global__ void conway_step_kernel(int* current_dev, int* next_dev, 
				   int height, int width);
__global__ void zeroMemory(int* ptr);
