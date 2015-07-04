/* Henry Cooney <hcooney@pdx.edu>
 * 2015-06-04
 *
 * Conway's game of life, computed on the GPU. Header file.
 */
   
#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <X11/Xlib.h> // for the graphics

#define WIDTH 800
#define HEIGHT 600


// The two boards 
int current[WIDTH * HEIGHT];
int next[WIDTH * HEIGHT];

// Function prototypes 

int main();
void fill_board(int* board);
void step();
void animate();
int test_gol();
void copy_array(int* dest, int* src, int n);
int are_arrays_equal(int* a, int* b, int n);
void assertEqual(int thing1, int thing2, char* message);
int* gpu_compute(int* initial, int height, int width, int timesteps);
