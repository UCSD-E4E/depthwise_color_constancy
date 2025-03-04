#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//https://developer.nvidia.com/blog/even-easier-introduction-cuda/
//This is frist for setting my my enviroment correctly
// test

// Kernel function to add the elements of two arrays
  __global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

__global__
void depthwiseColorConsistency(
  double * ds,
  double * output,
  float * depth,
  int width, 
  int height, 
  int channels
)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int cent_d_idx = (y * width + x);
  double eps = 10;  
  double exp_sum = 0;                                            
  
  //TODO: Find better defintion for softmax
  // compute denom for softmax                       
  for (int i = -1; i <= 1; i++) { // TODO generalize this
      for (int j = -1; j <= 1; j++) { //like with a footprint sys
          int nx = x + j;
          int ny = y + i;

          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {                              
              int idx_d = (ny * width + nx);
              exp_sum += expf(-abs(depth[idx_d] - depth[cent_d_idx]) - eps); 
          }               
      }
  } 

  // so now $\text{texp_sum} = \sum_{n \in N} e^{-|n_d - t_d|}}
  // softmax of a neighbor n is $\frac{e^{-|n_d - t_d|}}{exp_sum}$ 

  if (x < width && y < height) {
      // Apply avging to each color channels
      for (int c = 0; c < channels; c++) {  
          int cent_c_idx = (y * width + x) * channels + c;  
          double avg_color = 0;                  
          
          //convolution here
          for (int i = -1; i <= 1; i++) {
              for (int j = -1; j <= 1; j++) {
                  int nx = x + j;
                  int ny = y + i;    
                  
                  // $\sum_{n \in N} softmax(n) * color(n)
                  if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                      int idx = (ny * width + nx) * channels + c;
                      int idx_d = (ny * width + nx);
                      
                      //weight of the softmax by the color
                      double softmax_weight =  expf(-abs(depth[idx_d] - depth[cent_d_idx]) - eps)/exp_sum;
                      avg_color += softmax_weight *  ds[idx]; 
                  }            
              }
          }
          // set the color to the updated color
          output[cent_c_idx] = avg_color;                            
      }                    
  }                      
} 


float* load_bin_files(const char* path, float ** x, size_t element_size, size_t num_elements) {
  FILE *file = fopen(path, "rb");
    if (!file) {
      perror("Failed to open file");
    }
    //TODO Maloc then copy please
    cudaMallocManaged(x, num_elements * element_size);

    fread(*x, element_size, num_elements, file);
    fclose(file);

    return *x;
}

double* load_bin_files(const char* path, double ** x, size_t element_size, size_t num_elements) {
  FILE *file = fopen(path, "rb");
    if (!file) {
      perror("Failed to open file");
    }

    //TODO Maloc then copy please
    cudaMallocManaged(x, num_elements * element_size);

    fread(*x, element_size, num_elements, file);
    fclose(file);

    return *x;
}

int main(void)
{
  // image parameters
  int image_size =  11491200; //number of pixels * number of channels
  int width = 2400;
  int height = 1596;
  int channels = 3; //Ideally RGB for now

  // Init Memory
  float *lim, *depth;
  double *ds, *out;
  lim = load_bin_files("data/T_S04923_lim.bin",&lim, sizeof(float), image_size);
  ds = load_bin_files("data/T_S04923_ds.bin", &ds, sizeof(double), image_size);
  depth = load_bin_files("data/T_S04923_depth.bin",&depth, sizeof(float), 3830400); //should be image_size/3 but i'll handle that later
  cudaMallocManaged(&out, image_size * sizeof(double)); 

  //Debug: Intended to softly check to make sure the data is loading in correctly
  // for (size_t i = 0; i < 10; i++) {
  //   std::cout << lim[i] << std::endl;
  // }
  
  // Conduct Depthwise Operation
  dim3 dimGrid(ceil(width/32), ceil(height/32));
  dim3 dimBlock(32, 32, 1); //Going based from textbook might be a better/more approiate size of block
  depthwiseColorConsistency<<<dimGrid, dimBlock>>>(ds, out, depth, width, height, channels);
  cudaDeviceSynchronize(); //According to https://developer.nvidia.com/blog/even-easier-introduction-cuda/, prevents async weridness

  // Check for errors, based on https://developer.nvidia.com/blog/even-easier-introduction-cuda/
  // here we are trying to see how close we are to the pytorch implmention for finding lim
  float maxError = 0.0f;
  for (int i = 0; i < image_size; i++)
    maxError = fmax(maxError, fabs(out[i]-lim[i]));
  std::cout << "Max error: " << maxError << std::endl;
  // Update this techically is wrong
  // out isn't the illumanite map, its a_c
  // I'm rn testing this by making a new bin file for out
  // Then bringing this back to python

  //write the output for the new lim to test out!
  FILE* out_f = fopen("data/T_S04923_a_c.bin", "wb");
  fwrite(out, sizeof(double), image_size, out_f);
  fclose(out_f);

  // Free memory
  cudaFree(lim);
  cudaFree(ds);
  cudaFree(depth);
  cudaFree(out);
  
  return 0;
}