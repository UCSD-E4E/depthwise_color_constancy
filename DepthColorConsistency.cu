#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

//https://developer.nvidia.com/blog/even-easier-introduction-cuda/
//This is frist for setting my my enviroment correctly
// test

void checkCublas(cublasStatus_t result, const char* msg) {
  if (result != CUBLAS_STATUS_SUCCESS) {
      std::cerr << msg << std::endl;
      exit(EXIT_FAILURE);
  }
}

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
 //number of pixels * number of channels

  int iterations = 2000; //200
  double alpha = 0.99f; // 0.7
  double beta = 1.f - alpha;

  int width = 640; //480,640
  int height = 480;
  int channels = 3; //Ideally RGB for now
  int num_pixels =  width * height * 3;
  char * image_path = "data/realsense_tests/living_room_0046b_out_1-color.bin";
  char * depth_path = "data/realsense_tests/living_room_0046b_out_1-depth.bin";
  char * output_path = "data/realsense_tests/living_room_0046b_out_1-lim-99-500.bin";
  

  // Init Memory
  float *depth;
  double *ds, *a_c, *out;
  ds = load_bin_files(image_path, &ds, sizeof(double), num_pixels);
  a_c = load_bin_files(image_path, &ds, sizeof(double), num_pixels);
  depth = load_bin_files(depth_path,&depth, sizeof(float), width * height); //should be image_size/3 but i'll handle that later
  cudaMallocManaged(&out, num_pixels * sizeof(double)); 

  //Debug: Intended to softly check to make sure the data is loading in correctly
  // for (size_t i = 0; i < 10; i++) {
  //   std::cout << ds[i] << std::endl;
  // }
  
  dim3 dimGrid(ceil(width/32), ceil(height/32));
  dim3 dimBlock(32, 32, 1); //Going based from textbook might be a better/more approiate size of block

  //Gemm Convoltuon Implementation
  cublasHandle_t handle; 
  cublasCreate(&handle);

  // Conduct Depthwise Operation
  for (size_t i = 0; i < iterations; i++) { 
    depthwiseColorConsistency<<<dimGrid, dimBlock>>>(a_c, out, depth, width, height, channels);
    cudaDeviceSynchronize(); //According to https://developer.nvidia.com/blog/even-easier-introduction-cuda/, prevents async weridness

    //https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication#:~:text=As%20you%20said%2C%20cuBLAS%20interprets,for%20the%20column%2Dmajor%20interpretation.
    //According to here, we can just do the tranpose instead. I'm fine with that. 
    // std::cout << "test output before geam " << i << std::endl;
    // std::cout << a_c[0] << std::endl;

    checkCublas(cublasDgeam(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, 
      channels, height * width,
      &alpha, out, channels,
      &beta, ds, channels,
      a_c, channels
    ), "geam issues");

    cudaDeviceSynchronize();

    // std::cout << "test output after geam " << i << std::endl;
    // std::cout << a_c[0] << std::endl;
  }

  //write the output for the new lim to test out!
  FILE* out_f = fopen(output_path, "wb");
  fwrite(out, sizeof(double), num_pixels, out_f);
  fclose(out_f);

  // Free memory
  cublasDestroy(handle);
  cudaFree(ds);
  cudaFree(depth);
  cudaFree(out);
  
  return 0;
}