#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IN_TILE_WIDTH 32   
#define KERNEL_SIZE 1      
#define OUT_TILE_WIDTH (IN_TILE_WIDTH - 2 * KERNEL_SIZE)  
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
  float * ds,
  float * output,
  float * depth,
  int width, 
  int height, 
  int channels
)
{
  //Prep for tiling both depth and ds to preprocess some data
  // Tiled Convolution implmenetation based on 
  // Programming Massively Parallel Processors
  __shared__ float ds_tile[IN_TILE_WIDTH][IN_TILE_WIDTH][3];
  __shared__ float depth_tile[IN_TILE_WIDTH][IN_TILE_WIDTH];

  int col = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x - KERNEL_SIZE;
  int row = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y - KERNEL_SIZE;
  int pixel_idx = (row * width + col) * channels;
  if (row>=0 && row < height && col >= 0 && col < width) {
    depth_tile[threadIdx.y][threadIdx.x] = depth[row * width + col];
    ds_tile[threadIdx.y][threadIdx.x][0] = ds[pixel_idx];
    ds_tile[threadIdx.y][threadIdx.x][1] = ds[pixel_idx + 1];
    ds_tile[threadIdx.y][threadIdx.x][2] = ds[pixel_idx + 2];
  } else {
    depth_tile[threadIdx.y][threadIdx.x] = 0; //incase of access, this goes to 0
    ds_tile[threadIdx.y][threadIdx.x][0] = 1;
    ds_tile[threadIdx.y][threadIdx.x][1] = 0;
    ds_tile[threadIdx.y][threadIdx.x][2] = 0;
  }
  
  __syncthreads();

  int x = threadIdx.x;
  int y = threadIdx.y;
  float eps = 10;  
  float exp_sum = 0;  
  
  
  if (row>=0 && row < height && col >= 0 && col < width) {
    if(x>= KERNEL_SIZE && x<=OUT_TILE_WIDTH + KERNEL_SIZE && y>= KERNEL_SIZE && y<=OUT_TILE_WIDTH + KERNEL_SIZE) {
      //TODO: Find better defintion for softmax
      // compute denom for softmax                       
      for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) { // TODO generalize this
        for (int j = -KERNEL_SIZE; j <= KERNEL_SIZE; j++) { //like with a footprint sys
          int nx = x + j;
          int ny = y + i;
          if (ny>=0 && ny < height && nx >= 0 && nx < width) {
            if (nx >= 0 && nx < IN_TILE_WIDTH && ny >= 0 && ny < IN_TILE_WIDTH) {                              
                exp_sum += expf(-abs(depth_tile[ny][nx] - depth_tile[y][x]) - eps); 
            }  
          }             
        }
      } 
      // so now $\text{texp_sum} = \sum_{n \in N} e^{-|n_d - t_d|}}
      // softmax of a neighbor n is $\frac{e^{-|n_d - t_d|}}{exp_sum}$ 

      // Apply avging to each color channels
      for (int c = 0; c < channels; c++) {  
        float avg_color = 0;                  
        
        //convolution here
        for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
          for (int j = -KERNEL_SIZE; j <= KERNEL_SIZE; j++) {
            int nx = x + j;
            int ny = y + i;    
            
            // $\sum_{n \in N} softmax(n) * color(n)
            if (ny>=0 && ny < height && nx >= 0 && nx < width) {
              if (nx >= 0 && nx < IN_TILE_WIDTH && ny >= 0 && ny < IN_TILE_WIDTH) {                      
                //weight of the softmax by the color
                float softmax_weight =  expf(-abs(depth_tile[ny][nx] - depth_tile[y][x]) - eps)/exp_sum;
                avg_color += softmax_weight *  ds_tile[ny][nx][c]; 
              }        
            }    
          }
        }
        // set the color to the updated color
        output[pixel_idx + c] = avg_color;                                              
      } 
    }
  }                       
} 


float* load_bin_files(const char* path, float ** d_x, size_t element_size, size_t num_elements) {
  FILE *file = fopen(path, "rb");
    if (!file) {
      perror("Failed to open file");
    }

    float *h_temp = (float*)malloc(num_elements * element_size);
    cudaMalloc(d_x, num_elements * element_size);

    fread(h_temp, element_size, num_elements, file);

    cudaMemcpy(*d_x, h_temp, num_elements * element_size, cudaMemcpyHostToDevice);
    free(h_temp);
    fclose(file);

    return *d_x;
}

double* load_bin_files(const char* path, double ** d_x, size_t element_size, size_t num_elements) {
  FILE *file = fopen(path, "rb");
    if (!file) {
      perror("Failed to open file");
    }

    double *h_temp = (double*)malloc(num_elements * element_size);
    fread(h_temp, element_size, num_elements, file);
    // std::cout << "test data file load "<< std::endl;
    // std::cout << h_temp[0] << std::endl;

    cudaMalloc(d_x, num_elements * element_size);
    cudaMemcpy(*d_x, h_temp, num_elements * element_size, cudaMemcpyHostToDevice);
    free(h_temp);
    fclose(file);

    return *d_x;
}

int main(void)
{
  // image parameters
 //number of pixels * number of channels

  int iterations = 2000;
  float alpha = 0.99f;
  float beta = 1.f - alpha;

  int width = 640; //480,640
  int height = 480;
  int channels = 3; //Ideally RGB for now
  int num_pixels =  width * height * 3;
  char * image_path = "data/realsense_tests/living_room_0046b_out_1-color.bin";
  char * depth_path = "data/realsense_tests/living_room_0046b_out_1-depth.bin";
  char * output_path = "data/realsense_tests/living_room_0046b_out_1-lim-99-500.bin";
  

  // Init Memory
  float *depth, *ds, *a_c, *d_out;
  ds = load_bin_files(image_path, &ds, sizeof(float), num_pixels);
  a_c = load_bin_files(image_path, &ds, sizeof(float), num_pixels);
  depth = load_bin_files(depth_path,&depth, sizeof(float), width * height); //should be image_size/3 but i'll handle that later
  
  float *h_out = (float*)malloc(num_pixels * sizeof(float));
  cudaMalloc(&d_out, num_pixels * sizeof(float));
  
  // //Debug: Intended to softly check to make sure the data is loading in correctly
  // cudaMemcpy(h_out, a_c, num_pixels * sizeof(double), cudaMemcpyDeviceToHost);
  // for (size_t i = 0; i < 10; i++) {
  //   std::cout << h_out[i] << std::endl;
  // }
  
  dim3 dimGrid(ceil((width + IN_TILE_WIDTH)/IN_TILE_WIDTH) + 10, ceil((height + IN_TILE_WIDTH + 1)/IN_TILE_WIDTH) + 10);
  dim3 dimBlock(IN_TILE_WIDTH, IN_TILE_WIDTH); //Going based from textbook might be a better/more approiate size of block

  //Gemm Convoltuon Implementation
  cublasHandle_t handle; 
  cublasCreate(&handle);

  // Conduct Depthwise Operation
  for (size_t i = 0; i < iterations; i++) { 
    depthwiseColorConsistency<<<dimGrid, dimBlock>>>(a_c, d_out, depth, width, height, channels);
    //https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication#:~:text=As%20you%20said%2C%20cuBLAS%20interprets,for%20the%20column%2Dmajor%20interpretation.
    //According to here, we can just do the tranpose instead. I'm fine with that. 
    // std::cout << "test output before geam after depthwise " << i << std::endl;
    // std::cout << d_out[0] << std::endl;

    checkCublas(cublasSgeam(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, 
      channels, height * width,
      &alpha, d_out, channels,
      &beta, ds, channels,
      a_c, channels
    ), "geam issues");
    // std::cout << "test output after geam " << i << std::endl;
    // std::cout << a_c[0] << std::endl;
  }

  cudaDeviceSynchronize();
  //write the output for the new lim to test out!
  FILE* out_f = fopen(output_path, "wb");
  cudaMemcpy(h_out, a_c, num_pixels * sizeof(float), cudaMemcpyDeviceToHost);
  fwrite(h_out, sizeof(float), num_pixels, out_f);
  
  fclose(out_f);

  // Free memory
  cublasDestroy(handle);
  cudaFree(ds);
  cudaFree(depth);
  cudaFree(d_out);
  cudaFree(a_c);
  free(h_out);
  
  return 0;
}