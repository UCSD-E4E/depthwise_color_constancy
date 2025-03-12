#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IN_TILE_WIDTH 32   
#define KERNEL_SIZE 1      
#define OUT_TILE_WIDTH (IN_TILE_WIDTH - 2 * KERNEL_SIZE)  

__global__
void softmaxDepthAdverging(
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


float* load_bin_files(const char* path, size_t element_size, size_t num_elements) {
  FILE *file = fopen(path, "rb");
  if (!file) {
    perror("Failed to open file");
  }

  float *h_temp = (float*)malloc(num_elements * element_size);
  fread(h_temp, element_size, num_elements, file);
  fclose(file);

  return h_temp;
}

float * depthwiseColorConsistency(
  int iterations,
  int image_width,
  int image_height,
  int image_num_channels,
  float alpha,
  float *  h_depth_map_ptr,
  float *  h_image_ptr
) 
{
  float *d_depth_map_ptr, *d_in_image_ptr, *d_temp_ptr, *d_a_c_ptr;
  float beta = 1.f - alpha;
  int num_pixels =  image_width * image_height * image_num_channels;
  int depth_size =  image_width * image_height;

  //depth map of image
  cudaMalloc(&d_depth_map_ptr, depth_size * sizeof(float));
  cudaMemcpy(d_depth_map_ptr, h_depth_map_ptr, depth_size * sizeof(float), cudaMemcpyHostToDevice);
  
  //original image, needs to be kept to keep a_c stable each iteration
  cudaMalloc(&d_in_image_ptr, num_pixels * sizeof(float));
  cudaMemcpy(d_in_image_ptr, h_image_ptr, num_pixels * sizeof(float), cudaMemcpyHostToDevice);

  //a_c is the avged color image each iteration, starts with original image
  cudaMalloc(&d_a_c_ptr, num_pixels * sizeof(float));
  cudaMemcpy(d_a_c_ptr, h_image_ptr, num_pixels * sizeof(float), cudaMemcpyHostToDevice);
  
  //temp image for holding raw illumiant map
  cudaMalloc(&d_temp_ptr, num_pixels * sizeof(float));

  // +10 is a workaround for this missing a column
  // TODO fix this workaround
  dim3 dimGrid(
    ceil((image_width + IN_TILE_WIDTH)/IN_TILE_WIDTH) + 10, 
    ceil((image_height + IN_TILE_WIDTH)/IN_TILE_WIDTH) + 10
  );
  dim3 dimBlock(IN_TILE_WIDTH, IN_TILE_WIDTH);

  //cublas handlers
  cublasHandle_t handle; 
  cublasCreate(&handle);

  // Conduct Depthwise Operation
  for (size_t i = 0; i < iterations; i++) { 
    softmaxDepthAdverging<<<dimGrid, dimBlock>>>(
      d_a_c_ptr, d_temp_ptr, d_depth_map_ptr, 
      image_width, image_height, image_num_channels);
    
      //https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication#:~:text=As%20you%20said%2C%20cuBLAS%20interprets,for%20the%20column%2Dmajor%20interpretation.
    //According to here, we can just do the tranpose instead. I'm fine with that. 
    cublasSgeam(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, 
      image_num_channels, image_height * image_width,
      &alpha, d_temp_ptr, image_num_channels,
      &beta, d_in_image_ptr, image_num_channels,
      d_a_c_ptr, image_num_channels
    );
  }

  cudaDeviceSynchronize();
  
  //write the output for the new lim to test out!
  float *h_out = (float*)malloc(num_pixels * sizeof(float));
  cudaMemcpy(h_out, d_a_c_ptr, num_pixels * sizeof(float), cudaMemcpyDeviceToHost);

  // Free memory
  cublasDestroy(handle);
  cudaFree(d_in_image_ptr);
  cudaFree(d_depth_map_ptr);
  cudaFree(d_temp_ptr);
  cudaFree(d_a_c_ptr);
  return h_out;
}

//for python freeing
void free_array(float* arr) {
  free(arr);
}

//for python freeing
void free_array(double* arr) {
  free(arr);
}

int main(void)
{
  // image parameters
 //number of pixels * number of channels

  int iterations = 1000;
  float alpha = 0.9999f;

  int width = 640; //480,640
  int height = 480;
  int channels = 3; //Ideally RGB for now
  int num_pixels =  width * height * channels;
  char * image_path = "data/realsense_tests/living_room_0046b_out_1-color.bin";
  char * depth_path = "data/realsense_tests/living_room_0046b_out_1-depth.bin";
  char * output_path = "data/realsense_tests/living_room_0046b_out_1-lim.bin";
  

  // Init Memory
  float *depth, *ds;
  ds = load_bin_files(image_path, sizeof(float), num_pixels);
  depth = load_bin_files(depth_path, sizeof(float), width * height);
  
  float* h_out = depthwiseColorConsistency(
    iterations,
    width,
    height,
    channels,
    alpha,
    depth,
    ds
  );

  FILE* out_f = fopen(output_path, "wb");
  fwrite(h_out, sizeof(float), num_pixels, out_f);
  fclose(out_f);
  
  free(h_out);
  free(ds);
  free(depth);
  
  return 0;
}