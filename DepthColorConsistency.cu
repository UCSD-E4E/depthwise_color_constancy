#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

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

__global__
void createSoftmaxWeightsKernel(
  float * depth,
  float * unwrappedKernel,
  int width, 
  int height, 
  int k_h,
  int k_w
) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int cent_d_idx = (y * width + x);
  float eps = 10;  
  float exp_sum = 0;  

  int t_h = height - k_h + 1;
  int t_w = width - k_w + 1;

  int index =  x * (height) * width + x + y * (t_w) * height * width + y * (width);

  //stay within bounds of convolution images
  if ((x < t_w) && (y < t_h)) {
    // demoninator 
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

    //numerator and actual kernal item
    for (int i = -1; i <= 1; i++) { // TODO generalize this
      for (int j = -1; j <= 1; j++) { //like with a footprint sys
          int nx = x + j;
          int ny = y + i;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {                              
              int idx_d = (ny * width + nx);
              float softmax_weight = expf(-abs(depth[idx_d] - depth[cent_d_idx]) - eps)/exp_sum; 
          
              //TODO Create a mapping between nx, ny from image space to the kernal matrix
              //given a target value of x,y
              unwrappedKernel[index] = softmax_weight;
              index += 1; //the one added above
              //offset within K_i
          }
      }
      index += width - k_w; //not by t sice we added one above 
      //To be clear this is the offset of K_i
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
  int k_w = 3;
  int k_h = 3;
  int t_w = width - k_w + 1;
  int t_h = height - k_h + 1;

  // Init Memory
  //float *lim, *depth, *softmaxWeights, *ds, *out, *a_c;
  float *depth, *softmaxWeights;

  printf("initalizing, \n");

  //t_h * t_w * height * width
  //lim = load_bin_files("data/T_S04923_lim.bin",&lim, sizeof(float), image_size);
  //ds = load_bin_files("data/T_S04923_ds.bin", &ds, sizeof(float), image_size);
  //a_c = load_bin_files("data/T_S04923_ds.bin", &a_c, sizeof(float), image_size);
  depth = load_bin_files("data/T_S04923_depth.bin",&depth, sizeof(float), width * height); //should be image_size/3 but i'll handle that later


  for (int i = 0; i < 9; i++)
    std::cout << "depth test: " << depth[i] << std::endl;

  printf("depth test done, \n");

  //cudaMallocManaged(&out, image_size * sizeof(float)); 
  cudaMallocManaged(&softmaxWeights,  t_w * t_h *  width * height * sizeof(float)); 
  cudaMemset(softmaxWeights, 0, t_w * t_h * width * height * sizeof(float));

  printf("loaded memory starting softmax test, \n");
  
  /* Convole the softmax depth weights for each color iteratively over time */
  for (int i = 0; i < 9; i++)
    std::cout << "test: " << softmaxWeights[i] << std::endl;

  printf("test done, \n");
  // Get the softmax weights as a kernel
  dim3 dimGrid(ceil(width/64), ceil(height/64));
  dim3 dimBlock(64, 64, 1); //Going based from textbook might be a better/more approiate size of block
  createSoftmaxWeightsKernel<<<dimGrid, dimBlock>>>(depth, softmaxWeights, width, height, k_h, k_w);
  cudaDeviceSynchronize(); //According to https://developer.nvidia.com/blog/even-easier-introduction-cuda/, prevents async weridness

  printf("convolution completed, \n");

  for (int i = 0; i < 9; i++)
    std::cout << "test: " << softmaxWeights[i] << std::endl;

  printf("sanity check, \n");  


  FILE* out_f_soft = fopen("data/T_S04923_softmaxes.bin", "wb");
  fwrite(softmaxWeights, sizeof(float), t_w * t_h * width * height, out_f_soft);
  fclose(out_f_soft);

  printf("written, \n");

  
  // //Gemm Convoltuon Implementation
  // cublasHandle_t handle; 
  // cublasCreate(&handle);

  // //https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication#:~:text=As%20you%20said%2C%20cuBLAS%20interprets,for%20the%20column%2Dmajor%20interpretation.
  // //According to here, we can just do the tranpose instead. I'm fine with that. 
  // // SO quick break down of the matrices
  // // Since we are doing B^TA^T in the background
  // // B =  a_c 
  // // A = softmax weights
  // // There are image_size number of things in a_c
  // float alpha = 1.f;
  // float beta = 0.f;

  // cublasSgemm(handle, 
  //   CUBLAS_OP_N, //Treat problem as B^T = a_c
  //   CUBLAS_OP_N, //Treat problem as A^T = softmax weights
  //   channels,  //There are 3 channels columns things in a_c, and the output (remember transpose)
  //   t_w * t_h, //rows of softmax weights (remember transpose) rows of c
  //   width*height, //columns shared by a_c and softmax_weights
  //   &alpha, //1
  //   a_c, // d_b
  //   channels, 
  //   softmaxWeights,
  //   width*height,
  //   &beta,
  //   out,
  //   channels
  // );

  // //write the output for the new lim to test out!
  // FILE* out_f = fopen("data/T_S04923_a_c.bin", "wb");
  // fwrite(out, sizeof(float), image_size, out_f);
  // fclose(out_f);

  // Free memory
  //cudaFree(lim);
  //cudaFree(ds);
  cudaFree(depth);
  //cudaFree(out);
  cudaFree(softmaxWeights);
  //cudaFree(a_c);
  
  printf("freed, \n");
  return 0;
}