#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>

#define IN_TILE_WIDTH 32
//#define KERNEL_SIZE 1
//#define OUT_TILE_WIDTH (IN_TILE_WIDTH - 2 * KERNEL_SIZE) //TODO REPLACE WITH PARAMETRIZATION

__global__ void softmaxDepthAdverging(
    float *ds,
    float *output,
    float *depth,
    float *kernel_shape,
    int width,
    int height,
    int channels,
    int kernel_size)
{
    int actual_kernel_size = kernel_size * 2 + 1;
    int OUT_TILE_WIDTH = IN_TILE_WIDTH - 2 * kernel_size;
    // Prep for tiling both depth and ds to preprocess some data
    //  Tiled Convolution implmenetation based on
    //  Programming Massively Parallel Processors
    __shared__ float ds_tile[IN_TILE_WIDTH][IN_TILE_WIDTH][3];
    __shared__ float depth_tile[IN_TILE_WIDTH][IN_TILE_WIDTH];

    int col = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x - kernel_size;
    int row = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y - kernel_size;
    int pixel_idx = (row * width + col) * channels;
    if (row >= 0 && row <= height && col >= 0 && col <= width)
    {
        depth_tile[threadIdx.y][threadIdx.x] = depth[row * width + col];
        ds_tile[threadIdx.y][threadIdx.x][0] = ds[pixel_idx];
        ds_tile[threadIdx.y][threadIdx.x][1] = ds[pixel_idx + 1];
        ds_tile[threadIdx.y][threadIdx.x][2] = ds[pixel_idx + 2];
    }
    else
    {
        depth_tile[threadIdx.y][threadIdx.x] = 0; // incase of access, this goes to 0
        ds_tile[threadIdx.y][threadIdx.x][0] = 1;
        ds_tile[threadIdx.y][threadIdx.x][1] = 0;
        ds_tile[threadIdx.y][threadIdx.x][2] = 0;
    }

    __syncthreads();

    int x = threadIdx.x;
    int y = threadIdx.y;
    float eps = 10;
    float exp_sum = 0;

    if (row >= 0 && row <= height && col >= 0 && col <= width)
    {
        if (x >= kernel_size && x <= OUT_TILE_WIDTH + kernel_size && y >= kernel_size && y <= OUT_TILE_WIDTH + kernel_size)
        {
            //  compute denom for softmax
            for (int i = -kernel_size; i <= kernel_size; i++)
            { // TODO generalize this
                for (int j = -kernel_size; j <= kernel_size; j++)
                { // like with a footprint sys
                    int nx = x + j;
                    int ny = y + i;
                    if (ny >= 0 && ny <= height && nx >= 0 && nx <= width)
                    {
                        if (nx >= 0 && nx < IN_TILE_WIDTH && ny >= 0 && ny < IN_TILE_WIDTH)
                        {
                            int kernel_i = i + kernel_size;
                            int kernel_j = j + kernel_size;
                            exp_sum += kernel_shape[kernel_i * actual_kernel_size + kernel_j] * expf(-abs(depth_tile[ny][nx] - depth_tile[y][x]) - eps);
                        }
                    }
                }
            }
            // so now $\text{texp_sum} = \sum_{n \in N} e^{-|n_d - t_d|}}
            // softmax of a neighbor n is $\frac{e^{-|n_d - t_d|}}{exp_sum}$

            // Apply avging to each color channels
            for (int c = 0; c < channels; c++)
            {
                float avg_color = 0;

                // convolution here
                for (int i = -kernel_size; i <= kernel_size; i++)
                {
                    for (int j = -kernel_size; j <= kernel_size; j++)
                    {
                        int nx = x + j;
                        int ny = y + i;

                        // $\sum_{n \in N} softmax(n) * color(n)
                        if (ny >= 0 && ny <= height && nx >= 0 && nx <= width)
                        {
                            if (nx >= 0 && nx < IN_TILE_WIDTH && ny >= 0 && ny < IN_TILE_WIDTH)
                            {
                                int kernel_i = i + kernel_size;
                                int kernel_j = j + kernel_size;
                                // weight of the softmax by the color
                                float softmax_weight = expf(-abs(depth_tile[ny][nx] - depth_tile[y][x]) - eps) / exp_sum;
                                avg_color +=  kernel_shape[kernel_i * actual_kernel_size + kernel_j] * softmax_weight * ds_tile[ny][nx][c];
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


__global__ void NaiveAdverging(
    float *ds,
    float *output,
    float *depth,
    float *kernel_shape,
    int width,
    int height,
    int channels,
    int kernel_size,
    float thres)
{
    int actual_kernel_size = kernel_size * 2 + 1;
    int OUT_TILE_WIDTH = IN_TILE_WIDTH - 2 * kernel_size;
    // Prep for tiling both depth and ds to preprocess some data
    //  Tiled Convolution implmenetation based on
    //  Programming Massively Parallel Processors
    __shared__ float ds_tile[IN_TILE_WIDTH][IN_TILE_WIDTH][3];
    __shared__ float depth_tile[IN_TILE_WIDTH][IN_TILE_WIDTH];

    int col = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x - kernel_size;
    int row = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y - kernel_size;
    int pixel_idx = (row * width + col) * channels;
    if (row >= 0 && row < height && col >= 0 && col < width)
    {
        depth_tile[threadIdx.y][threadIdx.x] = depth[row * width + col];
        ds_tile[threadIdx.y][threadIdx.x][0] = ds[pixel_idx];
        ds_tile[threadIdx.y][threadIdx.x][1] = ds[pixel_idx + 1];
        ds_tile[threadIdx.y][threadIdx.x][2] = ds[pixel_idx + 2];
    }
    else
    {
        depth_tile[threadIdx.y][threadIdx.x] = 0; // incase of access, this goes to 0
        ds_tile[threadIdx.y][threadIdx.x][0] = 1;
        ds_tile[threadIdx.y][threadIdx.x][1] = 0;
        ds_tile[threadIdx.y][threadIdx.x][2] = 0;
    }

    __syncthreads();

    int x = threadIdx.x;
    int y = threadIdx.y;
    float eps = 10;
    float sum = 0;

    if (row >= 0 && row < height && col >= 0 && col < width)
    {
        if (x >= kernel_size && x <= OUT_TILE_WIDTH + kernel_size && y >= kernel_size && y <= OUT_TILE_WIDTH + kernel_size)
        {
            //  compute denom for softmax
            for (int i = -kernel_size; i <= kernel_size; i++)
            { // TODO generalize this
                for (int j = -kernel_size; j <= kernel_size; j++)
                { // like with a footprint sys
                    int nx = x + j;
                    int ny = y + i;
                    if (ny >= 0 && ny <= height && nx >= 0 && nx <= width)
                    {
                        if (nx >= 0 && nx < IN_TILE_WIDTH && ny >= 0 && ny < IN_TILE_WIDTH)
                        {
                            int kernel_i = i + kernel_size;
                            int kernel_j = j + kernel_size;
                            sum += kernel_shape[kernel_i * actual_kernel_size + kernel_j] * float(abs(depth_tile[ny][nx] - depth_tile[y][x]) < thres);
                        }
                    }
                }
            }
            // so now $\text{texp_sum} = \sum_{n \in N} e^{-|n_d - t_d|}}
            // softmax of a neighbor n is $\frac{e^{-|n_d - t_d|}}{exp_sum}$

            // Apply avging to each color channels
            for (int c = 0; c < channels; c++)
            {
                float avg_color = 0;

                // convolution here
                for (int i = -kernel_size; i <= kernel_size; i++)
                {
                    for (int j = -kernel_size; j <= kernel_size; j++)
                    {
                        int nx = x + j;
                        int ny = y + i;

                        // $\sum_{n \in N} softmax(n) * color(n)
                        if (ny >= 0 && ny <= height && nx >= 0 && nx <= width)
                        {
                            if (nx >= 0 && nx < IN_TILE_WIDTH && ny >= 0 && ny < IN_TILE_WIDTH)
                            {
                                int kernel_i = i + kernel_size;
                                int kernel_j = j + kernel_size;
                                // weight of the softmax by the color
                                float weight = float(abs(depth_tile[ny][nx] - depth_tile[y][x]) < thres) / sum;
                                avg_color +=  kernel_shape[kernel_i * actual_kernel_size + kernel_j] * weight * ds_tile[ny][nx][c];
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

extern "C" void depthwiseColorConstancy(unsigned int iterations, unsigned int image_width,
                                          unsigned int image_height, unsigned int image_num_channels,
                                          unsigned int kernal_size, float alpha, const float *h_depth_map_ptr,
                                          const float *h_image_ptr, const float *h_kernal_shape, float *h_out, 
                                          bool doesNaive = false, const float threshold=0.1)
{ 
    //TODO: add to parameters
    //float kernal_size = 3;

    float *d_depth_map_ptr, *d_in_image_ptr, *d_temp_ptr, *d_a_c_ptr, *kernal_shape_ptr;
    float beta = 1.f - alpha;
    int num_pixels = image_width * image_height * image_num_channels;
    int depth_size = image_width * image_height;

    // depth map of image
    cudaMalloc(&d_depth_map_ptr, depth_size * sizeof(float));
    cudaMemcpy(d_depth_map_ptr, h_depth_map_ptr, depth_size * sizeof(float),
               cudaMemcpyHostToDevice);

    // original image, needs to be kept to keep a_c stable each iteration
    cudaMalloc(&d_in_image_ptr, num_pixels * sizeof(float));
    cudaMemcpy(d_in_image_ptr, h_image_ptr, num_pixels * sizeof(float),
               cudaMemcpyHostToDevice);

    // a_c is the avged color image each iteration, starts with original image
    cudaMalloc(&d_a_c_ptr, num_pixels * sizeof(float));
    cudaMemcpy(d_a_c_ptr, h_image_ptr, num_pixels * sizeof(float),
               cudaMemcpyHostToDevice);

    // temp image for holding raw illumiant map
    cudaMalloc(&d_temp_ptr, num_pixels * sizeof(float));

    // describes which pixels to select for 
    //float h_kernal_shape[9] = {1,1,1,1,1,1,1,1,1};
    //TODO ADD ABOVE TO PARAMETERS
    cudaMalloc(&kernal_shape_ptr, kernal_size * kernal_size * sizeof(float));
    cudaMemcpy(kernal_shape_ptr, h_kernal_shape, kernal_size * kernal_size * sizeof(float),
               cudaMemcpyHostToDevice);

    // +10 is a workaround for this missing a column
    // TODO fix this workaround
    dim3 dimGrid(
        ceil(float(image_width) / IN_TILE_WIDTH) + 1,
        ceil(float(image_height) / IN_TILE_WIDTH) + 1
    );
    dim3 dimBlock(IN_TILE_WIDTH, IN_TILE_WIDTH);

    // cublas handlers
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Conduct Depthwise Operation
    for (unsigned int i = 0; i < iterations; i++)
    {
        if (doesNaive) 
            NaiveAdverging<<<dimGrid, dimBlock>>>(
                d_a_c_ptr, d_temp_ptr, d_depth_map_ptr, kernal_shape_ptr, image_width, image_height,
                image_num_channels, int(kernal_size/2), threshold);
        else
            softmaxDepthAdverging<<<dimGrid, dimBlock>>>(
                d_a_c_ptr, d_temp_ptr, d_depth_map_ptr, kernal_shape_ptr, image_width, image_height,
                image_num_channels, int(kernal_size/2));

        // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
        // According to here, we can just do the tranpose instead. I'm fine with
        // that.
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, image_num_channels,
                    image_height * image_width, &alpha, d_temp_ptr,
                    image_num_channels, &beta, d_in_image_ptr, image_num_channels,
                    d_a_c_ptr, image_num_channels);
    }

    cudaDeviceSynchronize();

    // write the output for the new lim to test out!
    // We expect the output to be the same size as the input.
    cudaMemcpy(h_out, d_a_c_ptr, num_pixels * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Free memory
    cublasDestroy(handle);
    cudaFree(d_in_image_ptr);
    cudaFree(d_depth_map_ptr);
    cudaFree(d_temp_ptr);
    cudaFree(d_a_c_ptr);
}
