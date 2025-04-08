#pragma once

extern "C" void depthwiseColorConstancy(unsigned int iterations, unsigned int image_width,
                                          unsigned int image_height, unsigned int image_num_channels,
                                          unsigned int kernal_size, float alpha,
                                          const float *h_depth_map_ptr, const float *h_image_ptr,
                                          const float *h_kernal_shape, float *h_out,
                                          bool doesNaive = false, const float threshold=0.1);