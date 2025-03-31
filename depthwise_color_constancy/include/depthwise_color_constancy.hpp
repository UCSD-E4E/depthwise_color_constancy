#pragma once

extern "C" void depthwiseColorConstancy(unsigned int iterations, unsigned int image_width,
                                          unsigned int image_height, unsigned int image_num_channels,
                                          float alpha, const float *h_depth_map_ptr,
                                          const float *h_image_ptr, float *h_out);