#pragma once

extern "C"
float *depthwiseColorConsistency(size_t iterations, int image_width,
                                 int image_height, int image_num_channels,
                                 float alpha, float *h_depth_map_ptr,
                                 float *h_image_ptr);