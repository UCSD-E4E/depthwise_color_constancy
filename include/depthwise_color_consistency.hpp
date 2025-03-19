#pragma once

extern "C"
void depthwiseColorConsistency(unsigned int iterations, int image_width,
                                 int image_height, int image_num_channels,
                                 float alpha, const float *h_depth_map_ptr,
                                 const float *h_image_ptr, float* h_out);