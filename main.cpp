#include <iostream>

#include "depthwise_color_consistency.hpp"

float *load_bin_files(const char *path, size_t element_size,
                      size_t num_elements)
{
    FILE *file = fopen(path, "rb");
    if (!file)
    {
        perror("Failed to open file");
    }

    float *h_temp = (float *)malloc(num_elements * element_size);
    fread(h_temp, element_size, num_elements, file);
    fclose(file);

    return h_temp;
}

int main(void)
{
    // image parameters
    // number of pixels * number of channels

    int iterations = 1000;
    float alpha = 0.9999f;

    int width = 640; // 480,640
    int height = 480;
    int channels = 3; // Ideally RGB for now
    int num_pixels = width * height * channels;
    char *image_path = "data/realsense_tests/living_room_0046b_out_1-color.bin";
    char *depth_path = "data/realsense_tests/living_room_0046b_out_1-depth.bin";
    char *output_path = "data/realsense_tests/living_room_0046b_out_1-lim.bin";

    // Init Memory
    float *depth, *ds;
    ds = load_bin_files(image_path, sizeof(float), num_pixels);
    depth = load_bin_files(depth_path, sizeof(float), width * height);

    float *h_out = depthwiseColorConsistency(
        iterations,
        width,
        height,
        channels,
        alpha,
        depth,
        ds);

    FILE *out_f = fopen(output_path, "wb");
    //   fwrite(h_out, sizeof(float), num_pixels, out_f);
    fclose(out_f);

    //   free(h_out);
    free(ds);
    free(depth);

    return 0;
}