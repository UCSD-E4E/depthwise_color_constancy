use ndarray::Array2;
use ndarray::Array3;

mod depthwise_color_consistency;

#[cfg(cuda)]
pub fn depthwise_color_consistency(
    iterations: u32,
    image_width: i32,
    image_height: i32,
    image_num_channels: i32,
    alpha: f32,
    h_depth_map: Array2<f32>,
    h_image: Array3<f32>) -> Array3<f32> {
        let h_depth_map_ptr = h_depth_map.as_ptr();
        let h_image_ptr = h_image.as_ptr();
        let mut h_out = Array3::<f32>::zeros(h_image.dim());

        unsafe {
            depthwise_color_consistency::depthwiseColorConsistency(
                iterations,
                image_width,
                image_height,
                image_num_channels,
                alpha,
                h_depth_map_ptr,
                h_image_ptr,
                h_out.as_mut_ptr(),
            );
        }

        return h_out;
}
