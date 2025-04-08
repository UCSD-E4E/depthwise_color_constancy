use ndarray::Array2;
use ndarray::Array3;

#[cfg(cuda)]
mod depthwise_color_constancy;

#[cfg(cuda)]
pub fn depthwise_color_constancy(
    iterations: u32,
    alpha: f32,
    doesNaive: bool,
    threshold: f32,
    h_depth_map: &Array2<f32>,
    h_kernal: &Array2<f32>,
    h_image: &Array3<f32>) -> Array3<f32> {
        let h_depth_map_ptr = h_depth_map.as_ptr();
        let h_image_ptr = h_image.as_ptr();
        let h_kernal_ptr = h_kernal.as_ptr();

        let (image_height, image_width, image_num_channels) = h_image.dim();
        let (kernal_size, _) = h_kernal.dim();
        let mut h_out = Array3::<f32>::zeros(h_image.dim());

        unsafe {
            depthwise_color_constancy::depthwiseColorConstancy(
                iterations,
                image_width as u32,
                image_height as u32,
                image_num_channels as u32,
                kernal_size as u32,
                alpha,
                h_depth_map_ptr,
                h_image_ptr,
                h_kernal_ptr,
                h_out.as_mut_ptr(),
                doesNaive,
                threshold,
            );
        }

        return h_out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depthwise_color_constancy_test() {
        let h_img = Array3::<f32>::ones((480, 600, 3));
        let h_depth = Array2::<f32>::zeros((480, 600));

        let h_out = depthwise_color_constancy(100, 0f32, &h_depth, &h_img, false, 0.0);
        
        assert_eq!(h_img, h_out);
    }
}