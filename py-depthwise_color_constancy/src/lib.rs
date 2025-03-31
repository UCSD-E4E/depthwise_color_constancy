use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};

#[pyfunction(name="depthwise_color_constancy")]
fn depthwise_color_constancy_python<'py>(
    py: Python<'py>,
    iterations: u32,
    alpha: f32,
    h_depth_map: PyReadonlyArray2<'py, f32>,
    h_image: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let h_depth_map = h_depth_map.as_array().mapv(|x| x);
    let h_image = h_image.as_array().mapv(|x| x);
    let h_out = depthwise_color_constancy::depthwise_color_constancy(
        iterations,
        alpha,
        &h_depth_map,
        &h_image,
    );

    Ok(h_out.into_pyarray(py).to_owned())
}

#[pymodule(gil_used = false, name="depthwise_color_constancy")]
fn py_depthwise_color_constancy<'a>(_py: Python, m: Bound<'a, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(depthwise_color_constancy_python, &m)?)?;

    Ok(())
}