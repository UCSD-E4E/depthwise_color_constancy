use std::path::Path;

fn build_cuda() {
    // Check if CUDA is installed
    if !Path::new("/usr/local/cuda").exists() {
        return; // CUDA is not installed
    }

    // Configure CUDA build
    println!("cargo::rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo::rustc-link-lib=cublas");
    println!("cargo::rustc-cfg=cuda");
    println!("cargo::rerun-if-changed=src/depthwise_color_consistency.cu");
    println!("cargo::rerun-if-changed=include/depthwise_color_consistency.hpp");

    // Compile CUDA code
    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .file("src/depthwise_color_consistency.cu")
        .compile("depthwise_color_consistency");

    bindgen::Builder::default()
        .header("include/depthwise_color_consistency.hpp")
        .clang_arg("-I/usr/local/cuda/include")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("src/depthwise_color_consistency.rs")
        .expect("Unable to write bindings");
}

fn main() {
    build_cuda();
}