use std::path::Path;

fn build_cuda() {
    // Check if CUDA is installed
    if !Path::new("/usr/local/cuda").exists() {
        return; // CUDA is not installed
    }

    // Configure CUDA build
    println!("cargo::rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo::rustc-link-lib=cublas");
    println!("cargo::rerun-if-changed=src/depthwise_color_consistency.cu");

    // Compile CUDA code
    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .file("src/depthwise_color_consistency.cu")
        .compile("depthwise_color_consistency");
}

fn main() {
    build_cuda();
}