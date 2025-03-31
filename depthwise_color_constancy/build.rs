#[cfg(target_os = "linux")]
fn check_cuda() -> bool {
    use std::path::Path;

    return Path::new("/usr/local/cuda").exists();
}

#[cfg(target_os = "windows")]
fn check_cuda() -> bool {
    use winreg::enums::*;
    use winreg::RegKey;

    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    let cuda_key = hklm.open_subkey("SOFTWARE\\NVIDIA Corporation\\GPU Computing Toolkit\\CUDA");

    match cuda_key {
        Ok(_) => true,
        Err(error) =>
            if error.kind() == std::io::ErrorKind::NotFound {
                false
            }
            else {
                panic!("{}", error)
            }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn check_cuda() -> bool {
    return false;
}

fn build_cuda() {
    // Check if CUDA is installed
    if !check_cuda() {
        return; // CUDA is not installed
    }

    // Configure CUDA build
    if cfg!(target_os = "linux") {
        println!("cargo::rustc-link-search=native=/usr/local/cuda/lib64");
    }
    //todo windows linking for cublas
    println!("cargo::rustc-link-lib=cublas");
    println!("cargo::rustc-cfg=cuda");
    println!("cargo::rerun-if-changed=src/depthwise_color_constancy.cu");
    println!("cargo::rerun-if-changed=include/depthwise_color_constancy.hpp");

    // Compile CUDA code
    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .file("src/depthwise_color_constancy.cu")
        .compile("depthwise_color_constancy");

    bindgen::Builder::default()
        .header("include/depthwise_color_constancy.hpp")
        .clang_arg("-I/usr/local/cuda/include")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("src/depthwise_color_constancy.rs")
        .expect("Unable to write bindings");
}

fn main() {
    build_cuda();
}