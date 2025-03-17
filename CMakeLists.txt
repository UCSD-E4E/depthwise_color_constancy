cmake_minimum_required(VERSION 3.18)
project(hardware_accelerated_depthwise_color_consistency LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add CUDA source file
add_library(depthwise_color_consistency src/depthwise_color_consistency.cu)

# Include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add executable
add_executable(dcc main.cpp)

# Link CUDA library to the executable
target_link_libraries(dcc depthwise_color_consistency -lcublas)