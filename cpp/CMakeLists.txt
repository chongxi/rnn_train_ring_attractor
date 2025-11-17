# This file is only for intellisense purpose, it's not used for compilation
cmake_minimum_required(VERSION 3.18)
project(CustomKernel CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "120")
find_package(CUDAToolkit REQUIRED)
add_executable(bwd bwd.cu
        kernels/bwd_simple.cuh
        kernels/bwd_tc_idx.cuh
        utils.cuh)
target_include_directories(bwd PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(bwd PRIVATE
        CUDA::cublas
)