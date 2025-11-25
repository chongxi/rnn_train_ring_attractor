# This file is only for compiling int main() function in bwd.cu for profiling purposes
cmake_minimum_required(VERSION 3.18)
project(CustomKernel CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "120")
find_package(CUDAToolkit REQUIRED)
add_executable(bwd bwd.cu
        kernels/bwd_simple.cuh
        kernels/bwd_mixed_precision_wmma.cuh
        utils.cuh
        kernels/bwd_full_precision.cuh
        kernels/bwd_mixed_precision_aggressive.cuh)
target_include_directories(bwd PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(bwd PRIVATE
        CUDA::cublas
#        CUDA::cublasLt
)
target_compile_options(bwd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)