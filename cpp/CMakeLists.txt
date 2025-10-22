cmake_minimum_required(VERSION 3.18)
project(CustomKernel CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "120")
find_package(CUDAToolkit REQUIRED)
add_executable(fwd fwd.cu)
target_include_directories(fwd PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
)