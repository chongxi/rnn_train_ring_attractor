#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>


// #define _USE_MATH_DEFINES
// #include "math.h"

// #include <vector>

#define WARPSIZE 32

namespace cg = cooperative_groups;
using namespace nvcuda;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename To, typename From>
__device__ __forceinline__ To cuda_cast(From x);

// float → half
template <>
__device__ __forceinline__ __half cuda_cast<__half, float>(float x) {
    return __float2half(x);
}

// float → bfloat16
template <>
__device__ __forceinline__ __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float x) {
    return __float2bfloat16(x);
}

enum class Activation { RELU, GELU, TANH, SILU };

template<Activation ACT>
__device__ __forceinline__ float activation(float x) {
    if constexpr (ACT == Activation::RELU) {
        return fmaxf(x, 0.f);
    } else if constexpr (ACT == Activation::GELU) {
        return x * normcdff(x);
    } else if constexpr (ACT == Activation::SILU) {
        float exp_arg = fminf(fmaxf(-x, -88.f), 88.f);
        return x / (1.f + __expf(exp_arg));
    } else { // TANH
        return tanhf(x);
    }
}

template<Activation ACT>
__device__ __forceinline__ float activation_derivative(float x) {
    if constexpr (ACT == Activation::RELU) {
        return x > 0.f ? 1.f : 0.f;
    } else if constexpr (ACT == Activation::GELU) {
        // d/dx[x * Φ(x)] = Φ(x) + x * φ(x)
        float exp_arg = fminf(fmaxf(-0.5f * x * x, -88.f), 88.f);
        float phi = __expf(exp_arg) * 0.56418958354775628694807945156077f;
        return normcdff(x) + x * phi;
    } else if constexpr (ACT == Activation::SILU) {
        float exp_arg = fminf(fmaxf(-x, -88.f), 88.f);
        float sigmoid_x = 1.f / (1.f + __expf(exp_arg));
        return sigmoid_x * (1.f + x * (1.f - sigmoid_x));
    } else { // TANH
        float tanh_x = tanhf(x);
        return 1.f - tanh_x * tanh_x;
    }
}



template <typename T, uint32_t num_dim>
struct IndexWrapper
{
    template <typename... Dims>
    constexpr __host__ __device__ explicit IndexWrapper(T* ptr, Dims... dims) : dimensions{static_cast<uint32_t>(dims)...}, m_ptr(ptr)
    {
        static_assert(sizeof...(dims) == num_dim);
    }

    __host__ __device__ T* ptr() { return m_ptr; }
    __host__ __device__ const T* ptr() const { return m_ptr; }

    template <typename... Idx>
    __host__ __device__ T& at(Idx... idx)
    {
        static_assert(sizeof...(Idx) == num_dim);
        return m_ptr[_calc_1D_idx<0, Idx...>(idx...)];
    }

    template <typename... Idx>
    __host__ __device__ const T& at(Idx... idx) const
    {
        static_assert(sizeof...(Idx) == num_dim);
        return m_ptr[_calc_1D_idx<0, Idx...>(idx...)];
    }

    template <uint32_t dim_idx>
    constexpr __host__ __device__ uint32_t stride_size() const {
        if constexpr (dim_idx < num_dim) {
            return dimensions[dim_idx] * stride_size<dim_idx + 1>();
        } else {
            return 1;
        }
    }

    template <uint32_t dim_idx, typename Idx>
    constexpr static __host__ __device__ uint32_t _calc_1D_idx(Idx idx)
    {
        static_assert(dim_idx == num_dim - 1);
        return idx;
    }

    template <uint32_t dim_idx, typename Idx, typename... Tail>
    constexpr __host__ __device__ uint32_t _calc_1D_idx(Idx idx, Tail... tail) const
    {
        return idx * stride_size<dim_idx+1>() + _calc_1D_idx<dim_idx + 1, Tail...>(tail...);
    }

    uint32_t dimensions[num_dim];
    T* m_ptr;
};

