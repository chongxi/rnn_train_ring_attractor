#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// #define _USE_MATH_DEFINES
// #include "math.h"

#include <iostream>
// #include <vector>

#define WARPSIZE 32

namespace cg = cooperative_groups;
using namespace nvcuda;

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
        // return 0.5f * x * (1.f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
        // return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));
        // grad: float pdf = __expf(-0.5f * x * x) * M_1_SQRT2PI; return normcdf(x) + x * pdf;
        return x * normcdff(x);
    } else if constexpr (ACT == Activation::SILU) {
        return x / (1.f + __expf(-x));
    } else { // TANH
        return tanhf(x);
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