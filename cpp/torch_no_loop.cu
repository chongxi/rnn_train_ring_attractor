#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>

#define WARP_SIZE 32
#define BLOCK_SIZE 288
#define CLUSTER_SIZE 8

namespace cg = cooperative_groups;

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

void torch_sum_cuda(
    void* A,
    void* Wa,
    void* J0,
    float J1,
    void* Wo,

    void* r, // init value for r

    void* W_delta7,
    void* W_eff,
    void* bump_history,
    void* r_delta7,
    void* r_history,
    void* re_inp,
    int N,
    int a_dim,
    int seq_len,
    int batch_size
);

__global__ void persistent_splitK_kernel(
    const float* A, // shape [batch_size, 128, 32]
    const float* Wa, // shape [32, 256, 256]
    const float* J0, // shape [256, 256]
    float J1,
    const float* Wo,
    float* r,
    const float* W_delta7,
    const float* W_eff,
    float* bump_history,
    float* r_history,
    int N,
    int a_dim,
    int seq_len,
    int batch_size
){  
    __shared__ float r_buf[BLOCK_SIZE];
    __shared__ float re_inp_buf[BLOCK_SIZE];
    __shared__ float r_d7_buf[BLOCK_SIZE];

    int batch_idx = blockIdx.x;
    
    r_buf[threadIdx.x] = r[batch_idx * N + threadIdx.x];

    constexpr float kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5f;

    int row_idx = threadIdx.x;
    
    for (int t = 0; t < seq_len; ++t){

        re_inp_buf[threadIdx.x] = 0.f;
        r_d7_buf[threadIdx.x] = 0.f;

        __syncthreads();

        for (int k_idx = 0; k_idx < N; ++k_idx){
            float wa_weighted = 0.f;
            for (int action = 0; action < a_dim; ++action){
                wa_weighted += A[batch_idx * seq_len * a_dim + t * a_dim + action] * Wa[action * N * N + row_idx * N + k_idx];
            }
            
            float w_eff = J0[row_idx * N + k_idx] + J1 * Wo[row_idx * N + k_idx] + wa_weighted;
            atomicAdd(&re_inp_buf[threadIdx.x], w_eff * r_buf[k_idx]);
            // re_inp_buf[threadIdx.x] += w_eff * r_buf[k_idx];
        }

        __syncthreads();

        float re_inp_act = re_inp_buf[threadIdx.x];
        float re_inp_val = 0.5f * re_inp_act * (1.f + tanhf(kBetaVec * fmaf(0.044715f, re_inp_act * re_inp_act * re_inp_act, re_inp_act)));
        
        float alpha = 0.15f;
        float r_updated = (1.f - alpha) * r_buf[threadIdx.x] + alpha * re_inp_val;

        bump_history[batch_idx * seq_len * N + t * N + threadIdx.x] = r_updated;
        r_buf[threadIdx.x] = r_updated;

        __syncthreads();

        for (int k_idx = 0; k_idx < N; ++k_idx){
            atomicAdd(&r_d7_buf[threadIdx.x], r_buf[k_idx] * W_delta7[k_idx * N + threadIdx.x]);
        }

        __syncthreads();

        float r_d7_lane = r_d7_buf[threadIdx.x];

        // Fixed reduction for finding maximum
        for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                r_d7_buf[threadIdx.x] = fmaxf(r_d7_buf[threadIdx.x], r_d7_buf[threadIdx.x + offset]);
            }
            __syncthreads();
        }
        
        r_history[batch_idx * seq_len * N + t * N + threadIdx.x] = r_d7_lane / r_d7_buf[0];

    } // end for t
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) persistent_splitK_tbc_kernel(
    const float* A, // shape [batch_size, 128, 32]
    const float* Wa, // shape [32, 256, 256]
    const float* J0, // shape [256, 256]
    float J1,
    const float* Wo,
    float* r,
    const float* W_delta7,
    const float* W_eff,
    float* bump_history,
    float* r_history,
    int N,
    int a_dim,
    int seq_len,
    int batch_size
){  
    __shared__ float r_buf[256];
    __shared__ float re_inp_buf[256];
    __shared__ float r_d7_buf[256];

    auto this_cluster = cg::this_cluster();

    int cluster_idx = this_cluster.block_rank();

    

    int batch_idx = blockIdx.x;
    
    r_buf[threadIdx.x] = r[batch_idx * N + threadIdx.x];

    constexpr float kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5f;

    int row_idx = threadIdx.x;
    
    for (int t = 0; t < seq_len; ++t){

        re_inp_buf[threadIdx.x] = 0.f;
        r_d7_buf[threadIdx.x] = 0.f;

        __syncthreads();

        for (int k_idx = 0; k_idx < N; ++k_idx){
            float wa_weighted = 0.f;
            for (int action = 0; action < a_dim; ++action){
                wa_weighted += A[batch_idx * seq_len * a_dim + t * a_dim + action] * Wa[action * N * N + row_idx * N + k_idx];
            }
            
            float w_eff = J0[row_idx * N + k_idx] + J1 * Wo[row_idx * N + k_idx] + wa_weighted;
            atomicAdd(&re_inp_buf[threadIdx.x], w_eff * r_buf[k_idx]);
        }

        __syncthreads();

        float re_inp_act = re_inp_buf[threadIdx.x];
        float re_inp_val = 0.5f * re_inp_act * (1.f + tanhf(kBetaVec * fmaf(0.044715f, re_inp_act * re_inp_act * re_inp_act, re_inp_act)));
        
        float alpha = 0.15f;
        float r_updated = (1.f - alpha) * r_buf[threadIdx.x] + alpha * re_inp_val;

        bump_history[batch_idx * seq_len * N + t * N + threadIdx.x] = r_updated;
        r_buf[threadIdx.x] = r_updated;

        __syncthreads();

        for (int k_idx = 0; k_idx < N; ++k_idx){
            atomicAdd(&r_d7_buf[threadIdx.x], r_buf[k_idx] * W_delta7[k_idx * N + threadIdx.x]);
        }

        __syncthreads();

        float r_d7_lane = r_d7_buf[threadIdx.x];

        // Fixed reduction for finding maximum
        for (int offset = 128; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                r_d7_buf[threadIdx.x] = fmaxf(r_d7_buf[threadIdx.x], r_d7_buf[threadIdx.x + offset]);
            }
            __syncthreads();
        }
        
        r_history[batch_idx * seq_len * N + t * N + threadIdx.x] = r_d7_lane / r_d7_buf[0];

    } // end for t
}

void torch_sum_cuda(
    void* A,
    void* Wa,
    void* J0,
    float J1,
    void* Wo,
    void* r, // init value for r
    void* W_delta7,
    void* W_eff,
    void* bump_history,
    void* r_delta7,
    void* r_history,
    void* re_inp,
    int N,
    int a_dim,
    int seq_len,
    int batch_size
){
    dim3 blockSize, gridSize;

    blockSize = BLOCK_SIZE; gridSize = batch_size;

    persistent_splitK_kernel<<<gridSize, blockSize>>>(
        static_cast<const float*>(A),
        static_cast<const float*>(Wa),
        static_cast<const float*>(J0),
        J1,
        static_cast<const float*>(Wo),
        static_cast<float*>(r),
        static_cast<float *>(W_delta7),
        static_cast<float *>(W_eff),
        static_cast<float *>(bump_history),
        // static_cast<float *>(r_delta7),
        static_cast<float *>(r_history),
        // static_cast<float *>(re_inp),
        N,
        a_dim,
        seq_len,
        batch_size        
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // blockSize = BLOCK_SIZE; gridSize = batch_size * CLUSTER_SIZE;

    // persistent_splitK_tbc_kernel<<<gridSize, blockSize>>>(
    //     static_cast<const float*>(A),
    //     static_cast<const float*>(Wa),
    //     static_cast<const float*>(J0),
    //     J1,
    //     static_cast<const float*>(Wo),
    //     static_cast<float*>(r),
    //     static_cast<float *>(W_delta7),
    //     static_cast<float *>(W_eff),
    //     static_cast<float *>(bump_history),
    //     // static_cast<float *>(r_delta7),
    //     static_cast<float *>(r_history),
    //     // static_cast<float *>(re_inp),
    //     N,
    //     a_dim,
    //     seq_len,
    //     batch_size        
    // );
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());    
}