#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
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


__device__ float smem_reduce(float val_per_thread) {
    // Return result stored in smem and copied to every threads' register, so all thread have access to them
    
    __shared__ float s_data[256];
    
    s_data[threadIdx.x] = val_per_thread;
    __syncthreads();
    
    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    return s_data[0]; //broadcast, not bank conflict
}

// FIX 4: In CUDA kernel - use Kahan summation for stable reduction
__device__ float stable_reduce(float val_per_thread) {
    __shared__ float s_data[256];
    __shared__ float s_comp[256];  // Compensation for lost precision
    
    s_data[threadIdx.x] = val_per_thread;
    s_comp[threadIdx.x] = 0.0f;
    __syncthreads();
    
    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            float y = s_data[threadIdx.x + stride] - s_comp[threadIdx.x];
            float t = s_data[threadIdx.x] + y;
            s_comp[threadIdx.x] = (t - s_data[threadIdx.x]) - y;
            s_data[threadIdx.x] = t;
        }
        __syncthreads();
    }
    
    return s_data[0];
}

__device__ float deterministic_reduce(float val_per_thread) {
    __shared__ float s_data[256];
    s_data[threadIdx.x] = val_per_thread;
    __syncthreads();
    
    // CRITICAL: Only thread 0 does reduction in fixed order
    if (threadIdx.x == 0) {
        double sum = 0.0;  // Use double precision for accumulation
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += (double)s_data[i];  // Fixed order: 0,1,2,...,255
        }
        s_data[0] = (float)sum;
    }
    __syncthreads();
    
    return s_data[0];
}

__device__ float warp_reduce(float val_per_thread) {
    // Return result stored in first thread's register, only first thread has access to correct reduction sum, other threads have result = 0.f their registers

    auto cta = cg::this_thread_block();
    int warp_idx = cta.thread_rank() / WARP_SIZE;
    int warp_lane_idx = cta.thread_rank() % WARP_SIZE;
    
    float warp_sum = val_per_thread;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }
    
    __shared__ float temp_reduction[32]; // only need float temp_reduction[8], but set 32 to remove if statement
    
    // First lane in each warp writes to shared memory
    if (warp_lane_idx == 0) {
        temp_reduction[warp_idx] = warp_sum;
    }
    
    __syncthreads();
    
    float reduction_result = 0.f;

    // First warp performs final reduction
    if (warp_idx == 0) {
        reduction_result = temp_reduction[warp_lane_idx];
        
        // Reduce 8 warp results
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            reduction_result += __shfl_down_sync(0xffffffff, reduction_result, offset);
        }
    }

    return reduction_result;
}

__device__ float smem_reduce_max(float val_per_thread) {
    __shared__ float s_data[256];
    
    s_data[threadIdx.x] = val_per_thread;
    __syncthreads();
    
    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_data[threadIdx.x] = fmaxf(s_data[threadIdx.x], s_data[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    return s_data[0];
}

__device__ float warp_reduce_max(float val_per_thread) {
    auto cta = cg::this_thread_block();
    int warp_idx = cta.thread_rank() / WARP_SIZE;
    int warp_lane_idx = cta.thread_rank() % WARP_SIZE;
    
    float warp_max = val_per_thread;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
    }
    
    __shared__ float temp_reduction[32];
    
    if (warp_lane_idx == 0) {
        temp_reduction[warp_idx] = warp_max;
    }
    
    __syncthreads();
    
    float reduction_result = -INFINITY;

    if (warp_idx == 0) {
        reduction_result = temp_reduction[warp_lane_idx];
        
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            reduction_result = fmaxf(reduction_result, __shfl_down_sync(0xffffffff, reduction_result, offset));
        }
    }

    return reduction_result;
}

__global__ void persistent_reduce_kernel(
    const float* A, // shape [1, 128, 32]
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
    // Todo, use smem, use num threads = 1024 -> tile = 256x32, use vectorized load
    // Or use tile 128*64 fp32 or 256*64 bf16, try to saturate the huge bandwidth of sm_120
    // Load all of A_t into smem, and pragma unroll to get lds.128
    // Can implement smem for storing w_eff in vectorized mode

    __shared__ float re_inp_buf[256];
    __shared__ float r_d7_buf[256];
    __shared__ float inv_r_d7_max;
    // __shared__ float 

    constexpr float kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5f;


    // int row_idx = blockIdx.x;
    int col_idx = threadIdx.x;
    
    for (int t = 0; t < seq_len; ++t){

        for (int row_idx = 0; row_idx < N; ++row_idx){

            float wa_weighted = 0.f;
            for (int action = 0; action < a_dim; ++action){
                // TODO: change/transpose/permute Wa for coalesced memory access ?
                // Maybe already good
                wa_weighted += A[t * a_dim + action] * Wa[action * N * N + row_idx * N + col_idx];
            } // end for action

            float w_eff = J0[row_idx * N + col_idx] + J1 * Wo[row_idx * N + col_idx] + wa_weighted;

            float re_inp_lane = w_eff * r[threadIdx.x];

            // syncthreads() inside warp_reduce, be aware of dead lock
            float re_inp_wr =  warp_reduce(re_inp_lane);

            if (threadIdx.x == 0){
                float re_inp_val = 0.5f * re_inp_wr * (1.f + tanhf(kBetaVec * fmaf(0.044715f, re_inp_wr * re_inp_wr * re_inp_wr, re_inp_wr)));
                re_inp_buf[row_idx] = re_inp_val;
            }
        } // end for row_idx

        __syncthreads();

        // Update r
        float alpha = 0.15;
        float r_updated = (1 - alpha) * r[threadIdx.x] + alpha * re_inp_buf[threadIdx.x];

        bump_history[t * N + threadIdx.x] = r_updated;

        for (int col_idx = 0; col_idx < N; ++col_idx){
            float r_d7_lane = r_updated * W_delta7[threadIdx.x * N + col_idx];
            float r_d7_wr = warp_reduce(r_d7_lane);

            if (threadIdx.x == 0){
                r_d7_buf[col_idx] = r_d7_wr;
            }
        } // end for col_idx

        __syncthreads();

        float r_d7_lane = r_d7_buf[threadIdx.x];
        float r_d7_wrm = warp_reduce_max(r_d7_lane);

        if (threadIdx.x == 0){
            inv_r_d7_max = 1.f / r_d7_wrm;
        }

        __syncthreads();

        r_history[t * N + threadIdx.x] = inv_r_d7_max * r_d7_buf[threadIdx.x];

        // re_inp[threadIdx.x] = r_buf[threadIdx.x];

    } // end for t

}


__global__ void persistent_splitK_kernel(
    const float* A, // shape [1, 128, 32]
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
    // Todo, use smem, use num threads = 1024 -> tile = 256x32, use vectorized load
    // Or use tile 128*64 fp32 or 256*64 bf16, try to saturate the huge bandwidth of sm_120
    // Load all of A_t into smem, and pragma unroll to get lds.128
    // Can implement smem for storing w_eff in vectorized mode

    __shared__ float r_buf[256];
    __shared__ float re_inp_buf[256];
    __shared__ float r_d7_buf[256];

    r_buf[threadIdx.x] = r[threadIdx.x];

    constexpr float kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5f;

    int row_idx = threadIdx.x;
    
    for (int t = 0; t < seq_len; ++t){

        re_inp_buf[threadIdx.x] = 0.f;
        r_d7_buf[threadIdx.x] = 0.f;

        __syncthreads();

        for (int k_idx = 0; k_idx < N; ++k_idx){

            float wa_weighted = 0.f;
            for (int action = 0; action < a_dim; ++action){
                // TODO: change/transpose/permute Wa for coalesced memory access ?
                // Maybe already good
                wa_weighted += A[t * a_dim + action] * Wa[action * N * N + row_idx * N + k_idx];
            } // end for action            

            float w_eff = J0[row_idx * N + k_idx] + J1 * Wo[row_idx * N + k_idx] + wa_weighted;

            atomicAdd(&re_inp_buf[threadIdx.x], w_eff * r_buf[k_idx]);

        } // end for k_idx

        __syncthreads();

        float re_inp_act = re_inp_buf[threadIdx.x];
        float re_inp_val = 0.5f * re_inp_act * (1.f + tanhf(kBetaVec * fmaf(0.044715f, re_inp_act * re_inp_act * re_inp_act, re_inp_act)));
        
        float alpha = 0.15f;
        float r_updated = (1.f - alpha) * r_buf[threadIdx.x] + alpha * re_inp_val;

        bump_history[t * N + threadIdx.x] = r_updated;

        for (int k_idx = 0; k_idx < N; ++k_idx){
            atomicAdd(&r_d7_buf[threadIdx.x], r_updated * W_delta7[k_idx * N + threadIdx.x]);
        }// end for k_idx

        __syncthreads();

        // OR
        float r_d7_lane = r_d7_buf[threadIdx.x];
        r_buf[threadIdx.x] = r_updated;

        for (int offset = 128; offset > 0; offset /= 2) {
            if (threadIdx.x < 256 - offset) {
                r_d7_buf[threadIdx.x] = fmaxf(r_d7_buf[threadIdx.x], r_d7_buf[threadIdx.x + offset]);
            }
            __syncthreads();
        } // for offset
        
        r_history[t * N + threadIdx.x] = r_d7_lane / r_d7_buf[0];

    } // end for t

}

__global__ void persistent_splitK_batch_kernel(
    const float* A, // shape [batch_size, 128, 32]
    const float* Wa, // shape [32, 256, 256]
    const float* J0, // shape [256, 256]
    float J1,
    const float* Wo,
    float* r, // shape [batch_size, 256]
    const float* W_delta7,
    const float* W_eff,
    float* bump_history, // shape [batch_size, 128, 256]
    float* r_history, // shape [batch_size, 128, 256]
    int N,
    int a_dim,
    int seq_len,
    int batch_size
){  
    __shared__ float r_buf[256];
    __shared__ float re_inp_buf[256];
    __shared__ float r_d7_buf[256];

    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    // Load initial r for this batch
    r_buf[thread_idx] = r[batch_idx * N + thread_idx];

    __syncthreads();

    constexpr float kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5f;
    
    for (int t = 0; t < seq_len; ++t){

        re_inp_buf[thread_idx] = 0.f;
        r_d7_buf[thread_idx] = 0.f;

        __syncthreads();

        for (int k_idx = 0; k_idx < N; ++k_idx){

            float wa_weighted = 0.f;
            for (int action = 0; action < a_dim; ++action){
                // Access A for current batch
                wa_weighted += A[batch_idx * seq_len * a_dim + t * a_dim + action] * 
                              Wa[action * N * N + thread_idx * N + k_idx];
            } // end for action            

            float w_eff = J0[thread_idx * N + k_idx] + J1 * Wo[thread_idx * N + k_idx] + wa_weighted;

            atomicAdd(&re_inp_buf[thread_idx], w_eff * r_buf[k_idx]);

        } // end for k_idx

        __syncthreads();

        float re_inp_act = re_inp_buf[thread_idx];
        float re_inp_val = 0.5f * re_inp_act * (1.f + tanhf(kBetaVec * fmaf(0.044715f, re_inp_act * re_inp_act * re_inp_act, re_inp_act)));
        
        float alpha = 0.15f;
        float r_updated = (1.f - alpha) * r_buf[thread_idx] + alpha * re_inp_val;

        // Store to batch-specific location
        bump_history[batch_idx * seq_len * N + t * N + thread_idx] = r_updated;

        for (int k_idx = 0; k_idx < N; ++k_idx){
            atomicAdd(&r_d7_buf[thread_idx], r_updated * W_delta7[k_idx * N + thread_idx]);
        }// end for k_idx

        __syncthreads();

        float r_d7_lane = r_d7_buf[thread_idx];
        r_buf[thread_idx] = r_updated;
        
        for (int offset = 128; offset >= 1; offset /= 2) {
            if (thread_idx < offset) {
                r_d7_buf[thread_idx] = fmaxf(r_d7_buf[thread_idx], r_d7_buf[thread_idx + offset]);
            }
            __syncthreads();
        } // for offset
        
        // TODO: Implement prefetching next timestep data, see if it's faster
        // if (t + 1 < seq_len) {
        //     // Prefetch A[batch_idx * seq_len * a_dim + (t+1) * a_dim + ...]
        // }

        // Store to batch-specific location
        r_history[batch_idx * seq_len * N + t * N + thread_idx] = r_d7_lane / r_d7_buf[0];

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

    // persistent_reduce_kernel<<<gridSize, blockSize>>>(
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
}