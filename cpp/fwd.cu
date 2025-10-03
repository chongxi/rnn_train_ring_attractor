#include "cuda_common.cuh"
#include "kernels/fwd_hardcode.cuh"
#include "kernels/fwd_n128_a23.cuh"

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

void fwd_cuda(
    void* A,
    void* Wa,
    // void* J0,
    float J0,
    float J1,
    void* Wo,
    void* r_init,
    void* W_delta7,
    void* bump_history,
    void* r_history,
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size
);

void fwd_cuda(
    void* A,
    void* Wa,
    // void* J0,
    float J0,
    float J1,
    void* Wo,
    void* r_init,
    void* W_delta7,
    void* bump_history,
    void* r_history,
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size
){

    if (N == 128 && a_dim < 4){
        fwd_n128_a23_kernel_launcher(
            A,
            Wa,
            J0,
            J1,
            Wo,
            r_init,
            W_delta7,
            bump_history,
            r_history,
            alpha,
            N,
            a_dim,
            seq_len,
            batch_size
        );

        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    else if (N == 256){
        hardcode_kernel_launcher(
            A,
            Wa,
            J0,
            J1,
            Wo,
            r_init,
            W_delta7,
            bump_history,
            r_history,
            alpha,
            N,
            a_dim,
            seq_len,
            batch_size
        );

        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    else {
        std::cerr << "Error: N=" << N << ", a_dim=" << a_dim << " not supported" << std::endl;
        return;
    }

}


// __device__ float smem_reduce(float val_per_thread) {
//     __shared__ float s_data[256];
    
//     s_data[threadIdx.x] = val_per_thread;
//     __syncthreads();
    
//     for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
//         if (threadIdx.x < stride) {
//             s_data[threadIdx.x] += s_data[threadIdx.x + stride];
//         }
//         __syncthreads();
//     }
    
//     return s_data[0];
// }

// __device__ float warp_reduce(float val_per_thread) {
//     auto cta = cg::this_thread_block();
//     int warp_idx = cta.thread_rank() / WARP_SIZE;
//     int warp_lane_idx = cta.thread_rank() % WARP_SIZE;
    
//     float warp_sum = val_per_thread;
//     for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
//         warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
//     }
    
//     __shared__ float temp_reduction[32]; // only need float temp_reduction[8], but set 32 to remove if statement
    
//     // First lane in each warp writes to shared memory
//     if (warp_lane_idx == 0) {
//         temp_reduction[warp_idx] = warp_sum;
//     }
    
//     __syncthreads();
    
//     float reduction_result = 0.f;

//     // First warp performs final reduction
//     if (warp_idx == 0) {
//         reduction_result = temp_reduction[warp_lane_idx];
        
//         // Reduce 8 warp results
//         for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
//             reduction_result += __shfl_down_sync(0xffffffff, reduction_result, offset);
//         }
//     }

//     return reduction_result;
// }

// __global__ void persistent_splitK_kernel(
//     const float* A, // shape [BATCH_SIZE, 128, 32]
//     const float* Wa, // shape [32, 256, 256]
//     const float* J0, // shape [256, 256]
//     float J1,
//     const float* Wo,
//     float* r,
//     const float* W_delta7,
//     const float* W_eff,
//     float* bump_history,
//     float* r_history,
//     int N,
//     int A_DIM,
//     int SEQ_LEN,
//     int BATCH_SIZE
// ){  
//     __shared__ float r_buf[BLOCK_SIZE];
//     __shared__ float re_inp_buf[BLOCK_SIZE];
//     __shared__ float r_d7_buf[BLOCK_SIZE];

//     int batch_idx = blockIdx.x;
    
//     r_buf[threadIdx.x] = r[batch_idx * N + threadIdx.x];

//     constexpr float kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5f;

//     int row_idx = threadIdx.x;
    
//     for (int t = 0; t < SEQ_LEN; ++t){

//         re_inp_buf[threadIdx.x] = 0.f;
//         r_d7_buf[threadIdx.x] = 0.f;

//         __syncthreads();

//         for (int k_idx = 0; k_idx < N / CLUSTER_SIZE; ++k_idx){
//             float wa_weighted = 0.f;
//             for (int action = 0; action < A_DIM; ++action){
//                 wa_weighted += A[batch_idx * SEQ_LEN * A_DIM + t * A_DIM + action] * Wa[action * N * N + row_idx * N + k_idx];
//             }
            
//             float w_eff = J0[row_idx * N + k_idx] + J1 * Wo[row_idx * N + k_idx] + wa_weighted;
//             atomicAdd(&re_inp_buf[threadIdx.x], w_eff * r_buf[k_idx]);
//             // re_inp_buf[threadIdx.x] += w_eff * r_buf[k_idx];
//         }


//         __syncthreads();

//         float re_inp_act = re_inp_buf[threadIdx.x];
//         // float re_inp_val = 0.5f * re_inp_act * (1.f + tanhf(kBetaVec * fmaf(0.044715f, re_inp_act * re_inp_act * re_inp_act, re_inp_act)));

//         float re_inp_val = fmax(re_inp_act, 0.f);
        
//         float alpha = 0.15f;
//         float r_updated = (1.f - alpha) * r_buf[threadIdx.x] + alpha * re_inp_val;

//         bump_history[batch_idx * SEQ_LEN * N + t * N + threadIdx.x] = r_updated;
//         r_buf[threadIdx.x] = r_updated;

//         __syncthreads();

//         for (int k_idx = 0; k_idx < N; ++k_idx){
//             atomicAdd(&r_d7_buf[threadIdx.x], r_buf[k_idx] * W_delta7[k_idx * N + threadIdx.x]);
//         }

//         __syncthreads();

//         float r_d7_lane = r_d7_buf[threadIdx.x];

//         // Fixed reduction for finding maximum
//         for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
//             if (threadIdx.x < offset) {
//                 r_d7_buf[threadIdx.x] = fmaxf(r_d7_buf[threadIdx.x], r_d7_buf[threadIdx.x + offset]);
//             }
//             __syncthreads();
//         }
        
//         r_history[batch_idx * SEQ_LEN * N + t * N + threadIdx.x] = r_d7_lane / r_d7_buf[0];

//     } // end for t
// }
