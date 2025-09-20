#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
namespace cg = cooperative_groups;


void torch_sum_cuda(
    void* A_t,
    void* Wa,
    void* J0,
    float J1,
    void* Wo,

    void* r, // init value for r

    void* Wa_weighted, // internal use
    void* re_inp, // internal use
    void* W_eff,

    int N,
    int a_dim
);


__device__ float smem_reduce(float val_per_thread) {
    __shared__ float s_data[256];
    
    s_data[threadIdx.x] = val_per_thread;
    __syncthreads();
    
    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    return s_data[0];
}

__device__ float warp_reduce(float val_per_thread) {
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

__global__ void torch_sum_kernel(
    const float* A_t, // shape (32,)
    const float* Wa, // shape [32, 256, 256]
    const float* J0, // shape [256, 256]
    float J1,
    const float* Wo,

    float* r, // init value for r

    float* Wa_weighted, // [1, 256, 256], internal use
    float* re_inp, // internal use [1, 256]
    float* W_eff,

    int N,
    int a_dim
){  
    int row_idx = blockIdx.x;
    int col_idx = threadIdx.x;

    float SQR_2_PI = sqrtf(2.f / M_PIf32);

    // for (int t = 0; t < seq_len; ++t){

    //     // Step 1:

    //     Wa_weighted[row_idx * N + col_idx] = Wa[][row_idx][col_idx]

    // } // end for t
    float accum = 0.f;

    for (int action_idx = 0; action_idx < a_dim; ++action_idx){
        accum += A_t[action_idx] * Wa[action_idx * N * N + row_idx * N + col_idx];
    } // end for action_idx
    

    // Wa_weighted[row_idx * N + col_idx] = __float2bfloat16(accum); // note that this has shape [1, 256, 256]

    float w_eff_temp = J0[row_idx * N + col_idx] + J1 * Wo[row_idx * N + col_idx] + accum;


    float val_per_thread = w_eff_temp * r[col_idx];

    // float result = smem_reduce(val_per_thread);
    // OR
    // float result = warp_reduce(val_per_thread);
    // if (threadIdx.x == 0){
    //     // re_inp[blockIdx.x] = result;

    //     // GELU activation
    //     float re_inp_val_activ = 0.5f * result * (1.f + tanhf(SQR_2_PI * (result + 0.044715f * result * result * result)));
    //     re_inp[blockIdx.x] = re_inp_val_activ;

    //     // float re_inp_val_activ = 0.5f * result * (1.f + tanhf(SQR_2_PI * (result + 0.044715f * result * result * result)));
        
    //     // float alpha = 0.15f;

    //     // float temp = (1.f - alpha) * r[blockIdx.x] + alpha * re_inp_val_activ;
    //     // r[blockIdx.x] = temp;

    // }

    float result = smem_reduce(val_per_thread);
    float re_inp_val_activ = 0.5f * result * (1.f + tanhf(SQR_2_PI * (result + 0.044715f * result * result * result)));

    re_inp[blockIdx.x] = re_inp_val_activ;


    // float alpha = 0.15f;
    // float temp = r[blockIdx.x];
    // r[blockIdx.x] = temp * (1.f - alpha) + re_inp_val_activ * alpha;
}

void torch_sum_cuda(
    void* A_t,
    void* Wa,
    void* J0,
    float J1,
    void* Wo,

    void* r, // init value for r

    void* Wa_weighted, // internal use
    void* re_inp, // internal use
    void* W_eff,

    int N,
    int a_dim
){
    dim3 blockSize, gridSize;

    blockSize = BLOCK_SIZE; gridSize = N;
    torch_sum_kernel<<<gridSize, blockSize>>>(
        static_cast<const float*>(A_t),
        static_cast<const float*>(Wa),
        static_cast<const float*>(J0),
        J1,
        static_cast<const float*>(Wo),

        static_cast<float*>(r),

        static_cast<float *>(Wa_weighted), // internal use
        static_cast<float *>(re_inp),
        static_cast<float *>(W_eff),
        N,
        a_dim        
    );
}