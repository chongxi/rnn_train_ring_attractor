#include <cuda_bf16.h>
#include <cuda_runtime.h>

void torch_sum_cuda(
    void* A_t,
    void* Wa,
    
    void* Wa_weighted, // internal use



    int N,
    int a_dim
);

__global__ void torch_sum_kernel(
    const __nv_bfloat16* A_t, // shape (32,)
    const __nv_bfloat16* Wa, // shape [32, 256, 256]

    __nv_bfloat16* Wa_weighted, // [1, 256, 256], internal use



    int N,
    int a_dim
){  
    int row_idx = blockIdx.x;
    int col_idx = threadIdx.x;

    // for (int t = 0; t < seq_len; ++t){

    //     // Step 1:

    //     Wa_weighted[row_idx * N + col_idx] = Wa[][row_idx][col_idx]

    // } // end for t
    __nv_bfloat16 accum = CUDART_ZERO_BF16;

    for (int action_idx = 0; action_idx < a_dim; ++action_idx){
        accum += A_t[action_idx] * Wa[action_idx * N * N + row_idx * N + col_idx];
    } // end for action_idx

    Wa_weighted[row_idx * N + col_idx] = accum;
}

void torch_sum_cuda(
    void* A_t,
    void* Wa,

    
    void* Wa_weighted, // internal use


    int N,
    int a_dim
){
    dim3 blockSize, gridSize;

    blockSize = N; gridSize = N;
    torch_sum_kernel<<<gridSize, blockSize>>>(
        static_cast<const __nv_bfloat16*>(A_t),
        static_cast<const __nv_bfloat16*>(Wa),
        
        static_cast<__nv_bfloat16 *>(Wa_weighted), // internal use

        N,
        a_dim        
    );
}