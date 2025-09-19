// #include <cuda_bf16.h>
// #include <cuda_runtime.h>

// void torch_sum_cuda(
//     void* A_t,
//     void* Wa,
//     void* J0,
//     float J1,
    
//     void* r, // init value for r

//     void* Wa_weighted, // internal use
//     // void* re_inp, // internal use
//     void* Wo,

//     int N,
//     int a_dim
// );

// __global__ void torch_sum_kernel(
//     const __nv_bfloat16* A_t, // shape (32,)
//     const __nv_bfloat16* Wa, // shape [32, 256, 256]
//     const __nv_bfloat16* J0, // shape [256, 256]
//     float J1,
//     const __nv_bfloat16* Wo,

//     __nv_bfloat16* r, // init value for r

//     __nv_bfloat16* Wa_weighted, // [1, 256, 256], internal use
//     // __nv_bfloat16* re_inp, // internal use [1, 256]


//     int N,
//     int a_dim
// ){  
//     int row_idx = blockIdx.x;
//     int col_idx = threadIdx.x;

//     float SQR_2_PI = sqrtf(2.f / M_PIf32);
    

//     // for (int t = 0; t < seq_len; ++t){

//     //     // Step 1:

//     //     Wa_weighted[row_idx * N + col_idx] = Wa[][row_idx][col_idx]

//     // } // end for t
//     __nv_bfloat16 accum = CUDART_ZERO_BF16;

//     for (int action_idx = 0; action_idx < a_dim; ++action_idx){
//         accum += A_t[action_idx] * Wa[action_idx * N * N + row_idx * N + col_idx];
//     } // end for action_idx

//     // Wa_weighted[row_idx * N + col_idx] = accum; // note that this has shape [1, 256, 256]

//     __nv_bfloat16 w_eff;
//     __nv_bfloat16 J1_bf16 = __float2bfloat16(J1);

//     w_eff = J0[row_idx * N + col_idx] + J1_bf16 * Wo[row_idx * N + col_idx] + accum;


//     // Vector matrix multiplication
//     __nv_bfloat16 re_inp_val = r[threadIdx.x] * w_eff;

//     float re_inp_val_f = __bfloat162float(re_inp_val);

//     // GELU activation

//     float re_inp_val_act = 0.5f * re_inp_val_f * (1.f + tanhf(SQR_2_PI * (re_inp_val_f + 0.044715f * re_inp_val_f * re_inp_val_f * re_inp_val_f)));

//     r[blockIdx.x] = __float2bfloat16(re_inp_val_act);
// }

// void torch_sum_cuda(
//     void* A_t,
//     void* Wa,
//     void* J0,
//     float J1,
//     void* Wo,
    
//     void* r, // init value for r

//     void* Wa_weighted, // internal use
//     // void* re_inp, // internal use


//     int N,
//     int a_dim
// ){
//     dim3 blockSize, gridSize;

//     blockSize = N; gridSize = N;
//     torch_sum_kernel<<<gridSize, blockSize>>>(
//         static_cast<const __nv_bfloat16*>(A_t),
//         static_cast<const __nv_bfloat16*>(Wa),
//         static_cast<const __nv_bfloat16*>(J0),
//         J1,
//         static_cast<const __nv_bfloat16*>(Wo),

//         static_cast<__nv_bfloat16*>(r),

//         static_cast<__nv_bfloat16 *>(Wa_weighted), // internal use
//         // static_cast<__nv_bfloat16 *>(re_inp),

//         N,
//         a_dim        
//     );
// }

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

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
    // __nv_bfloat16 accum = CUDART_ZERO_BF16;
    float accum = 0.f;

    for (int action_idx = 0; action_idx < a_dim; ++action_idx){
        __nv_bfloat16 accum_temp = A_t[action_idx] * Wa[action_idx * N * N + row_idx * N + col_idx];
        accum += __bfloat162float(accum_temp);
    } // end for action_idx

    Wa_weighted[row_idx * N + col_idx] = __float2bfloat16(accum);
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