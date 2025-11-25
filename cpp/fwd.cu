#include "cuda_common.cuh"
#include "kernels/fwd_1loop_tc_idx.cuh"
#include  "kernels/fwd_fp32.cuh"

void fwd_cuda(
    void* A,
    void* Wa,
    float J0,
    float J1,
    void* Wo,
    void* r_init,
    // void* W_delta7,
    void* bump_history,
    // void* r_history,
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size,
    int activation_type
);

void fwd_cuda(
    void* A,
    void* Wa,
    float J0,
    float J1,
    void* Wo,
    void* r_init,
    // void* W_delta7,
    void* bump_history,
    // void* r_history,
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size,
    int activation_type
){
    if (batch_size % 16 != 0 || a_dim % 16 != 0 || N % 16 != 0) {
        // printf("WARNING: Using non-tensor core kernel. Make dimensions (batch_size=%d, a_dim=%d, N=%d) divisible by 16 to activate tensor core version.\n", batch_size, a_dim, N);
        fwd_fp32::fwd_fp32_launcher(
            A, Wa, J0, J1, Wo, r_init, //W_delta7,
            bump_history, //r_history,
            alpha, N, a_dim, seq_len, batch_size, activation_type
        );
    } else {
        fwd_mixed::fwd_wmma_launcher(
            A, Wa, J0, J1, Wo, r_init, //W_delta7,
            bump_history, //r_history,
            alpha, N, a_dim, seq_len, batch_size, activation_type
        );
    }

    // fwd_fp32_launcher(
    //     A, Wa, J0, J1, Wo, r_init, //W_delta7,
    //     bump_history, //r_history,
    //     alpha, N, a_dim, seq_len, batch_size, activation_type
    // );

    CHECK_CUDA_ERROR(cudaGetLastError());
}