#include "cuda_common.cuh"
#include "kernels/bwd_mixed_precision_aggressive.cuh"

void bwd_cuda(
    void* grad_output,
    void* A,
    void* Wa,
    float J0,
    float J1,
    void* Wo,
    void* bump_history,
    void* re_history,
    void* r_init,
    void* grad_Wa,
    void* grad_Wo,
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size,
    int activation_type,
    // Workspace buffers
    void* grad_r,
    void* W_eff_temp,
    void* grad_re_temp,
    void* grad_W_eff_temp_bf16,
    void* A_t_temp_bf16,
    void* grad_re_temp_T,
    void* cublas_handle
){
    cuda_wmma::bwd_wmma_launcher(
       static_cast<const float*>(grad_output),
       static_cast<const float*>(A),
       static_cast<const float*>(Wa),
       J0,
       J1,
       static_cast<const float*>(Wo),
       static_cast<const float*>(bump_history),
       static_cast<const float*>(r_init),
       static_cast<float*>(grad_Wa),
       static_cast<float*>(grad_Wo),
       alpha, N, a_dim, seq_len, batch_size, activation_type,
       static_cast<float*>(grad_r),
       static_cast<float*>(W_eff_temp),
       static_cast<float*>(grad_re_temp),
       static_cast<__nv_bfloat16*>(grad_W_eff_temp_bf16),
       static_cast<__nv_bfloat16*>(A_t_temp_bf16),
       static_cast<float*>(grad_re_temp_T),
       static_cast<cublasHandle_t>(cublas_handle)
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

