#include "cuda_common.cuh"
#include "kernels/bwd_simple.cuh"

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

void bwd_cuda(
    void* grad_output,
    void* A,
    void* Wa,
    float J0,
    float J1,
    void* Wo,
    void* bump_history,
    void* grad_Wa,
    void* grad_Wo,
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size,
    int activation_type
);

void bwd_cuda(
    void* grad_output,
    void* A,
    void* Wa,
    float J0,
    float J1,
    void* Wo,
    void* bump_history,
    void* grad_Wa,
    void* grad_Wo,
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size,
    int activation_type
){
    // if (batch_size % 16 != 0 || a_dim % 16 != 0 || N % 16 != 0) {
    //     printf("ERROR: Backward pass for dimension not divisible by 16 is not implemented\n");
    //     // bwd_fp32_launcher(
    //     //     static_cast<const float*>(grad_output),
    //     //     static_cast<const float*>(A),
    //     //     static_cast<const float*>(Wa),
    //     //     static_cast<const float*>(J0),
    //     //     J1,
    //     //     static_cast<const float*>(Wo),
    //     //     static_cast<const float*>(bump_history),
    //     //     static_cast<float*>(grad_Wa),
    //     //     static_cast<float*>(grad_Wo),
    //     //     alpha, N, a_dim, seq_len, batch_size, activation_type
    //     // );
    // } else {
        bwd_wmma_launcher(
            static_cast<const float*>(grad_output),
            static_cast<const float*>(A),
            static_cast<const float*>(Wa),
            J0,
            J1,
            static_cast<const float*>(Wo),
            static_cast<const float*>(bump_history),
            static_cast<float*>(grad_Wa),
            static_cast<float*>(grad_Wo),
            alpha, N, a_dim, seq_len, batch_size, activation_type
        );
    // }

    CHECK_CUDA_ERROR(cudaGetLastError());
}