#include "cuda_common.cuh"
// #include "kernels/fwd_n128_a23.cuh"
// #include "kernels/fwd_128_simple.cuh"
// #include "kernels/fwd_3kernels_simple_act_loops.cuh"
// #include "kernels/fwd_loops_opt.cuh"
// #include "kernels/fwd_1loop.cuh"
#include "kernels/fwd_1loop_tc_idx.cuh"
// #include  "kernels/fwd_fp32.cuh"

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
    void* W_delta7,
    void* bump_history,
    void* r_history,
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size,
    int activation_type
){
    fwd_n128_a23_global_launcher(
        A, Wa, J0, J1, Wo, r_init, W_delta7,
        bump_history, r_history, alpha,
        N, a_dim, seq_len, batch_size, activation_type
    );
    
    CHECK_CUDA_ERROR(cudaGetLastError());
}