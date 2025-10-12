#include "../cuda_common.cuh"
#include <cublasLt.h>

enum class Activation { RELU, GELU, TANH };

template<Activation ACT>
__device__ __forceinline__ float activation(float x) {
    if constexpr (ACT == Activation::RELU) {
        return fmaxf(x, 0.f);
    } else if constexpr (ACT == Activation::GELU) {
        return 0.5f * x * (1.f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
    } else { // TANH
        return tanhf(x);
    }
}

// Custom epilogue kernel: r = (1-alpha)*r_prev + alpha*activation(gemv_result)
template<Activation ACT>
__global__ void leaky_integration_kernel(
    const float* gemv_result,  // (batch, N)
    const float* r_prev,        // (batch, N)
    float* r_out,               // (batch, N)
    float alpha,
    int batch_size,
    int n_neur
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n_neur;
    
    if (idx < total) {
        float activated = activation<ACT>(gemv_result[idx]);
        r_out[idx] = (1.0f - alpha) * r_prev[idx] + alpha * activated;
    }
}

template<Activation ACT>
void fwd_n128_a23_global_launcher_impl_cublas_fused(
    void* A, void* Wa, float J0, float J1, void* Wo, void* r_init,
    void* W_delta7, void* bump_history, void* r_history, float alpha,
    int N, int a_dim, int seq_len, int batch_size
){
    // Static resources
    static cublasLtHandle_t ltHandle = nullptr;
    static cublasHandle_t handle = nullptr;
    static float *Wa_flat = nullptr;
    static float *Wa_weighted = nullptr;
    static float *W_eff = nullptr;
    static float *A_t = nullptr;
    static float *r_current = nullptr;
    static float *recurrent_input = nullptr;
    static int cached_N = 0, cached_batch = 0, cached_adim = 0;
    
    if (ltHandle == nullptr) {
        cublasLtCreate(&ltHandle);
        cublasCreate(&handle);
    }
    
    if (Wa_flat == nullptr || cached_N != N || cached_batch != batch_size || cached_adim != a_dim) {
        if (Wa_flat) {
            cudaFree(Wa_flat);
            cudaFree(Wa_weighted);
            cudaFree(W_eff);
            cudaFree(A_t);
            cudaFree(r_current);
            cudaFree(recurrent_input);
        }
        
        cudaMalloc(&Wa_flat, a_dim * N * N * sizeof(float));
        cudaMalloc(&Wa_weighted, batch_size * N * N * sizeof(float));
        cudaMalloc(&W_eff, batch_size * N * N * sizeof(float));
        cudaMalloc(&A_t, batch_size * a_dim * sizeof(float));
        cudaMalloc(&r_current, batch_size * N * sizeof(float));
        cudaMalloc(&recurrent_input, batch_size * N * sizeof(float));
        
        cached_N = N;
        cached_batch = batch_size;
        cached_adim = a_dim;
    }
    
    cudaMemcpy(Wa_flat, Wa, a_dim * N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Initialize r_current with r_init
    cudaMemcpy(r_current, r_init, batch_size * N * sizeof(float), cudaMemcpyDeviceToDevice);
    
    float alpha_gemm = 1.0f, beta_gemm = 0.0f;
    
    // Setup cublasLt for GEMM with epilogue
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    
    for (int t = 0; t < seq_len; ++t) {
        bool is_first = (t == 0);
        
        // Extract A[:, t, :]
        for (int b = 0; b < batch_size; ++b) {
            cudaMemcpy(
                A_t + b * a_dim,
                static_cast<float*>(A) + b * seq_len * a_dim + t * a_dim,
                a_dim * sizeof(float),
                cudaMemcpyDeviceToDevice
            );
        }
        
        // Step 1: GEMM with epilogue: W_eff = J0 + J1 * Wo + Wa_weighted
        // First compute: Wa_weighted = A_t @ Wa_flat
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N * N, batch_size, a_dim,
            &alpha_gemm,
            Wa_flat, N * N,
            A_t, a_dim,
            &beta_gemm,
            Wa_weighted, N * N
        );
        
        // Epilogue kernel: W_eff = J0 + J1 * Wo + Wa_weighted
        int total_elements = batch_size * N * N;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        
        auto compute_weff_kernel = [=] __device__ (
            const float* Wo_ptr, const float* Wa_w, float* W_e, 
            float j0, float j1, int n, int batch, int nn
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch * nn * nn) {
                int b = idx / (nn * nn);
                int remainder = idx % (nn * nn);
                int i = remainder / nn;
                int j = remainder % nn;
                
                // Wo is shared across batch, Wa_weighted is per-batch
                float wo_val = Wo_ptr[i * nn + j];
                float wa_val = Wa_w[b * nn * nn + i * nn + j];
                W_e[idx] = j0 + j1 * wo_val + wa_val;
            }
        };
        
        compute_weff_kernel<<<blocks, threads>>>(
            static_cast<float*>(Wo), Wa_weighted, W_eff,
            J0, J1, N, batch_size, N
        );
        
        // Step 2: Batched GEMV: recurrent_input = W_eff @ r_current
        // Use batched GEMV
        float* r_prev = is_first ? static_cast<float*>(r_init) : r_current;
        
        // cublasSgemv doesn't support batched, use Sgemm with N=1
        for (int b = 0; b < batch_size; ++b) {
            cublasSgemv(
                handle,
                CUBLAS_OP_N,
                N, N,
                &alpha_gemm,
                W_eff + b * N * N, N,
                r_prev + b * N, 1,
                &beta_gemm,
                recurrent_input + b * N, 1
            );
        }
        
        // Step 3: Activation + leaky integration epilogue
        int n_elements = batch_size * N;
        int epilogue_blocks = (n_elements + 255) / 256;
        
        leaky_integration_kernel<ACT><<<epilogue_blocks, 256>>>(
            recurrent_input,
            r_prev,
            r_current,
            alpha,
            batch_size,
            N
        );
        
        // Store r_current to bump_history
        cudaMemcpy(
            static_cast<float*>(bump_history) + t * batch_size * N,
            r_current,
            batch_size * N * sizeof(float),
            cudaMemcpyDeviceToDevice
        );
    }
}

void fwd_n128_a23_global_launcher(
    void* A, void* Wa, float J0, float J1, void* Wo, void* r_init,
    void* W_delta7, void* bump_history, void* r_history, float alpha,
    int N, int a_dim, int seq_len, int batch_size, int activation_type
){
    switch(activation_type){
        case 0: fwd_n128_a23_global_launcher_impl_cublas_fused<Activation::RELU>(
            A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
        case 1: fwd_n128_a23_global_launcher_impl_cublas_fused<Activation::GELU>(
            A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
        case 2: fwd_n128_a23_global_launcher_impl_cublas_fused<Activation::TANH>(
            A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
    }
}