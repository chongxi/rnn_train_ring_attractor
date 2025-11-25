#include "cuda_common.cuh"
#include "kernels/bwd_simple.cuh"
// #include "kernels/bwd_mixed_precision_wmma.cuh"
#include "kernels/bwd_mixed_precision_aggressive.cuh"
#include "kernels/bwd_mixed_precision_aggo_re.cuh"
// #include "kernels/bwd_full_precision.cuh"

#include "utils.cuh"

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
    // NEW: workspace buffers
    void* grad_r,
    void* W_eff_temp,
    void* grad_re_temp,
    void* grad_W_eff_temp_bf16,
    void* A_t_temp_bf16,
    void* grad_re_temp_T,       // NEW: added this parameter
    void* cublas_handle         // NEW: cuBLAS handle
){
    // // New WMMA version (uses saved re_history from forward)
    // cuda_wmma_re::bwd_wmma_launcher(
    //    static_cast<const float*>(grad_output),
    //    static_cast<const float*>(A),
    //    static_cast<const float*>(Wa),
    //    J0,
    //    J1,
    //    static_cast<const float*>(Wo),
    //    static_cast<const float*>(bump_history),
    //    static_cast<const float*>(r_init),
    //    static_cast<const float*>(re_history),  // saved re from forward
    //    static_cast<float*>(grad_Wa),
    //    static_cast<float*>(grad_Wo),
    //    alpha, N, a_dim, seq_len, batch_size, activation_type,
    //    // NEW: workspace buffers
    //    static_cast<float*>(grad_r),
    //    static_cast<float*>(W_eff_temp),
    //    static_cast<float*>(grad_re_temp),
    //    static_cast<__nv_bfloat16*>(grad_W_eff_temp_bf16),
    //    static_cast<__nv_bfloat16*>(A_t_temp_bf16),
    //    static_cast<float*>(grad_re_temp_T),           // NEW: added this parameter
    //    static_cast<cublasHandle_t>(cublas_handle)     // NEW: pass handle
    // );
    // CHECK_CUDA_ERROR(cudaGetLastError());

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
       // NEW: workspace buffers
       static_cast<float*>(grad_r),
       static_cast<float*>(W_eff_temp),
       static_cast<float*>(grad_re_temp),
       static_cast<__nv_bfloat16*>(grad_W_eff_temp_bf16),
       static_cast<__nv_bfloat16*>(A_t_temp_bf16),
       static_cast<float*>(grad_re_temp_T),           // NEW: added this parameter
       static_cast<cublasHandle_t>(cublas_handle)     // NEW: pass handle
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}


int main(int argc, char** argv) {

    // ---------------------------------------------------------------
    // Problem size configuration
    // ---------------------------------------------------------------
    Config cfg = parse_args(argc, argv);
    print_config(cfg);

    int grad_output_size = cfg.batch_size * cfg.seq_len * cfg.N;
    int A_size = cfg.batch_size * cfg.seq_len * cfg.a_dim;
    int Wa_size = cfg.a_dim * cfg.N * cfg.N;
    int Wo_size = cfg.N * cfg.N;
    int bump_history_size = cfg.batch_size * cfg.seq_len * cfg.N;  // [S, B, N]
    int re_history_size = cfg.batch_size * cfg.seq_len * cfg.N;    // [S, B, N]
    int r_init_size = cfg.batch_size * cfg.N;

    // NEW: workspace buffer sizes
    int grad_r_size = cfg.batch_size * cfg.N;                      // [B, N]
    int W_eff_temp_size = cfg.batch_size * cfg.N * cfg.N;          // [B, N, N]
    int grad_re_temp_size = cfg.batch_size * cfg.N;                // [B, N]
    int grad_W_eff_temp_bf16_size = cfg.batch_size * cfg.N * cfg.N; // [B, N, N]
    int A_t_temp_bf16_size = cfg.batch_size * cfg.a_dim;           // [B, a_dim]
    int grad_re_temp_T_size = cfg.batch_size * cfg.N;              // NEW: [B, N]

    // ---------------------------------------------------------------
    // Initialize random seed
    // ---------------------------------------------------------------
    srand(42);

    // Allocate and initialize host data
    thrust::host_vector<float> h_grad_output(grad_output_size);
    thrust::host_vector<float> h_A(A_size);
    thrust::host_vector<float> h_Wa(Wa_size);
    thrust::host_vector<float> h_Wo(Wo_size);
    thrust::host_vector<float> h_bump_history(bump_history_size);
    thrust::host_vector<float> h_re_history(re_history_size);
    thrust::host_vector<float> h_r_init(r_init_size);
    thrust::host_vector<float> h_grad_Wa_fp32(Wa_size, 0.0f);
    thrust::host_vector<float> h_grad_Wo_fp32(Wo_size, 0.0f);
    thrust::host_vector<float> h_grad_Wa_wmma(Wa_size, 0.0f);
    thrust::host_vector<float> h_grad_Wo_wmma(Wo_size, 0.0f);
    thrust::host_vector<float> h_grad_Wa_wmma_re(Wa_size, 0.0f);
    thrust::host_vector<float> h_grad_Wo_wmma_re(Wo_size, 0.0f);

    for (int i = 0; i < grad_output_size; i++) h_grad_output[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    for (int i = 0; i < A_size; i++) h_A[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    for (int i = 0; i < Wa_size; i++) h_Wa[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    for (int i = 0; i < Wo_size; i++) h_Wo[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    for (int i = 0; i < bump_history_size; i++) h_bump_history[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    for (int i = 0; i < re_history_size; i++) h_re_history[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    for (int i = 0; i < r_init_size; i++) h_r_init[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

    // ---------------------------------------------------------------
    // Create cuBLAS handle
    // ---------------------------------------------------------------
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // ---------------------------------------------------------------
    // Allocate device memory and copy data
    // ---------------------------------------------------------------
    thrust::device_vector<float> d_grad_output = h_grad_output;
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_Wa = h_Wa;
    thrust::device_vector<float> d_Wo = h_Wo;
    thrust::device_vector<float> d_bump_history = h_bump_history;
    thrust::device_vector<float> d_re_history = h_re_history;
    thrust::device_vector<float> d_r_init = h_r_init;
    thrust::device_vector<float> d_grad_Wa_fp32 = h_grad_Wa_fp32;
    thrust::device_vector<float> d_grad_Wo_fp32 = h_grad_Wo_fp32;
    thrust::device_vector<float> d_grad_Wa_wmma = h_grad_Wa_wmma;
    thrust::device_vector<float> d_grad_Wo_wmma = h_grad_Wo_wmma;
    thrust::device_vector<float> d_grad_Wa_wmma_re = h_grad_Wa_wmma_re;
    thrust::device_vector<float> d_grad_Wo_wmma_re = h_grad_Wo_wmma_re;

    // NEW: Allocate workspace buffers
    thrust::device_vector<float> d_grad_r(grad_r_size);
    thrust::device_vector<float> d_W_eff_temp(W_eff_temp_size);
    thrust::device_vector<float> d_grad_re_temp(grad_re_temp_size);
    thrust::device_vector<__nv_bfloat16> d_grad_W_eff_temp_bf16(grad_W_eff_temp_bf16_size);
    thrust::device_vector<__nv_bfloat16> d_A_t_temp_bf16(A_t_temp_bf16_size);
    thrust::device_vector<float> d_grad_re_temp_T(grad_re_temp_T_size);  // NEW: allocate this buffer

    // ---------------------------------------------------------------
    // Single run kernels (warmup)
    // ---------------------------------------------------------------
    // New WMMA with saved re_history
    // cuda_wmma_re::bwd_wmma_launcher(
    //     d_grad_output.data().get(),
    //     d_A.data().get(),
    //     d_Wa.data().get(),
    //     cfg.J0, cfg.J1,
    //     d_Wo.data().get(),
    //     d_bump_history.data().get(),
    //     d_r_init.data().get(),
    //     d_re_history.data().get(),  // saved re from forward
    //     d_grad_Wa_wmma_re.data().get(),
    //     d_grad_Wo_wmma_re.data().get(),
    //     cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type,
    //     // NEW: workspace buffers
    //     d_grad_r.data().get(),
    //     d_W_eff_temp.data().get(),
    //     d_grad_re_temp.data().get(),
    //     d_grad_W_eff_temp_bf16.data().get(),
    //     d_A_t_temp_bf16.data().get(),
    //     d_grad_re_temp_T.data().get(),  // NEW: added this parameter
    //     cublas_handle  // NEW: pass handle
    // );
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    cuda_wmma::bwd_wmma_launcher(
        d_grad_output.data().get(),
        d_A.data().get(),
        d_Wa.data().get(),
        cfg.J0, cfg.J1,
        d_Wo.data().get(),
        d_bump_history.data().get(),
        d_r_init.data().get(),
        d_grad_Wa_wmma_re.data().get(),
        d_grad_Wo_wmma_re.data().get(),
        cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type,
        // NEW: workspace buffers
        d_grad_r.data().get(),
        d_W_eff_temp.data().get(),
        d_grad_re_temp.data().get(),
        d_grad_W_eff_temp_bf16.data().get(),
        d_A_t_temp_bf16.data().get(),
        d_grad_re_temp_T.data().get(),  // NEW: added this parameter
        cublas_handle  // NEW: pass handle
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // ---------------------------------------------------------------
    // Cleanup cuBLAS handle
    // ---------------------------------------------------------------
    cublasDestroy(cublas_handle);

    // // Original WMMA (recomputes re)
    // cuda_wmma::bwd_wmma_launcher(
    //     d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
    //     cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
    //     d_r_init.data().get(), d_grad_Wa_wmma.data().get(), d_grad_Wo_wmma.data().get(),
    //     cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
    // );
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //
    // // Reference FP32
    // cuda_simple::bwd_fp32_launcher(
    //     d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
    //     cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
    //     d_grad_Wa_fp32.data().get(), d_grad_Wo_fp32.data().get(),
    //     cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
    // );
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //
    // // Single kernel baseline
    // cuda_single::bwd_single_launcher(
    //     d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
    //     cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
    //     d_r_init.data().get(), d_grad_Wa_fp32.data().get(), d_grad_Wo_fp32.data().get(),
    //     cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
    // );
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //
    // // ---------------------------------------------------------------
    // // Check correctness
    // // ---------------------------------------------------------------
    // if (cfg.run_check) {
    //     // Reset gradients
    //     cudaMemset(d_grad_Wa_fp32.data().get(), 0, Wa_size * sizeof(float));
    //     cudaMemset(d_grad_Wo_fp32.data().get(), 0, Wo_size * sizeof(float));
    //     cudaMemset(d_grad_Wa_wmma.data().get(), 0, Wa_size * sizeof(float));
    //     cudaMemset(d_grad_Wo_wmma.data().get(), 0, Wo_size * sizeof(float));
    //     cudaMemset(d_grad_Wa_wmma_re.data().get(), 0, Wa_size * sizeof(float));
    //     cudaMemset(d_grad_Wo_wmma_re.data().get(), 0, Wo_size * sizeof(float));
    //
    //     // Run reference
    //     cuda_simple::bwd_fp32_launcher(
    //         d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
    //         cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
    //         d_grad_Wa_fp32.data().get(), d_grad_Wo_fp32.data().get(),
    //         cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
    //     );
    //     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //
    //     // Run WMMA (recomputes re)
    //     cuda_wmma::bwd_wmma_launcher(
    //         d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
    //         cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
    //         d_r_init.data().get(), d_grad_Wa_wmma.data().get(), d_grad_Wo_wmma.data().get(),
    //         cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
    //     );
    //     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //
    //     // Run WMMA with saved re
    //     cuda_wmma_re::bwd_wmma_launcher(
    //         d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
    //         cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
    //         d_r_init.data().get(), d_re_history.data().get(),
    //         d_grad_Wa_wmma_re.data().get(), d_grad_Wo_wmma_re.data().get(),
    //         cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
    //     );
    //     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //
    //     std::vector<float> grad_Wa_fp32(Wa_size), grad_Wo_fp32(Wo_size);
    //     std::vector<float> grad_Wa_wmma(Wa_size), grad_Wo_wmma(Wo_size);
    //     std::vector<float> grad_Wa_wmma_re(Wa_size), grad_Wo_wmma_re(Wo_size);
    //
    //     CHECK_CUDA_ERROR(cudaMemcpy(grad_Wa_fp32.data(), d_grad_Wa_fp32.data().get(), Wa_size * sizeof(float), cudaMemcpyDeviceToHost));
    //     CHECK_CUDA_ERROR(cudaMemcpy(grad_Wo_fp32.data(), d_grad_Wo_fp32.data().get(), Wo_size * sizeof(float), cudaMemcpyDeviceToHost));
    //     CHECK_CUDA_ERROR(cudaMemcpy(grad_Wa_wmma.data(), d_grad_Wa_wmma.data().get(), Wa_size * sizeof(float), cudaMemcpyDeviceToHost));
    //     CHECK_CUDA_ERROR(cudaMemcpy(grad_Wo_wmma.data(), d_grad_Wo_wmma.data().get(), Wo_size * sizeof(float), cudaMemcpyDeviceToHost));
    //     CHECK_CUDA_ERROR(cudaMemcpy(grad_Wa_wmma_re.data(), d_grad_Wa_wmma_re.data().get(), Wa_size * sizeof(float), cudaMemcpyDeviceToHost));
    //     CHECK_CUDA_ERROR(cudaMemcpy(grad_Wo_wmma_re.data(), d_grad_Wo_wmma_re.data().get(), Wo_size * sizeof(float), cudaMemcpyDeviceToHost));
    //
    //     printf("=== WMMA (recompute re) vs FP32 ===\n");
    //     printf("Checking grad_Wa correctness:\n");
    //     bool wa_correct = allClose(grad_Wa_fp32, grad_Wa_wmma, 0.00001f, 0.000001f, 10);
    //     printf("\nChecking grad_Wo correctness:\n");
    //     bool wo_correct = allClose(grad_Wo_fp32, grad_Wo_wmma, 0.00001f, 0.000001f, 10);
    //     printf("\n%s\n\n", (wa_correct && wo_correct) ? "WMMA (recompute) check PASSED" : "WMMA (recompute) check FAILED");
    //
    //     printf("=== WMMA (saved re) vs FP32 ===\n");
    //     printf("Checking grad_Wa correctness:\n");
    //     bool wa_re_correct = allClose(grad_Wa_fp32, grad_Wa_wmma_re, 0.00001f, 0.000001f, 10);
    //     printf("\nChecking grad_Wo correctness:\n");
    //     bool wo_re_correct = allClose(grad_Wo_fp32, grad_Wo_wmma_re, 0.00001f, 0.000001f, 10);
    //     printf("\n%s\n\n", (wa_re_correct && wo_re_correct) ? "WMMA (saved re) check PASSED" : "WMMA (saved re) check FAILED");
    // }
    //
    // // ---------------------------------------------------------------
    // // Benchmarking
    // // ---------------------------------------------------------------
    // if (cfg.run_benchmark) {
    //     printf("\n=== BENCHMARK RESULTS ===\n\n");
    //
    //     benchmark_launcher("FP32 Reference", cuda_simple::bwd_fp32_launcher, cfg.timing_iterations,
    //         d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
    //         cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
    //         d_grad_Wa_fp32.data().get(), d_grad_Wo_fp32.data().get(),
    //         cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
    //     );
    //
    //     benchmark_launcher("Single Kernel", cuda_single::bwd_single_launcher, cfg.timing_iterations,
    //         d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
    //         cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
    //         d_r_init.data().get(), d_grad_Wa_fp32.data().get(), d_grad_Wo_fp32.data().get(),
    //         cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
    //     );
    //
    //     benchmark_launcher("WMMA (recompute re)", cuda_wmma::bwd_wmma_launcher, cfg.timing_iterations,
    //         d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
    //         cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
    //         d_r_init.data().get(), d_grad_Wa_wmma.data().get(), d_grad_Wo_wmma.data().get(),
    //         cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
    //     );
    //
    //     benchmark_launcher("WMMA (saved re)", cuda_wmma_re::bwd_wmma_launcher, cfg.timing_iterations,
    //         d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
    //         cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
    //         d_r_init.data().get(), d_re_history.data().get(),
    //         d_grad_Wa_wmma_re.data().get(), d_grad_Wo_wmma_re.data().get(),
    //         cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
    //     );
    // }

    return 0;
}

