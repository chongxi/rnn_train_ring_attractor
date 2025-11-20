#include "cuda_common.cuh"
#include "kernels/bwd_simple.cuh"
// #include "kernels/bwd_mixed_precision_wmma.cuh"
#include "kernels/bwd_mixed_precision_aggressive.cuh"
#include "kernels/bwd_full_precision.cuh"

#include "utils.cuh"

void bwd_cuda(
    void* grad_output,
    void* A,
    void* Wa,
    float J0,
    float J1,
    void* Wo,
    void* bump_history,
    void* r_init,
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
    void* r_init,
    void* grad_Wa,
    void* grad_Wo,
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size,
    int activation_type
){
    // cuda_simple::bwd_fp32_launcher(
    //     static_cast<const float*>(grad_output),
    //     static_cast<const float*>(A),
    //     static_cast<const float*>(Wa),
    //     J0,
    //     J1,
    //     static_cast<const float*>(Wo),
    //     static_cast<const float*>(bump_history),
    //     static_cast<float*>(grad_Wa),
    //     static_cast<float*>(grad_Wo),
    //     alpha, N, a_dim, seq_len, batch_size, activation_type
    // );
    // CHECK_CUDA_ERROR(cudaGetLastError());

    // cuda_single::bwd_wmma_launcher(
    //    static_cast<const float*>(grad_output),
    //    static_cast<const float*>(A),
    //    static_cast<const float*>(Wa),
    //    J0,
    //    J1,
    //    static_cast<const float*>(Wo),
    //    static_cast<const float*>(bump_history),
    //    static_cast<const float*>(r_init),
    //    static_cast<float*>(grad_Wa),
    //    static_cast<float*>(grad_Wo),
    //    alpha, N, a_dim, seq_len, batch_size, activation_type
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
       alpha, N, a_dim, seq_len, batch_size, activation_type
    );
    CHECK_CUDA_ERROR(cudaGetLastError());


}

// int main(int argc, char** argv) {
//
//
//     // ---------------------------------------------------------------
//     // Problem size configuration
//     // ---------------------------------------------------------------
//     Config cfg = parse_args(argc, argv);
//     print_config(cfg);
//
//     int grad_output_size = cfg.batch_size * cfg.seq_len * cfg.N;
//     int A_size = cfg.batch_size * cfg.seq_len * cfg.a_dim;
//     int Wa_size = cfg.a_dim * cfg.N * cfg.N;
//     int Wo_size = cfg.N * cfg.N;
//     int bump_history_size = cfg.batch_size * (cfg.seq_len + 1) * cfg.N;
//
//     // ---------------------------------------------------------------
//     // Initialize random seed
//     // ---------------------------------------------------------------
//     srand(42);
//
//     // Allocate and initialize host data
//     thrust::host_vector<float> h_grad_output(grad_output_size);
//     thrust::host_vector<float> h_A(A_size);
//     thrust::host_vector<float> h_Wa(Wa_size);
//     thrust::host_vector<float> h_Wo(Wo_size);
//     thrust::host_vector<float> h_bump_history(bump_history_size);
//     thrust::host_vector<float> h_grad_Wa_fp32(Wa_size, 0.0f);
//     thrust::host_vector<float> h_grad_Wo_fp32(Wo_size, 0.0f);
//     thrust::host_vector<float> h_grad_Wa_wmma(Wa_size, 0.0f);
//     thrust::host_vector<float> h_grad_Wo_wmma(Wo_size, 0.0f);
//
//     for (int i = 0; i < grad_output_size; i++) h_grad_output[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//     for (int i = 0; i < A_size; i++) h_A[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//     for (int i = 0; i < Wa_size; i++) h_Wa[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//     for (int i = 0; i < Wo_size; i++) h_Wo[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//     for (int i = 0; i < bump_history_size; i++) h_bump_history[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//
//
//     // ---------------------------------------------------------------
//     // Allocate device memory and copy data
//     // ---------------------------------------------------------------
//     thrust::device_vector<float> d_grad_output = h_grad_output;
//     thrust::device_vector<float> d_A = h_A;
//     thrust::device_vector<float> d_Wa = h_Wa;
//     thrust::device_vector<float> d_Wo = h_Wo;
//     thrust::device_vector<float> d_bump_history = h_bump_history;
//     thrust::device_vector<float> d_grad_Wa_fp32 = h_grad_Wa_fp32;
//     thrust::device_vector<float> d_grad_Wo_fp32 = h_grad_Wo_fp32;
//     thrust::device_vector<float> d_grad_Wa_wmma = h_grad_Wa_wmma;
//     thrust::device_vector<float> d_grad_Wo_wmma = h_grad_Wo_wmma;
//
//     // ---------------------------------------------------------------
//     // Single run kernels
//     // ---------------------------------------------------------------
//
//     cuda_wmma::bwd_wmma_launcher(
//         d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
//         cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
//         d_grad_Wa_wmma.data().get(), d_grad_Wo_wmma.data().get(),
//         cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
//     );
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//     //
//     // cuda_simple::bwd_fp32_launcher(
//     //     d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
//     //     cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
//     //     d_grad_Wa_wmma.data().get(), d_grad_Wo_wmma.data().get(),
//     //     cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
//     // );
//     // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//     // cuda_single::bwd_wmma_launcher(
//     //     d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
//     //     cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
//     //     d_grad_Wa_wmma.data().get(), d_grad_Wo_wmma.data().get(),
//     //     cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
//     // );
//     // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//
//     // // ---------------------------------------------------------------
//     // // Check correctness
//     // // ---------------------------------------------------------------
//     // if (cfg.run_check) {
//     //     cuda_simple::bwd_fp32_launcher(
//     //         d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
//     //         cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
//     //         d_grad_Wa_fp32.data().get(), d_grad_Wo_fp32.data().get(),
//     //         cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
//     //     );
//     //     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//     //
//     //     std::vector<float> grad_Wa_fp32(Wa_size), grad_Wo_fp32(Wo_size);
//     //     std::vector<float> grad_Wa_wmma(Wa_size), grad_Wo_wmma(Wo_size);
//     //
//     //     CHECK_CUDA_ERROR(cudaMemcpy(grad_Wa_fp32.data(), d_grad_Wa_fp32.data().get(), Wa_size * sizeof(float), cudaMemcpyDeviceToHost));
//     //     CHECK_CUDA_ERROR(cudaMemcpy(grad_Wo_fp32.data(), d_grad_Wo_fp32.data().get(), Wo_size * sizeof(float), cudaMemcpyDeviceToHost));
//     //     CHECK_CUDA_ERROR(cudaMemcpy(grad_Wa_wmma.data(), d_grad_Wa_wmma.data().get(), Wa_size * sizeof(float), cudaMemcpyDeviceToHost));
//     //     CHECK_CUDA_ERROR(cudaMemcpy(grad_Wo_wmma.data(), d_grad_Wo_wmma.data().get(), Wo_size * sizeof(float), cudaMemcpyDeviceToHost));
//     //
//     //     printf("Checking grad_Wa correctness:\n");
//     //     bool wa_correct = allClose(grad_Wa_fp32, grad_Wa_wmma, 0.00001f, 0.000001f, 10);
//     //     printf("\nChecking grad_Wo correctness:\n");
//     //     bool wo_correct = allClose(grad_Wo_fp32, grad_Wo_wmma, 0.00001f, 0.000001f, 10);
//     //     printf("\n%s\n\n", (wa_correct && wo_correct) ? "Correctness check PASSED" : "Correctness check FAILED");
//     // }
//     //
//     // // ---------------------------------------------------------------
//     // // Benchmarking
//     // // ---------------------------------------------------------------
//     // if (cfg.run_benchmark) {
//     //     benchmark_launcher("FP32 Kernel", cuda_fp32::bwd_fp32_launcher, cfg.timing_iterations,
//     //         d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
//     //         cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
//     //         d_grad_Wa_fp32.data().get(), d_grad_Wo_fp32.data().get(),
//     //         cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
//     //     );
//     //
//     //     benchmark_launcher("WMMA Kernel", cuda_wmma::bwd_wmma_launcher, cfg.timing_iterations,
//     //         d_grad_output.data().get(), d_A.data().get(), d_Wa.data().get(),
//     //         cfg.J0, cfg.J1, d_Wo.data().get(), d_bump_history.data().get(),
//     //         d_grad_Wa_wmma.data().get(), d_grad_Wo_wmma.data().get(),
//     //         cfg.alpha, cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size, cfg.activation_type
//     //     );
//     // }
//
//     return 0;
// }