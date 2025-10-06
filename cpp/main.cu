/*
    PURPOSE: for Nsight Compute profiling
*/

#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "kernels/fwd_3kernels_simple.cuh"

int main() {
    // Parameters
    const int batch_size = 2;
    const int n_neur = 256;      // Number of neurons (N)
    const int a_dim = 3;         // Action dimension
    const int seq_len = 10;      // Sequence length
    const float J0 = 0.5f;
    const float J1 = 0.3f;
    const float alpha = 0.1f;
    
    // Host memory allocation and initialization
    std::vector<float> h_A(batch_size * seq_len * a_dim);
    std::vector<float> h_Wa(a_dim * n_neur * n_neur);
    std::vector<float> h_Wo(n_neur * n_neur);
    std::vector<float> h_r_init(batch_size * n_neur);
    std::vector<float> h_W_delta7(n_neur * n_neur);
    std::vector<float> h_bump_history(batch_size * seq_len * n_neur);
    std::vector<float> h_r_history(batch_size * seq_len * n_neur);
    
    // Initialize with random values
    std::default_random_engine generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    
    for (auto& val : h_A) val = distribution(generator);
    for (auto& val : h_Wa) val = distribution(generator) * 0.1f;
    for (auto& val : h_Wo) val = distribution(generator) * 0.1f;
    for (auto& val : h_r_init) val = std::abs(distribution(generator));
    for (auto& val : h_W_delta7) val = distribution(generator) * 0.1f;
    
    // Device memory allocation
    float *d_A, *d_Wa, *d_Wo, *d_r_init, *d_W_delta7, *d_bump_history, *d_r_history;
    
    cudaMalloc(&d_A, batch_size * seq_len * a_dim * sizeof(float));
    cudaMalloc(&d_Wa, a_dim * n_neur * n_neur * sizeof(float));
    cudaMalloc(&d_Wo, n_neur * n_neur * sizeof(float));
    cudaMalloc(&d_r_init, batch_size * n_neur * sizeof(float));
    cudaMalloc(&d_W_delta7, n_neur * n_neur * sizeof(float));
    cudaMalloc(&d_bump_history, batch_size * seq_len * n_neur * sizeof(float));
    cudaMalloc(&d_r_history, batch_size * seq_len * n_neur * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), batch_size * seq_len * a_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wa, h_Wa.data(), a_dim * n_neur * n_neur * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wo, h_Wo.data(), n_neur * n_neur * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_init, h_r_init.data(), batch_size * n_neur * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_delta7, h_W_delta7.data(), n_neur * n_neur * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "Running forward pass..." << std::endl;
    
    // Launch the kernel
    fwd_n128_a23_global_launcher(
        d_A, d_Wa, J0, J1, d_Wo, d_r_init, d_W_delta7,
        d_bump_history, d_r_history, alpha,
        n_neur, a_dim, seq_len, batch_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_bump_history.data(), d_bump_history, batch_size * seq_len * n_neur * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_history.data(), d_r_history, batch_size * seq_len * n_neur * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Forward pass completed successfully!" << std::endl;
    std::cout << "Sample bump_history values (first 5): ";
    for (int i = 0; i < 5 && i < h_bump_history.size(); ++i) {
        std::cout << h_bump_history[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Sample r_history values (first 5): ";
    for (int i = 0; i < 5 && i < h_r_history.size(); ++i) {
        std::cout << h_r_history[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_Wa);
    cudaFree(d_Wo);
    cudaFree(d_r_init);
    cudaFree(d_W_delta7);
    cudaFree(d_bump_history);
    cudaFree(d_r_history);
    
    std::cout << "Memory cleaned up. Exiting." << std::endl;
    
    return 0;
}