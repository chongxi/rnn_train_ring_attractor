#include "../cuda_common.cuh"

// Helper for atomic max on float
__device__ __forceinline__ void atomicMax_float(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
            __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
}

__global__ void fwd_n128_a23_global_kernel(
    const float* A,
    const float* Wa,
    float J0,
    float J1,
    const float* Wo,
    float* r_init,
    const float* W_delta7,
    float* bump_history,
    float* r_history,
    float alpha,
    int n_neur,
    int a_dim,
    int seq_len,
    float* global_max_buffer
){
    auto cta = cg::this_thread_block();
    auto tile = cg::tiled_partition<128>(cta);
    auto grid = cg::this_grid();
    
    int batch_idx = blockIdx.x;
    int neuron_idx = blockIdx.y * 128 + tile.thread_rank();
    
    if (neuron_idx >= n_neur) return;
    
    float r = r_init[batch_idx * n_neur + neuron_idx];

    for (int t = 0; t < seq_len; ++t){
        // ========== Update r ==========
        float ri_local = 0.f;
        
        for (int k = tile.thread_rank(); k < n_neur; k += 128){
            float wa_weighted = 0.f;
            for (int a = 0; a < a_dim; ++a){
                wa_weighted += A[batch_idx * seq_len * a_dim + t * a_dim + a] 
                             * Wa[a * n_neur * n_neur + neuron_idx * n_neur + k];
            }
            
            float w_eff = J0 + J1 * Wo[neuron_idx * n_neur + k] + wa_weighted;
            ri_local += w_eff * r_init[batch_idx * n_neur + k];
        }
        
        float ri = cg::reduce(tile, ri_local, cg::plus<float>());
        
        float ri_activated = fmaxf(ri, 0.f);
        r = (1.f - alpha) * r + alpha * ri_activated;
        
        bump_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] = r;
        r_init[batch_idx * n_neur + neuron_idx] = r;
        
        grid.sync();  // Grid-wide sync
        
        // ========== Compute r7 ==========
        float r7_local = 0.f;
        for (int k = tile.thread_rank(); k < n_neur; k += 128){
            r7_local += r_init[batch_idx * n_neur + k] * W_delta7[neuron_idx * n_neur + k];
        }
        float r7 = cg::reduce(tile, r7_local, cg::plus<float>());
        
        // ========== Find global max of absolute values ==========
        // Initialize global max buffer (only one thread per batch)
        if (neuron_idx == 0) {
            global_max_buffer[batch_idx * seq_len + t] = 0.f;  // Changed from -INFINITY
        }
        
        grid.sync();
        
        // Each thread contributes absolute value of r7 to find max
        atomicMax_float(&global_max_buffer[batch_idx * seq_len + t], fabsf(r7));
        
        grid.sync();
        
        // Read global max and normalize (preserving sign)
        float global_max = global_max_buffer[batch_idx * seq_len + t];
        
        r_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] = r7 / global_max;
    }
}

void fwd_n128_a23_global_launcher(
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
    int batch_size
){
    int num_blocks_y = (N + 127) / 128;
    dim3 blockSize(128);
    dim3 gridSize(batch_size, num_blocks_y);
    
    // Allocate global max buffer
    float* global_max_buffer;
    cudaMalloc(&global_max_buffer, batch_size * seq_len * sizeof(float));
    
    // Launch with cooperative groups
    void* kernel_args[] = {
        &A, &Wa, &J0, &J1, &Wo, &r_init, &W_delta7,
        &bump_history, &r_history, &alpha, &N, &a_dim, &seq_len,
        &global_max_buffer
    };
    
    cudaLaunchCooperativeKernel(
        (void*)fwd_n128_a23_global_kernel,
        gridSize,
        blockSize,
        kernel_args
    );
    
    cudaFree(global_max_buffer);
}