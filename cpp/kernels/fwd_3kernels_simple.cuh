#include "../cuda_common.cuh"

__global__ void fwd_update_r_kernel(
    const float* A,
    const float* Wa,
    float J0,
    float J1,
    const float* Wo,
    float* r_init,
    float* bump_history,
    float alpha,
    int n_neur,
    int a_dim,
    int seq_len,
    int t
){
    auto cta = cg::this_thread_block();
    auto tile = cg::tiled_partition<128>(cta);
    
    int batch_idx = blockIdx.x;
    int neuron_idx = blockIdx.y * 128 + tile.thread_rank();
    
    if (neuron_idx >= n_neur) return;
    
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
    float r = (1.f - alpha) * r_init[batch_idx * n_neur + neuron_idx] + alpha * ri_activated;
    
    bump_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] = r;
    r_init[batch_idx * n_neur + neuron_idx] = r;
}

__global__ void fwd_compute_r7_kernel(
    const float* r_init,
    const float* W_delta7,
    float* r_history,
    int n_neur,
    int seq_len,
    int t
){
    auto cta = cg::this_thread_block();
    auto tile = cg::tiled_partition<128>(cta);
    
    int batch_idx = blockIdx.x;
    int neuron_idx = blockIdx.y;  // Each block handles one neuron
    
    float r7_local = 0.f;
    
    // Stripe across n_neur with tile, each thread accumulates partial sum
    for (int k = tile.thread_rank(); k < n_neur; k += 128){
        r7_local += r_init[batch_idx * n_neur + k] 
                  * W_delta7[neuron_idx * n_neur + k];
    }
    
    // Reduce across tile to get final r7 value
    float r7 = cg::reduce(tile, r7_local, cg::plus<float>());
    
    // Only thread 0 writes the result
    if (tile.thread_rank() == 0){
        r_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] = r7;
    }
}

__global__ void fwd_normalize_r7_kernel(
    float* r_history,
    int n_neur,
    int seq_len,
    int t
){
    extern __shared__ float tile_max[];
    
    auto tile = cg::tiled_partition<128>(cg::this_thread_block());
    
    int batch_idx = blockIdx.x;
    int num_tiles = (n_neur + 127) / 128;
    
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx){
        int neuron_idx = tile_idx * 128 + tile.thread_rank();
        float r7 = (neuron_idx < n_neur) ? 
            r_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] : -INFINITY;
        
        float local_max = cg::reduce(tile, r7, cg::greater<float>());
        
        if (tile.thread_rank() == 0){
            tile_max[tile_idx] = local_max;
        }
    }
    
    __syncthreads();
    
    float global_max = tile_max[0];
    if (tile.thread_rank() < num_tiles){
        for (int i = tile.thread_rank(); i < num_tiles; i += 128){
            global_max = fmaxf(global_max, tile_max[i]);
        }
    }
    global_max = cg::reduce(tile, global_max, cg::greater<float>());
    
    for (int neuron_idx = tile.thread_rank(); neuron_idx < n_neur; neuron_idx += 128){
        float r7 = r_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx];
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
    
    dim3 blockSize1(128);
    dim3 gridSize1(batch_size, num_blocks_y);
    
    // Changed: one block per neuron, not per 128 neurons
    dim3 blockSize2(128);
    dim3 gridSize2(batch_size, N);  // N blocks in y dimension
    
    dim3 blockSize3(128);
    dim3 gridSize3(batch_size);
    
    size_t smem_size = num_blocks_y * sizeof(float);

    for (int t = 0; t < seq_len; ++t){
        fwd_update_r_kernel<<<gridSize1, blockSize1>>>(
            static_cast<const float*>(A),
            static_cast<const float*>(Wa),
            J0, J1,
            static_cast<const float*>(Wo),
            static_cast<float*>(r_init),
            static_cast<float*>(bump_history),
            alpha, N, a_dim, seq_len, t
        );
        
        fwd_compute_r7_kernel<<<gridSize2, blockSize2>>>(
            static_cast<const float*>(r_init),
            static_cast<const float*>(W_delta7),
            static_cast<float*>(r_history),
            N, seq_len, t
        );
        
        fwd_normalize_r7_kernel<<<gridSize3, blockSize3, smem_size>>>(
            static_cast<float*>(r_history),
            N, seq_len, t
        );
    }
}