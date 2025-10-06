#include "../cuda_common.cuh"

__global__ void fwd_update_r_kernel(
    /*
    Only works with N = 512 and smaller (divisible by 128)
    */
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
    int batch_idx = blockIdx.x;
    int neuron_idx = threadIdx.x;
    
    float r7 = 0.f;
    for (int k = 0; k < n_neur; ++k){
        r7 += r_init[batch_idx * n_neur + k] * W_delta7[neuron_idx * n_neur + k];
    }
    
    __shared__ float r7_vals[512];
    r7_vals[neuron_idx] = r7;
    __syncthreads();
    
    float max_val = r7;
    for (int n = 0; n < n_neur; ++n){
        max_val = fmaxf(max_val, r7_vals[n]);
    }
    
    r_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] = r7 / max_val;
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
    dim3 blockSize1(128);
    dim3 gridSize1(batch_size, (N + 127) / 128);
    
    dim3 blockSize2(N);
    dim3 gridSize2(batch_size);

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
    }
}