#include "../cuda_common.cuh"

template<Activation ACT, int SPLIT, int BLOCKSIZE>
__global__ void fwd_update_r_kernel(
    const float* A,
    const float* Wa,
    float J0,
    float J1,
    const float* Wo,
    const float* r_init,
    float* bump_history,
    float alpha,
    int n_neur,
    int a_dim,
    int seq_len,
    int t
){
    auto cta = cg::this_thread_block();
    auto tile = cg::tiled_partition<BLOCKSIZE>(cta);
    
    int batch_idx = blockIdx.x;
    int base_neuron = blockIdx.y * SPLIT;
    bool is_first = (t == 0);
    
    extern __shared__ float smem[];
    float* s_A = smem;
    
    for (int a = threadIdx.x; a < a_dim; a += blockDim.x){
        s_A[a] = A[batch_idx * seq_len * a_dim + t * a_dim + a];
    }
    __syncthreads();
    
    for (int offset = 0; offset < SPLIT; ++offset){
        int neuron_idx = base_neuron + offset;
        if (neuron_idx >= n_neur) break;
        
        float ri_local = 0.f;
        
        for (int j = tile.thread_rank(); j < n_neur; j += BLOCKSIZE){
            float wa_weighted = 0.f;
            for (int a = 0; a < a_dim; ++a){
                wa_weighted += s_A[a] * Wa[a * n_neur * n_neur + neuron_idx * n_neur + j];
            }
            
            float w_eff = J0 + J1 * Wo[neuron_idx * n_neur + j] + wa_weighted;
            float r_prev_j = is_first ? r_init[batch_idx * n_neur + j] :
                                        bump_history[batch_idx * seq_len * n_neur + (t - 1) * n_neur + j];
            ri_local += w_eff * r_prev_j;
        }
        
        float ri = cg::reduce(tile, ri_local, cg::plus<float>());
        
        if (tile.thread_rank() == 0){
            float ri_activated = activation<ACT>(ri);
            float r_prev_neuron = is_first ? r_init[batch_idx * n_neur + neuron_idx] :
                                             bump_history[batch_idx * seq_len * n_neur + (t - 1) * n_neur + neuron_idx];
            float r = (1.f - alpha) * r_prev_neuron + alpha * ri_activated;
            bump_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] = r;
        }
    }
}

template<Activation ACT>
void fwd_n128_a23_global_launcher_impl(
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
    constexpr int SPLIT = 2;
    constexpr int BLOCKSIZE = 64;
    
    dim3 blockSize(BLOCKSIZE);
    dim3 gridSize(batch_size, (N + SPLIT - 1) / SPLIT);
    size_t smem_size = a_dim * sizeof(float);

    for (int t = 0; t < seq_len; ++t){
        fwd_update_r_kernel<ACT, SPLIT, BLOCKSIZE><<<gridSize, blockSize, smem_size>>>(
            static_cast<const float*>(A),
            static_cast<const float*>(Wa),
            J0, J1,
            static_cast<const float*>(Wo),
            static_cast<const float*>(r_init),
            static_cast<float*>(bump_history),
            alpha, N, a_dim, seq_len, t
        );
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
    int batch_size,
    int activation_type
){
    switch(activation_type){
        case 0: fwd_n128_a23_global_launcher_impl<Activation::RELU>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
        case 1: fwd_n128_a23_global_launcher_impl<Activation::GELU>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
        case 2: fwd_n128_a23_global_launcher_impl<Activation::TANH>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
    }
}