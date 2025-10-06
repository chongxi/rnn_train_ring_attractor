#include "../cuda_common.cuh"

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

template<Activation ACT>
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
    int a_dim,
    int seq_len
){
    constexpr int n_neur = 128;
    auto cta = cg::this_thread_block();
    auto tile = cg::tiled_partition<n_neur>(cta);
    
    int batch_idx = blockIdx.x;
    int neuron_idx = tile.thread_rank();  // Each thread handles one neuron
    
    __shared__ float r_shared[n_neur];
    
    float r = r_init[batch_idx * n_neur + neuron_idx];

    for (int t = 0; t < seq_len; ++t){
        // Load current r values into shared memory
        r_shared[neuron_idx] = r_init[batch_idx * n_neur + neuron_idx];
        cta.sync();
        
        // Compute ri[neuron_idx] = sum_j W_eff[neuron_idx, j] * r[j]
        float ri = 0.f;
        for (int j = 0; j < n_neur; ++j){
            // Compute wa_weighted for W_eff[neuron_idx, j]
            float wa_weighted = 0.f;
            for (int a = 0; a < a_dim; ++a){
                wa_weighted += A[batch_idx * seq_len * a_dim + t * a_dim + a] 
                             * Wa[a * n_neur * n_neur + neuron_idx * n_neur + j];
            }
            
            float w_eff = J0 + J1 * Wo[neuron_idx * n_neur + j] + wa_weighted;
            ri += w_eff * r_shared[j];
        }
        
        float ri_activated = activation<ACT>(ri);
        r = (1.f - alpha) * r + alpha * ri_activated;
        
        bump_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] = r;
        r_init[batch_idx * n_neur + neuron_idx] = r;
        
        cta.sync();
        
        // Compute r7[neuron_idx] = sum_i r[i] * W_delta7[neuron_idx, i]
        float r7 = 0.f;
        for (int i = 0; i < n_neur; ++i){
            r7 += r_init[batch_idx * n_neur + i] * W_delta7[neuron_idx * n_neur + i];
        }
        
        // Find max across all threads
        float max_val = cg::reduce(tile, r7, cg::greater<float>());
        
        r_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] = r7 / max_val;
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
    int a_dim,
    int seq_len,
    int batch_size
){
    constexpr int n_neur = 128;
    dim3 blockSize(n_neur);
    dim3 gridSize(batch_size, 1);

    fwd_n128_a23_global_kernel<ACT><<<gridSize, blockSize>>>(
        static_cast<const float*>(A),
        static_cast<const float*>(Wa),
        J0, J1,
        static_cast<const float*>(Wo),
        static_cast<float*>(r_init),
        static_cast<const float*>(W_delta7),
        static_cast<float*>(bump_history),
        static_cast<float*>(r_history),
        alpha, a_dim, seq_len
    );
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
        case 0: fwd_n128_a23_global_launcher_impl<Activation::RELU>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, a_dim, seq_len, batch_size); break;
        case 1: fwd_n128_a23_global_launcher_impl<Activation::GELU>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, a_dim, seq_len, batch_size); break;
        case 2: fwd_n128_a23_global_launcher_impl<Activation::TANH>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, a_dim, seq_len, batch_size); break;
    }
}