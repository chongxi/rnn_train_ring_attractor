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
__global__ void fwd_update_r_kernel(
    const float* A,
    const float* Wa,
    float J0,
    float J1,
    const float* Wo,
    const float* r_in,
    float* r_out,
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
    int neuron_idx = blockIdx.y;
    
    if (neuron_idx >= n_neur) return;
    
    float ri_local = 0.f;
    
    for (int j = tile.thread_rank(); j < n_neur; j += 128){
        float wa_weighted = 0.f;
        for (int a = 0; a < a_dim; ++a){
            wa_weighted += A[batch_idx * seq_len * a_dim + t * a_dim + a] 
                         * Wa[a * n_neur * n_neur + neuron_idx * n_neur + j];
        }
        
        float w_eff = J0 + J1 * Wo[neuron_idx * n_neur + j] + wa_weighted;
        ri_local += w_eff * r_in[batch_idx * n_neur + j];
    }
    
    float ri = cg::reduce(tile, ri_local, cg::plus<float>());
    
    if (tile.thread_rank() == 0){
        float ri_activated = activation<ACT>(ri);
        float r = (1.f - alpha) * r_in[batch_idx * n_neur + neuron_idx] + alpha * ri_activated;
        
        bump_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] = r;
        r_out[batch_idx * n_neur + neuron_idx] = r;
    }
}

__global__ void fwd_compute_r7_kernel(
    const float* bump_history,
    const float* W_delta7,
    float* r_history,
    int n_neur,
    int seq_len,
    int t
){
    auto cta = cg::this_thread_block();
    auto tile = cg::tiled_partition<128>(cta);
    
    int batch_idx = blockIdx.x;
    int block_id_y = blockIdx.y;
    
    for (int neuron_idx = block_id_y; neuron_idx < n_neur; neuron_idx += 8){
        float r7_local = 0.f;
        
        for (int k = tile.thread_rank(); k < n_neur; k += 128){
            r7_local += bump_history[batch_idx * seq_len * n_neur + t * n_neur + k] 
                      * W_delta7[neuron_idx * n_neur + k];
        }
        
        float r7 = cg::reduce(tile, r7_local, cg::plus<float>());
        
        if (tile.thread_rank() == 0){
            r_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] = r7;
        }
    }
}

__global__ void fwd_normalize_r7_kernel(
    float* r_history,
    int n_neur,
    int seq_len,
    int t
){
    __shared__ float shared_max;
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int base = batch_idx * seq_len * n_neur + t * n_neur;
    
    if (tid == 0) shared_max = -INFINITY;
    __syncthreads();
    
    // Find max
    float local_max = -INFINITY;
    for (int i = tid; i < n_neur; i += 128){
        local_max = fmaxf(local_max, r_history[base + i]);
    }
    atomicMax((int*)&shared_max, __float_as_int(local_max));
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < n_neur; i += 128){
        r_history[base + i] /= shared_max;
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
    dim3 blockSize1(128);
    dim3 gridSize1(batch_size, N);
    
    dim3 blockSize2(128);
    dim3 gridSize2(batch_size, 8);
    
    dim3 blockSize3(128);
    dim3 gridSize3(batch_size);
    
    int num_blocks_y = (N + 127) / 128;
    size_t smem_size = num_blocks_y * sizeof(float);
    
    float* r_temp;
    cudaMalloc(&r_temp, batch_size * N * sizeof(float));

    for (int t = 0; t < seq_len; ++t){
        fwd_update_r_kernel<ACT><<<gridSize1, blockSize1>>>(
            static_cast<const float*>(A),
            static_cast<const float*>(Wa),
            J0, J1,
            static_cast<const float*>(Wo),
            static_cast<const float*>(r_init),
            r_temp,
            static_cast<float*>(bump_history),
            alpha, N, a_dim, seq_len, t
        );
        cudaMemcpy(r_init, r_temp, batch_size * N * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(r_temp);
    
    for (int t = 0; t < seq_len; ++t){
        fwd_compute_r7_kernel<<<gridSize2, blockSize2>>>(
            static_cast<const float*>(bump_history),
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