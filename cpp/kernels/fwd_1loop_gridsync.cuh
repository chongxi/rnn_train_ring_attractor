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

template<Activation ACT, int BLOCKSIZE>
__global__ void fwd_persistent_kernel(
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
    int batch_size
){
    auto cta = cg::this_thread_block();
    auto tile = cg::tiled_partition<BLOCKSIZE>(cta);
    auto grid = cg::this_grid();
    
    int total_neurons = batch_size * n_neur;
    
    extern __shared__ float smem[];
    
    for (int t = 0; t < seq_len; ++t) {
        bool is_first = (t == 0);
        
        // Each block processes multiple neurons
        for (int global_neuron = blockIdx.x; global_neuron < total_neurons; global_neuron += gridDim.x) {
            int batch_idx = global_neuron / n_neur;
            int neuron_idx = global_neuron % n_neur;
            
            // Load A for this batch/timestep (reuse across neuron iterations)
            float* s_A = smem + threadIdx.x / BLOCKSIZE * a_dim;
            for (int a = threadIdx.x % BLOCKSIZE; a < a_dim; a += BLOCKSIZE) {
                s_A[a] = A[batch_idx * seq_len * a_dim + t * a_dim + a];
            }
            __syncthreads();
            
            float ri_local = 0.f;
            for (int j = tile.thread_rank(); j < n_neur; j += BLOCKSIZE) {
                float wa_weighted = 0.f;
                for (int a = 0; a < a_dim; ++a) {
                    wa_weighted += s_A[a] * Wa[a * n_neur * n_neur + neuron_idx * n_neur + j];
                }
                
                float w_eff = J0 + J1 * Wo[neuron_idx * n_neur + j] + wa_weighted;
                float r_prev_j = is_first ? r_init[batch_idx * n_neur + j] :
                                            bump_history[batch_idx * seq_len * n_neur + (t - 1) * n_neur + j];
                ri_local += w_eff * r_prev_j;
            }
            
            float ri = cg::reduce(tile, ri_local, cg::plus<float>());
            
            if (tile.thread_rank() == 0) {
                float ri_activated = activation<ACT>(ri);
                float r_prev = is_first ? r_init[batch_idx * n_neur + neuron_idx] :
                                         bump_history[batch_idx * seq_len * n_neur + (t - 1) * n_neur + neuron_idx];
                float r = (1.f - alpha) * r_prev + alpha * ri_activated;
                bump_history[batch_idx * seq_len * n_neur + t * n_neur + neuron_idx] = r;
            }
        }
        
        grid.sync();
    }
}

template<Activation ACT>
void fwd_n128_a23_global_launcher_impl(
    void* A, void* Wa, float J0, float J1, void* Wo, void* r_init,
    void* W_delta7, void* bump_history, void* r_history, float alpha,
    int N, int a_dim, int seq_len, int batch_size
){
    constexpr int BLOCKSIZE = 64;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int numBlocksPerSm;
    size_t smem_size = a_dim * sizeof(float);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm,
        fwd_persistent_kernel<ACT, BLOCKSIZE>,
        BLOCKSIZE, smem_size
    );
    
    int numBlocks = numBlocksPerSm * prop.multiProcessorCount;
    dim3 blockSize(BLOCKSIZE);
    dim3 gridSize(numBlocks);
    
    void* args[] = {&A, &Wa, &J0, &J1, &Wo, &r_init, &bump_history,
                   &alpha, &N, &a_dim, &seq_len, &batch_size};
    
    cudaLaunchCooperativeKernel(
        (void*)fwd_persistent_kernel<ACT, BLOCKSIZE>,
        gridSize, blockSize, args, smem_size
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
        case 0: fwd_n128_a23_global_launcher_impl<Activation::RELU>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
        case 1: fwd_n128_a23_global_launcher_impl<Activation::GELU>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
        case 2: fwd_n128_a23_global_launcher_impl<Activation::TANH>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
    }
}