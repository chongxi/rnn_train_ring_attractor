#include "../cuda_common.cuh"

template <int block_size, int cluster_size>
__global__ void __cluster_dims__(cluster_size, 1, 1) hardcode_kernel(
    const float* A, // shape [BATCH_SIZE, 128, 32]
    const float* Wa, // shape [32, 256, 256]
    // const float* J0, // shape [256, 256]
    float J0,
    float J1,
    const float* Wo,
    float* r,
    const float* W_delta7,
    float* bump_history,
    float* r_history,
    float alpha,
    int N,
    int a_dim,
    int seq_len
){  
    auto this_cluster = cg::this_cluster();

    __shared__ float r_buf[block_size];
    __shared__ float re_inp_buf[block_size];
    __shared__ float r_d7_buf[block_size];

    int batch_idx = blockIdx.x / cluster_size;
    
    r_buf[threadIdx.x] = r[batch_idx * N + threadIdx.x];

    // constexpr float kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5f;

    int row_idx = threadIdx.x;
    
    for (int t = 0; t < seq_len; ++t){

        re_inp_buf[threadIdx.x] = 0.f;
        r_d7_buf[threadIdx.x] = 0.f;

        // __syncthreads();
        this_cluster.sync();

        // Reduce results to the smem of the first block in a threadblock cluster

        for (int k_idx_local = 0; k_idx_local < N / cluster_size; ++k_idx_local){

            int k_idx = this_cluster.block_rank() * cluster_size + k_idx_local;

            float wa_weighted = 0.f;
            for (int action = 0; action < a_dim; ++action){
                wa_weighted += A[batch_idx * seq_len * a_dim + t * a_dim + action] * Wa[action * N * N + row_idx * N + k_idx];
            }
            
            // float w_eff = J0[row_idx * N + k_idx] + J1 * Wo[row_idx * N + k_idx] + wa_weighted;
            float w_eff = J0 + J1 * Wo[row_idx * N + k_idx] + wa_weighted;

            //Pointer to target block shared memory (first block)
            float* dst_smem = this_cluster.map_shared_rank(re_inp_buf, 0);
            float* src_r_buf = this_cluster.map_shared_rank(r_buf, 0);

            atomicAdd(dst_smem + threadIdx.x, w_eff * src_r_buf[k_idx]);
            // re_inp_buf[threadIdx.x] += w_eff * r_buf[k_idx];
        }

        // this_cluster.sync();

        if (this_cluster.block_rank() == 0) {

            float re_inp_act = re_inp_buf[threadIdx.x];
            // float re_inp_val = 0.5f * re_inp_act * (1.f + tanhf(kBetaVec * fmaf(0.044715f, re_inp_act * re_inp_act * re_inp_act, re_inp_act)));

            float re_inp_val = fmax(re_inp_act, 0.f);
            
            // float alpha = 0.15f;
            float r_updated = (1.f - alpha) * r_buf[threadIdx.x] + alpha * re_inp_val;

            bump_history[batch_idx * seq_len * N + t * N + threadIdx.x] = r_updated;
            r_buf[threadIdx.x] = r_updated;
        }
        this_cluster.sync();

        for (int k_idx_local = 0; k_idx_local < N / cluster_size; ++k_idx_local){
            int k_idx = this_cluster.block_rank() * (N / cluster_size) + k_idx_local;
            
            float* dst_smem = this_cluster.map_shared_rank(r_d7_buf, 0);
            float* src_r_buf = this_cluster.map_shared_rank(r_buf, 0);
            
            atomicAdd(dst_smem + threadIdx.x, src_r_buf[k_idx] * W_delta7[k_idx * N + threadIdx.x]);
        }

        this_cluster.sync();

        if (this_cluster.block_rank() == 0) {

            float r_d7_lane = r_d7_buf[threadIdx.x];

            // Fixed reduction for finding maximum
            for (int offset = block_size / 2; offset > 0; offset /= 2) {
                if (threadIdx.x < offset) {
                    r_d7_buf[threadIdx.x] = fmaxf(r_d7_buf[threadIdx.x], r_d7_buf[threadIdx.x + offset]);
                }
                __syncthreads();
            }
            
            r_history[batch_idx * seq_len * N + t * N + threadIdx.x] = r_d7_lane / r_d7_buf[0];

        }

    } // end for t
}

void hardcode_kernel_launcher(
    void* A,
    void* Wa,
    // void* J0,
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
    constexpr int block_size = 256;
    constexpr int cluster_size = 8;
    dim3 gridSize = batch_size * cluster_size;

    hardcode_kernel<block_size, cluster_size><<<gridSize, block_size>>>(
        static_cast<const float*>(A),
        static_cast<const float*>(Wa),
        // static_cast<const float*>(J0),
        J0,
        J1,
        static_cast<const float*>(Wo),
        static_cast<float*>(r_init),
        static_cast<float *>(W_delta7),
        static_cast<float *>(bump_history),
        static_cast<float *>(r_history),
        alpha,
        N,
        a_dim,
        seq_len       
    ); 
}