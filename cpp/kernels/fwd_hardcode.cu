#include "../cuda_common.cuh"

__global__ void __cluster_dims__(8, 1, 1) persistent_splitK_tbc_kernel(
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
    int A_DIM,
    int SEQ_LEN
){  
    constexpr int BLOCK_SIZE = 256;
    constexpr int CLUSTER_SIZE = 8;

    auto this_cluster = cg::this_cluster();

    __shared__ float r_buf[BLOCK_SIZE];
    __shared__ float re_inp_buf[BLOCK_SIZE];
    __shared__ float r_d7_buf[BLOCK_SIZE];

    int batch_idx = blockIdx.x / CLUSTER_SIZE;
    
    r_buf[threadIdx.x] = r[batch_idx * N + threadIdx.x];

    // constexpr float kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5f;

    int row_idx = threadIdx.x;
    
    for (int t = 0; t < SEQ_LEN; ++t){

        re_inp_buf[threadIdx.x] = 0.f;
        r_d7_buf[threadIdx.x] = 0.f;

        // __syncthreads();
        this_cluster.sync();

        // Reduce results to the smem of the first block in a threadblock cluster

        for (int k_idx_local = 0; k_idx_local < N / CLUSTER_SIZE; ++k_idx_local){

            int k_idx = this_cluster.block_rank() * CLUSTER_SIZE + k_idx_local;

            float wa_weighted = 0.f;
            for (int action = 0; action < A_DIM; ++action){
                wa_weighted += A[batch_idx * SEQ_LEN * A_DIM + t * A_DIM + action] * Wa[action * N * N + row_idx * N + k_idx];
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

            bump_history[batch_idx * SEQ_LEN * N + t * N + threadIdx.x] = r_updated;
            r_buf[threadIdx.x] = r_updated;
        }
        this_cluster.sync();

        for (int k_idx_local = 0; k_idx_local < N / CLUSTER_SIZE; ++k_idx_local){
            int k_idx = this_cluster.block_rank() * (N / CLUSTER_SIZE) + k_idx_local;
            
            float* dst_smem = this_cluster.map_shared_rank(r_d7_buf, 0);
            float* src_r_buf = this_cluster.map_shared_rank(r_buf, 0);
            
            atomicAdd(dst_smem + threadIdx.x, src_r_buf[k_idx] * W_delta7[k_idx * N + threadIdx.x]);
        }

        this_cluster.sync();

        if (this_cluster.block_rank() == 0) {

            float r_d7_lane = r_d7_buf[threadIdx.x];

            // Fixed reduction for finding maximum
            for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
                if (threadIdx.x < offset) {
                    r_d7_buf[threadIdx.x] = fmaxf(r_d7_buf[threadIdx.x], r_d7_buf[threadIdx.x + offset]);
                }
                __syncthreads();
            }
            
            r_history[batch_idx * SEQ_LEN * N + t * N + threadIdx.x] = r_d7_lane / r_d7_buf[0];

        }

    } // end for t
}