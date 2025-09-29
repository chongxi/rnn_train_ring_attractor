#include "../cuda_common.cuh"

__global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(512) fwd_n128_a23_kernel(
    /*
    Kernel for specific case: n_neur = 128, a_dim = 2 or a_dim = 3

    1 tile of 128x128 (use thread block cluster) is split into 8 cells 16x128 (thread block inside cluster)

    Ex:
    #define BLOCK_SIZE 512
    #define CLUSTER_SIZE 8
    #define TILE_DIM 128

    Inside 1 cluster:
        - Each thread block calculates 16x128 (=smem size) cell, atomicAdd results to a local smem[16]
            - Use 16x32 threads per block to load 16x128, TODO: use vectorized float4 to load
            - Use 16x32 threads per block to calculate 16x128, TODO: use float2 __ffma2_rz, __fmul2_rz, __fadd2_rz
        - After looping through 128 columns, result is atomicAdd local buffer[16], them written to distributed smem[128] in cluster.block_rank() = 0
    gridSize = batch_size * CLUSTER_SIZE;

    Smem per block: (128*4 + 3*16*130 + 2*16*130)* 4 / 1024 = 42.625 KB per block, can only launch 2 blocks (66.67 occupancy for RTX 5090, 100% occupancy for H100 or B200 due to larger SMEM per SM)
    */
    const float* A, // shape [BATCH_SIZE, 128, 2]
    const float* Wa, // shape [2, 256, 256]
    // const float* J0, // shape [256, 256]
    float J0,
    float J1,
    const float* Wo, // [256, 256]
    float* r_init,
    const float* W_delta7_tr, // Transpose W_delta7 to make to col-major, but W_delta7 is created to be symmetric, so no need to
    float* bump_history,
    float* r_history,
    float alpha,
    int n_neur, // num neurons
    int a_dim,
    int seq_len
){  
    auto tbc = cg::this_cluster();
    auto cta = cg::this_thread_block();

    constexpr int TILE_DIM = 128;
    constexpr int CLUSTER_SIZE = 8;
    // constexpr int BMN = 16; // = blockDim.y = 16
    constexpr int BK = 32; // = blockDim.x = 32

    // Tile: 128x128, cell: 16x128
    constexpr int CELL_DIM_MN = TILE_DIM / CLUSTER_SIZE; // 16
    constexpr int CELL_DIM_K = TILE_DIM; // 128


    // use only buffer dsmem stored in the first block in a cluster
    __shared__ float r_dsmem[TILE_DIM];
    __shared__ float ri_dsmem[TILE_DIM]; // recurrent input
    __shared__ float r7_dsmem[TILE_DIM];

    // local buffer smem in each block in a cluster
    __shared__ float r_smem_128[TILE_DIM];

    // // Uncomment in Ver 2: using local smem to atomicAdd
    // __shared__ float ri_smem_16[CELL_DIM_MN];
    // __shared__ float r7_smem_16[CELL_DIM_MN];

    // local smem tiles, constant accross time t
    __shared__ float tileWa[3][CELL_DIM_MN][CELL_DIM_K + 2];
    __shared__ float tileWo[CELL_DIM_MN][CELL_DIM_K + 2];
    __shared__ float tileW7[CELL_DIM_MN][CELL_DIM_K + 2];

    int batch_idx = blockIdx.x / CLUSTER_SIZE;
    
    // TODO: check if tbc.block_rank() == 0 make it faster
    if (cta.thread_rank() < TILE_DIM){
        r_dsmem[cta.thread_rank()] = r_init[batch_idx * n_neur + cta.thread_rank()];
    }

    tbc.sync();
    
    for (int k_jump = 0; k_jump < TILE_DIM; k_jump += BK){
        // Store J0 + J1 * Wo to smem
        tileWo[threadIdx.y][k_jump + threadIdx.x] 
            = J0 + J1 * Wo[(tbc.block_rank() * CELL_DIM_MN + threadIdx.y) * n_neur + k_jump + threadIdx.x];

        tileW7[threadIdx.y][k_jump + threadIdx.x] 
            = W_delta7_tr[(tbc.block_rank() * CELL_DIM_MN + threadIdx.y) * n_neur + k_jump + threadIdx.x];

        for (int a = 0; a < a_dim; ++a){
            tileWa[a][threadIdx.y][k_jump + threadIdx.x] 
                = Wa[a * n_neur * n_neur + (tbc.block_rank() * CELL_DIM_MN + threadIdx.y) * n_neur + k_jump + threadIdx.x];        
        }
    } // end for k_jump

    for (int t = 0; t < seq_len; ++t){

        // // Uncomment in Ver 2: using local smem to atomicAdd
        // if (cta.thread_rank() < CELL_DIM_MN){
        //     ri_smem_16[cta.thread_rank()] = 0.f;
        //     r7_smem_16[cta.thread_rank()] = 0.f;
        // }

        if (cta.thread_rank() < TILE_DIM){
            ri_dsmem[cta.thread_rank()] = 0.f;
            r7_dsmem[cta.thread_rank()] = 0.f;
        }        

        // Broadcast r from block 0 in cluster to others
        if (cta.thread_rank() < TILE_DIM){
            float* src_dsmem = tbc.map_shared_rank(r_dsmem, 0);
            r_smem_128[cta.thread_rank()] = src_dsmem[cta.thread_rank()];
        }   

        tbc.sync();

        float ri_sum_thread = 0.f;
        #pragma unroll
        for (int k_jump = 0; k_jump < TILE_DIM; k_jump += BK){
            float wa_weighted = 0.f;
            #pragma unroll
            for (int a = 0; a < a_dim; ++a){
                wa_weighted += A[batch_idx * seq_len * a_dim + t * a_dim + a] * tileWa[a][threadIdx.y][k_jump + threadIdx.x];
            }
            float w_eff = tileWo[threadIdx.y][k_jump + threadIdx.x] + wa_weighted;
            ri_sum_thread += w_eff * r_smem_128[k_jump + threadIdx.x];
        } // end for k_jump

        // Ver 1: 512 threads atomicAdd straight to ri_dsmem, no need for sync
        float* dst_dsem = tbc.map_shared_rank(ri_dsmem, 0);
        atomicAdd(&dst_dsem[tbc.block_rank() * CELL_DIM_MN + threadIdx.y], ri_sum_thread);

        // tbc.sync();

        // // Ver 2: atomicAdd to local smem first, need sync
        // atomicAdd(&ri_smem_16[cta.thread_rank()], ri_sum_thread);
        // if (cta.thread_rank() < CELL_DIM_MN) {
        //     float* dst_dsem = tbc.map_shared_rank(ri_dsmem, 0);
        //     dst_dsem[tbc.block_rank() * CELL_DIM_MN + cta.thread_rank()] = ri_smem_16[cta.thread_rank()];
        // }
        // tbc.sync();

        if (tbc.block_rank() == 0){
            if (cta.thread_rank() < TILE_DIM){
                float ri_activated = fmax(ri_dsmem[cta.thread_rank()], 0.f);
                float r_updated = (1.f - alpha) * r_dsmem[cta.thread_rank()] + alpha * ri_activated;
                bump_history[batch_idx * seq_len * n_neur + t * n_neur + cta.thread_rank()] = r_updated;

                // update r
                r_dsmem[cta.thread_rank()] = r_updated;
            }
        }

        tbc.sync();

        // Broadcast r from block 0 in cluster to others
        if (cta.thread_rank() < TILE_DIM){
            float* src_dsmem = tbc.map_shared_rank(r_dsmem, 0);
            r_smem_128[cta.thread_rank()] = src_dsmem[cta.thread_rank()];
        } 

        tbc.sync();

        float r7_sum_thread = 0.f;
        #pragma unroll
        for (int k_jump = 0; k_jump < TILE_DIM; k_jump += BK){
            r7_sum_thread += r_smem_128[k_jump + threadIdx.x] * tileW7[threadIdx.y][k_jump + threadIdx.x];
        }

        dst_dsem = tbc.map_shared_rank(r7_dsmem, 0);
        atomicAdd(&dst_dsem[tbc.block_rank() * CELL_DIM_MN + threadIdx.y], r7_sum_thread);

        tbc.sync();

        if (tbc.block_rank() == 0){
            auto tile = cg::tiled_partition<128>(cta);
            if (tile.meta_group_rank() == 0) {
                float r7_lane = r7_dsmem[tile.thread_rank()];
                float max_val = cg::reduce(tile, r7_lane, cg::greater<float>());
                r_history[batch_idx * seq_len * n_neur + t * n_neur + tile.thread_rank()] = r7_lane / max_val;
            }
        }
        
    } // end for t

}