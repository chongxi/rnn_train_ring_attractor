#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

__global__ void __cluster_dims__(8, 1, 1) fwd_n128_a2_kernel(
    /*
    1 threadblock cluster calculates a tile of 128x128 elements
    N must be divisible to 128

    1 tile of 128x128 (use thread block cluster) is split into 8 cells 16x128 (thread block inside cluster)

    Ex:
    #define BLOCK_SIZE 512
    #define CLUSTER_SIZE 8
    #define TILE_DIM 128

    Inside 1 cluster:
        - Each thread block calculates 16x128 (=smem size) cell, atomicAdd results to a local smem[16]
            - Use 16x32 threads per block to load 16x128, TODO: use vectorized float4 to load
            - Use 16x32 threads per block to calculate 16x128, TODO: use float2 __ffma2_rz, __fmul2_rz, __fadd2_rz
        - After looping through 128 columns, result is atomicAdd to distributed smem[128] in cluster.block_rank() = 0.
    gridSize = (N / TILE_DIM, BATCH_SIZE)

    Smem per block: (128*3 + 16*3 + 16*128*3 + 16) * 4 /  1024 = 25.75 KB per block
    */
    const float* A, // shape [BATCH_SIZE, 128, 2]
    const float* Wa, // shape [2, 256, 256]
    // const float* J0, // shape [256, 256]
    float J0,
    float J1,
    const float* Wo, // [256, 256]
    float* r_init,
    const float* W_delta7, // Transpose W_delta7 to make to col-major
    float* bump_history,
    float* r_history,
    float alpha,
    int num_neuron,
    int a_dim,
    int seq_len
){  
    auto tbc = cg::this_cluster();
    auto cta = cg::this_thread_block();

    constexpr int BLOCK_SIZE = 512;
    constexpr int TILE_DIM = 128;
    constexpr int CLUSTER_SIZE = 8;

    int num_tile_mn = num_neuron / TILE_DIM;
    int num_tile_k = num_neuron / TILE_DIM;
    
    // Tile: 128x128, cell: 16x128
    constexpr int CELL_DIM_MN = TILE_DIM / CLUSTER_SIZE; // 16
    constexpr int CELL_DIM_K = TILE_DIM; // 128
    constexpr int NUM_CELL_MN = TILE_DIM / CELL_DIM_MN; // = 8 = CLUSTER_SIZE
    constexpr int NUM_CELL_K = TILE_DIM / CELL_DIM_K; // = 1
    constexpr int BMN = BLOCK_SIZE / 32; // = blockDim.y = 16
    constexpr int BK = BLOCK_SIZE % 32; // = blockDim.x = 32

    // use only buffer dsmem stored in the first block in a cluster
    __shared__ float r_dsmem_buf[TILE_DIM];
    __shared__ float ri_dsmem_buf[TILE_DIM]; // recurrent input
    __shared__ float r7_dsmem_buf[TILE_DIM];

    // local buffer smem in each block in a cluster
    __shared__ float r_smem_buf[CELL_DIM_MN];
    __shared__ float ri_smem_buf[CELL_DIM_MN];
    __shared__ float r7_smem_buf[CELL_DIM_MN];

    // local smem tiles, constant accross time t

    // TODO: use shared memory for tileWa
    // __shared__ float tileWa[CELL_DIM_MN][CELL_DIM_K];

    // __shared__ float tileJ0[CELL_DIM_MN][CELL_DIM_K];

    __shared__ float tileWa[2][CELL_DIM_MN][CELL_DIM_K + 2];

    __shared__ float tileWo[CELL_DIM_MN][CELL_DIM_K + 2];
    __shared__ float tileW7[CELL_DIM_MN][CELL_DIM_K + 2];
    // __shared__ float tileWd7[CELL_DIM_K][CELL_DIM_MN + 2];
    __shared__ float tileR[CELL_DIM_MN]; 

    int batch_idx = blockIdx.y;
    
    // TODO: check if tbc.block_rank() == 0 make it faster
    if (cta.thread_rank() < TILE_DIM){
        r_dsmem_buf[cta.thread_rank()] = r_init[batch_idx * num_neuron + cta.thread_rank()];
    }

    for (int k_jump = 0; k_jump < TILE_DIM; k_jump += BK){

        // Store J0 + J1 * Wo to smem
        tileWo[threadIdx.y][k_jump + threadIdx.x] = J0 + J1 * Wo[(tbc.block_rank() * BMN + threadIdx.y) * BK + k_jump + threadIdx.x];

        tileW7[threadIdx.y][k_jump + threadIdx.x] = W_delta7[(tbc.block_rank() * BMN + threadIdx.y) * BK + k_jump + threadIdx.x];

        tileWa[0][threadIdx.y][k_jump + threadIdx.x] = Wa[(tbc.block_rank() * BMN + threadIdx.y) * BK + k_jump + threadIdx.x];

        tileWa[1][threadIdx.y][k_jump + threadIdx.x] = Wa[num_neuron * num_neuron + (tbc.block_rank() * BMN + threadIdx.y) * BK + k_jump + threadIdx.x];

    } // end for k_jump

    // tbc.sync();

    for (int t = 0; t < seq_len; ++t){

        float2 At;
        At = *(reinterpret_cast<const float2*>(&A[batch_idx * seq_len + t]));

        if (cta.thread_rank() < CELL_DIM_MN){
            ri_smem_buf[cta.thread_rank()] = 0.f;
            r7_smem_buf[cta.thread_rank()] = 0.f;
        }
        
        tbc.sync();

        for (int k_jump = 0; k_jump < TILE_DIM; k_jump += BK){

            
            float wa_weighted1 = At.x * tileWa[0][threadIdx.y][k_jump + threadIdx.x];
            float wa_weighted2 = At.y * tileWa[1][threadIdx.y][k_jump + threadIdx.x];
            float wa_weighted = wa_weighted1 + wa_weighted2;

            float w_eff = tileWo[threadIdx.y][k_jump + threadIdx.x] + wa_weighted;

            atomicAdd(&ri_smem_buf[threadIdx.y], w_eff[threadIdx.y]);
            

        } // end for k_jump



    } // end for t

}