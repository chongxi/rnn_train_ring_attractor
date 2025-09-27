#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

__global__ void __cluster_dims__(8, 1, 1) fwd_128_kernel(
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
        - After looping through 128 columns, result is atomicAdd to distributed smem[128] in cluster.block_rank() = 0, then go to  the next 128 columns until all columns are reached.

    gridSize = (N / TILE_DIM, BATCH_SIZE)

    Smem per block: (128*3 + 16*3 + 16*128*3 + 16) * 4 /  1024 = 25.75 KB per block
    */
    const float* A, // shape [BATCH_SIZE, 128, 32]
    const float* Wa, // shape [32, 256, 256]
    // const float* J0, // shape [256, 256]
    float J0,
    float J1,
    const float* Wo, // [256, 256]
    float* r_init,
    const float* W_delta7,
    float* bump_history,
    float* r_history,
    int num_neuron,
    int A_DIM,
    int SEQ_LEN
){  
    auto tbc = cg::this_cluster();
    auto cta = cg::this_thread_block();

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
    __shared__ float tileWo[CELL_DIM_MN][CELL_DIM_K];
    __shared__ float tileWd7[CELL_DIM_MN][CELL_DIM_K];
    __shared__ float tileR[CELL_DIM_MN]; 

    int batch_idx = blockIdx.y;
    
    // TODO: check if tbc.block_rank() == 0 make it faster

    for (int tile_idx_mn = 0; tile_idx_mn < num_tile_mn; ++tile_idx_mn){

        // Load r_init to smem
        if (cta.thread_rank() < TILE_DIM){
            r_dsmem_buf[cta.thread_rank()] = r_init[batch_idx * num_neuron + tile_idx_mn * TILE_DIM + cta.thread_rank()];
        }

        for (int tile_idx_k = 0; tile_idx_k < num_tile_k; ++tile_idx_k){

            // Load tileWo and tileWd7 to smem
            for (int inner_k_idx = 0; inner_k_idx < CELL_DIM_K; inner_k_idx+=BK){
                tileWo[threadIdx.y][inner_k_idx + threadIdx.x] = Wo[threadIdx.y][threadIdx.x];

            }

        } // end for tile_idx_k
    } // end for tile_idx_mn

}