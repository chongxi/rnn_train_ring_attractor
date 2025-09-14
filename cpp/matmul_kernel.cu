#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cuda_fp16.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

namespace cg = cooperative_groups;
using namespace nvcuda;

void matmul_cuda(void* a, void* b, void* c, int M, int N, int K);

__global__ void gemm_tc_4x_kernel(const float* matA, const float* matB, float* matC, int M, int N, int K){
    /*
    split a block of 512 threads into 16 warps. Each block calculates 128x128 elements (32x arithmetic intensity)
    matA MxK row-major
    matB KxN col-major
    matC MxN row-major
    */
    constexpr int K_JUMP = 64;
    // constexpr int NUM_WARP_X = 4;
    // constexpr int NUM_WARP_Y = 4;
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int ld = K_JUMP + 8;

    __shared__ half tileA[BM][ld]; // row-maj to row-maj
    __shared__ half tileB[BN][ld]; // row-maj to col-maj, col-maj = transpose(row-maj)

    int lda = ld;
    int ldb = ld;

    cg::thread_block cta = cg::this_thread_block();
    auto group32 = cg::tiled_partition<32>(cta);
    int t32x = group32.thread_rank(); // 0 -> 31, thread idx in a warp
    int t32y = group32.meta_group_rank(); // 0 -> 15, warp idx in a block

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[2][2];

    #pragma unroll
    for (int i=0; i<2; ++i){
        #pragma unroll
        for (int j=0; j<2; ++j){
            wmma::fill_fragment(frag_acc[i][j], 0.f);
        }
    }
    for (int cta_k = 0; cta_k < K; cta_k += K_JUMP){
        float2 regA, regB;
        __half2 regA_half, regB_half;
        #pragma unroll
        for (int warp_mn = 0; warp_mn < BM; warp_mn += group32.meta_group_size()){

            regA = *(reinterpret_cast<const float2*>(&matA[(blockIdx.y * BM + warp_mn + t32y) * K + cta_k + t32x * 2]));
            regB = *(reinterpret_cast<const float2*>(&matB[cta_k + t32x * 2 + K * (blockIdx.x * BN + warp_mn + t32y)]));  
            regA_half = __float22half2_rn(regA);
            regB_half = __float22half2_rn(regB);

            *(reinterpret_cast<__half2*>(&tileA[warp_mn + t32y][t32x * 2])) = regA_half;
            *(reinterpret_cast<__half2*>(&tileB[warp_mn + t32y][t32x * 2])) = regB_half;      
        } // end for warp_mn   

        __syncthreads();

        int warp_idx_x = t32y % 4;
        int warp_idx_y = t32y / 4;
        // Perform matrix multiplication on shared memory
        #pragma unroll 1
        for (int mma_k = 0; mma_k < K_JUMP; mma_k += WMMA_K){
            for (int i=0; i<2; ++i){
                for (int j=0; j<2; ++j){
                    wmma::load_matrix_sync(frag_a[i], &tileA[0][0] + (warp_idx_y * WMMA_M + i * BM/2) * lda + mma_k, lda);
                    wmma::load_matrix_sync(frag_b[j], &tileB[0][0] + (warp_idx_x * WMMA_N + j * BN/2) * ldb + mma_k, ldb);
                    wmma::mma_sync(frag_acc[i][j], frag_a[i], frag_b[j], frag_acc[i][j]);
                }
            }

        } // end for mma_k

        __syncthreads();
    } // end cta_k

    int warp_idx_x = t32y % 4;
    int warp_idx_y = t32y / 4;

    int col_base_C = blockIdx.x * BN + warp_idx_x * WMMA_N;
    int row_base_C = blockIdx.y * BM + warp_idx_y * WMMA_M;

    #pragma unroll
    for (int i=0; i<2; ++i){
        #pragma unroll
        for (int j=0; j<2; ++j){
            wmma::store_matrix_sync(matC + (row_base_C + i * BM/2) * N + col_base_C + j * BN/2, frag_acc[i][j], N, wmma::mem_row_major);
        }
    }
}

void matmul_cuda(void* a, void* b, void* c, int M, int N, int K){
    int BM, BN; dim3 blockSize, gridSize;

    BM = 128; BN = 128; blockSize = dim3(BM * BN / 32); gridSize = dim3((N+BN-1)/BN, (M+BM-1)/BM);
    gemm_tc_4x_kernel<<<gridSize, blockSize>>>(static_cast<const float *>(a), static_cast<const float *>(b), static_cast<float *>(c), M, N, K);
}