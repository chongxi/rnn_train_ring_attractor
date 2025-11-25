// #include "cuda_common.cuh"
#include "../cuda_common.cuh"
#include <cuda_bf16.h>

namespace cuda_wmma_re {
    template<int WMMA_M, int WMMA_N, int WMMA_K>
    __device__ void calc_Weff(wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& frag_Wa_weighted,
        const float* Wo, float J0, float J1)
    {
        /*
        *Calc Weff using columnwise multiplication and addition, require using thread/register - data mapping
        *16x16 FragC Register Layout (4 quadrants, each with 2 registers)
        *Only support m16n16k16 and m32n8k16.

             +---------+---------+
             | r0  r1  | r4  r5  |
        0-7  |         |         |
             | Block 0 | Block 1 |
             +---------+---------+
             | r2  r3  | r6  r7  |
        8-15 |         |         |
             | Block 2 | Block 3 |
             +---------+---------+
               0-7       8-15
        */

        constexpr int NUM_REGBLOCK_ROWS = WMMA_M / 8;
        constexpr int NUM_REGBLOCK_COLS = WMMA_N / 8;
        constexpr int REGS_PER_BLOCK = 2;

        size_t threadID_in_warp = threadIdx.x % WARPSIZE;
        // size_t groupID_in_warp = threadID_in_warp / 4;
        size_t threadID_in_group = threadID_in_warp % 4;

        for (int regBlockRow = 0; regBlockRow < NUM_REGBLOCK_ROWS; ++regBlockRow) {
            for (int regBlockCol = 0; regBlockCol < NUM_REGBLOCK_COLS; ++regBlockCol) {
                for (int i = 0; i < REGS_PER_BLOCK; ++i) {
                    int regID = (regBlockRow * REGS_PER_BLOCK) +
                           (regBlockCol * NUM_REGBLOCK_ROWS * REGS_PER_BLOCK) + i;
                    // size_t rowData = regBlockRow * 8 + groupID_in_warp;
                    size_t colData = regBlockCol * 8 + threadID_in_group * 2 + i;

                    frag_Wa_weighted.x[regID] = frag_Wa_weighted.x[regID] = fmaf(J1, Wo[colData], J0 + frag_Wa_weighted.x[regID]);
                    // frag_Wa_weighted.x[regID] = J1 * Wo[colData] + J0 + frag_Wa_weighted.x[regID];
                }
            }
        }
    }


    // ---------------------------------------------------------------
    // Kernel 1 – recompute W_eff only
    // ---------------------------------------------------------------
    template<int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
    __global__ void __launch_bounds__((BN / WMMA_N) * WARPSIZE * (BM / WMMA_M))
    recompute_Weff_kernel(
        const float* A,               // [B, S, A_dim]
        const float* Wa,              // [A_dim, N, N]
        float J0,
        float J1,
        const float* Wo,              // [N, N]
        float*       W_eff_out,       // [B, N, N]
        __nv_bfloat16* A_t_temp,      // [B, A_dim] - BF16
        int          N,
        int          a_dim,
        int          seq_len,
        int          t,               // current time‑step (0‑based)
        int          batch_size)
    {
        // -----------------------------------------------------------------
        // Index wrappers
        // -----------------------------------------------------------------
        IndexWrapper<const float, 3> A_idx (A,  batch_size, seq_len, a_dim);
        IndexWrapper<const float, 3> Wa_idx(Wa, a_dim, N, N);
        IndexWrapper<const float, 2> Wo_idx(Wo, N, N);
        IndexWrapper<float, 2> W_eff_out_idx(W_eff_out, batch_size, N * N);

        auto cta = cg::this_thread_block();

        const size_t warp_idx_x = threadIdx.x / WARPSIZE;
        const size_t warp_idx_y = threadIdx.y;

        const int global_m_base = blockIdx.y * BM;
        const int global_Non = blockIdx.x;

        if (global_m_base >= batch_size || global_Non >= N) return;

        constexpr int ld = BK + 8;
        constexpr int lda = ld;
        constexpr int ldb = ld;

        __shared__ nv_bfloat16 tileA[BM][ld];
        __shared__ nv_bfloat16 tileB[BN][ld];


        // if (a_dim == 16) {
        //     int cta_k = 0;
        //     const int total_threads = blockDim.x * blockDim.y;
        //     const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        //     const int num_loads = (BM * BK) / 4;
        //
        //     for (int idx = thread_id; idx < num_loads; idx += total_threads) {
        //         int load_m = idx / (BK / 4);
        //         int load_k_base = (idx % (BK / 4)) * 4;
        //
        //         int global_m = global_m_base + load_m;
        //         int global_k = cta_k + load_k_base;
        //
        //         if (global_m < batch_size && global_k < a_dim) {
        //             const float* src_ptr = &A_idx.at(global_m, t, global_k);
        //             float4 data = *reinterpret_cast<const float4*>(src_ptr);
        //
        //             nv_bfloat162 bf16_01 = __float22bfloat162_rn(make_float2(data.x, data.y));
        //             nv_bfloat162 bf16_23 = __float22bfloat162_rn(make_float2(data.z, data.w));
        //
        //             *reinterpret_cast<nv_bfloat162*>(&tileA[load_m][load_k_base]) = bf16_01;
        //             *reinterpret_cast<nv_bfloat162*>(&tileA[load_m][load_k_base + 2]) = bf16_23;
        //
        //             *reinterpret_cast<nv_bfloat162*>(&A_t_temp[global_m * a_dim + global_k]) = bf16_01;
        //             *reinterpret_cast<nv_bfloat162*>(&A_t_temp[global_m * a_dim + global_k + 2]) = bf16_23;
        //         } else {
        //             nv_bfloat162 zeros = __float22bfloat162_rn(make_float2(0.f, 0.f));
        //             *reinterpret_cast<nv_bfloat162*>(&tileA[load_m][load_k_base]) = zeros;
        //             *reinterpret_cast<nv_bfloat162*>(&tileA[load_m][load_k_base + 2]) = zeros;
        //         }
        //     }
        //     __syncthreads();
        // }


        for (int cta_n = 0; cta_n < N; cta_n += BN) {

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_Wa_weighted;
            wmma::fill_fragment(frag_Wa_weighted, 0.f);

            for (int cta_k = 0; cta_k < a_dim; cta_k += BK) {

                //===================== Load A into tileA and A_t_temp ===========================================
                // if (a_dim > 16)
                {
                    const int total_threads = blockDim.x * blockDim.y;
                    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
                    const int num_loads = (BM * BK) / 4;

                    for (int idx = thread_id; idx < num_loads; idx += total_threads) {
                        int load_m = idx / (BK / 4);
                        int load_k_base = (idx % (BK / 4)) * 4;

                        int global_m = global_m_base + load_m;
                        int global_k = cta_k + load_k_base;

                        if (global_m < batch_size && global_k < a_dim) {
                            const float* src_ptr = &A_idx.at(global_m, t, global_k);
                            float4 data = *reinterpret_cast<const float4*>(src_ptr);

                            nv_bfloat162 bf16_01 = __float22bfloat162_rn(make_float2(data.x, data.y));
                            nv_bfloat162 bf16_23 = __float22bfloat162_rn(make_float2(data.z, data.w));

                            *reinterpret_cast<nv_bfloat162*>(&tileA[load_m][load_k_base]) = bf16_01;
                            *reinterpret_cast<nv_bfloat162*>(&tileA[load_m][load_k_base + 2]) = bf16_23;

                            *reinterpret_cast<nv_bfloat162*>(&A_t_temp[global_m * a_dim + global_k]) = bf16_01;
                            *reinterpret_cast<nv_bfloat162*>(&A_t_temp[global_m * a_dim + global_k + 2]) = bf16_23;
                        } else {
                            nv_bfloat162 zeros = __float22bfloat162_rn(make_float2(0.f, 0.f));
                            *reinterpret_cast<nv_bfloat162*>(&tileA[load_m][load_k_base]) = zeros;
                            *reinterpret_cast<nv_bfloat162*>(&tileA[load_m][load_k_base + 2]) = zeros;
                        }
                    }
                }

                {   // Transpose load into tileB
                    const int total_threads = blockDim.x * blockDim.y;
                    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
                    const int num_loads = (BK * BN) / 2;

                    for (int idx = thread_id; idx < num_loads; idx += total_threads) {
                        int load_k = idx / (BN / 2);
                        int load_n_base = (idx % (BN / 2)) * 2;

                        int global_k = cta_k + load_k;
                        int global_n = cta_n + load_n_base;

                        if (global_k < a_dim && global_n + 1 < N) {
                            const float* src_ptr = &Wa_idx.at(global_k, global_Non, global_n);
                            float2 data = *reinterpret_cast<const float2*>(src_ptr);

                            tileB[load_n_base][load_k] = __float2bfloat16(data.x);
                            tileB[load_n_base + 1][load_k] = __float2bfloat16(data.y);
                        } else {
                            tileB[load_n_base][load_k] = __float2bfloat16(0.f);
                            tileB[load_n_base + 1][load_k] = __float2bfloat16(0.f);
                        }
                    }
                }

                __syncthreads();

                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, nv_bfloat16, wmma::row_major> frag_a;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, nv_bfloat16, wmma::col_major> frag_b;

    #pragma unroll
                for (int mma_k = 0; mma_k < BK; mma_k += WMMA_K) {
                    wmma::load_matrix_sync(frag_a, &tileA[0][0] + (warp_idx_y * WMMA_M) * lda + mma_k, lda);
                    wmma::load_matrix_sync(frag_b, &tileB[0][0] + (warp_idx_x * WMMA_N) * ldb + mma_k, ldb);
                    wmma::mma_sync(frag_Wa_weighted, frag_a, frag_b, frag_Wa_weighted);
                }

                __syncthreads();
            } // end for cta_k

            //============================ Apply epilogue to get W_eff[BM, BN] =====================================
            int frag_n = cta_n + warp_idx_x * WMMA_N;
            if (frag_n < N) {
                const float* pos_Wo = &Wo_idx.at(global_Non, frag_n);
                calc_Weff<WMMA_M, WMMA_N, WMMA_K>(frag_Wa_weighted, pos_Wo, J0, J1);
                int frag_m = global_m_base + warp_idx_y * WMMA_M;
                if (frag_m < batch_size) {
                    int offset = global_Non * N + frag_n;
                    wmma::store_matrix_sync(&W_eff_out_idx.at(frag_m, offset), frag_Wa_weighted, N * N, wmma::mem_row_major);
                }
            }
        } // end for cta_n
    }


    // ---------------------------------------------------------------
    // Kernel 2 – Compute grad_re and update grad_r
    // ---------------------------------------------------------------
    template<Activation ACT>
    __global__ void compute_grad_re_and_update_grad_r_kernel(
        const float* W_eff,         // [B, N, N]
        const float* re_ctx_t,      // [B, N]
        const float* grad_output_t, // [B, N]
        float* grad_r,              // [B, N] - in/out
        float* grad_re_out,         // [B, N] - output
        float alpha,
        int N,
        int batch_size)
    {
        int b = blockIdx.x;
        int i = blockIdx.y * blockDim.x + threadIdx.x;

        if (b >= batch_size || i >= N) return;

        int idx = b * N + i;

        // Step 1: Accumulate gradient from output
        float grad_r_current = grad_output_t[idx] + grad_r[idx];

        // Step 2: Compute grad_re
        float re_val = re_ctx_t[idx];
        float dphi = activation_derivative<ACT>(re_val);
        float grad_re = dphi * grad_r_current * alpha;

        // Store grad_re for weight gradient computation
        grad_re_out[idx] = grad_re;

        // Step 3: Update grad_r for next iteration (t-1)
        // grad_r_next = W_eff^T @ grad_re + grad_r_current * (1 - alpha)
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float w = W_eff[b * N * N + j * N + i];  // W_eff^T
            sum += w * grad_re;
        }

        grad_r[idx] = sum + grad_r_current * (1.0f - alpha);
    }

    // Kernel 2A: Compute grad_re only
    template<Activation ACT>
    __global__ void compute_grad_re_kernel(
        const float* re_ctx_t,      // [B, N]
        const float* grad_output_t, // [B, N]
        const float* grad_r,        // [B, N] - input
        float* grad_re_out,         // [B, N] - output
        float alpha,
        int N,
        int batch_size)
    {
        int b = blockIdx.x;
        int i = blockIdx.y * blockDim.x + threadIdx.x;

        if (b >= batch_size || i >= N) return;

        int idx = b * N + i;

        // Accumulate gradient from output
        float grad_r_current = grad_output_t[idx] + grad_r[idx];

        // Compute grad_re
        float re_val = re_ctx_t[idx];
        float dphi = activation_derivative<ACT>(re_val);
        float grad_re = dphi * grad_r_current * alpha;

        // Store grad_re
        grad_re_out[idx] = grad_re;
    }

    // Kernel 2B: Update grad_r using grad_re
    __global__ void update_grad_r_kernel(
        const float* W_eff,       // [B, N, N]
        const float* grad_re,     // [B, N]
        const float* grad_output_t, // [B, N]
        float* grad_r,            // [B, N] - in/out
        float alpha,
        int N,
        int batch_size)
    {
        int b = blockIdx.x;
        int i = blockIdx.y * blockDim.x + threadIdx.x;

        if (b >= batch_size || i >= N) return;

        int idx = b * N + i;

        float grad_r_current = grad_output_t[idx] + grad_r[idx];

        // Update grad_r for next iteration (t-1)
        // grad_r_next = W_eff^T @ grad_re + grad_r_current * (1 - alpha)
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float w = W_eff[b * N * N + j * N + i];  // W_eff^T
            sum += w * grad_re[b * N + j];
        }

        grad_r[idx] = sum + grad_r_current * (1.0f - alpha);
    }

    // ---------------------------------------------------------------
    // Kernel 4 – WMMA-based grad_Wa computation
    // grad_Wa[a_dim, N*N] += A_t^T[a_dim, B] @ grad_W_eff[B, N*N]
    // ---------------------------------------------------------------
    template<int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int WARP_TILING_X, int WARP_TILING_Y>
    __global__ void __launch_bounds__((BN / WMMA_N / WARP_TILING_Y) * WARPSIZE * (BM / WMMA_M / WARP_TILING_X))
    gemm_kernel(
        const __nv_bfloat16* A,
        const __nv_bfloat16* B,
        float* C,
        int M, int N, int K
    ){
        IndexWrapper<const __nv_bfloat16, 2> A_idx(A, K, M);  // [K, M] col-major
        IndexWrapper<const __nv_bfloat16, 2> B_idx(B, K, N);  // [K, N] col-major
        IndexWrapper<     float, 2> C_idx(C, M, N);  // [M, N] row-major

        constexpr int padd = 8;
        constexpr int ldsA = BM + padd;  // Padding on M dimension
        constexpr int ldsB = BN + padd;  // Padding on N dimension
        __shared__ __nv_bfloat16 sA[BK][ldsA]; // [BK][BM+8]
        __shared__ __nv_bfloat16 sB[BK][ldsB]; // [BK][BN+8]

        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int numThreads = blockDim.x * blockDim.y;

        // 2D array of accumulator fragments
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[WARP_TILING_X][WARP_TILING_Y];

        // Initialize all accumulators to zero
        #pragma unroll
        for (int ti = 0; ti < WARP_TILING_X; ti++) {
            #pragma unroll
            for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                wmma::fill_fragment(frag_acc[ti][tj], 0.f);
            }
        }

        size_t warpIdx = threadIdx.x / WARPSIZE;
        size_t warpIdy = threadIdx.y;

        for (int ctak = 0; ctak < K; ctak += BK) {
            // Load sA: A is [K, M] col-major, we want [BK][BM+8] in shared memory
            constexpr int VEC_SIZE = 8;
            {
                int numLoads = (BK * BM) / VEC_SIZE;
                for (int i = tid; i < numLoads; i += numThreads) {
                    int k = i / (BM / VEC_SIZE);
                    int m = (i % (BM / VEC_SIZE)) * VEC_SIZE;
                    int globalK = ctak + k;
                    int globalM = blockIdx.y * BM + m;

                    if (globalK < K && globalM < M) {
                        // Load from A[globalK, globalM] which is col-major
                        float4 tmp = *reinterpret_cast<const float4*>(&A_idx.at(globalK, globalM));
                        *reinterpret_cast<float4*>(&sA[k][m]) = tmp;
                    } else {
                        for (int j = 0; j < VEC_SIZE; j++) {
                            sA[k][m + j] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }

            // Load sB: B is [K, N] col-major, we want [BK][BN+8] in shared memory
            {
                int numLoads = (BK * BN) / VEC_SIZE;
                for (int i = tid; i < numLoads; i += numThreads) {
                    int k = i / (BN / VEC_SIZE);
                    int n = (i % (BN / VEC_SIZE)) * VEC_SIZE;
                    int globalK = ctak + k;
                    int globalN = blockIdx.x * BN + n;

                    if (globalK < K && globalN < N) {
                        // Load from B[globalK, globalN] which is col-major
                        float4 tmp = *reinterpret_cast<const float4*>(&B_idx.at(globalK, globalN));
                        *reinterpret_cast<float4*>(&sB[k][n]) = tmp;
                    } else {
                        for (int j = 0; j < VEC_SIZE; j++) {
                            sB[k][n + j] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }

            __syncthreads();

            // Fragment arrays for A and B tiles
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> frag_a[WARP_TILING_X];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_b[WARP_TILING_Y];

            // MMA computation with 2D tiling
            for (int k = 0; k < BK; k += WMMA_K) {
                // Load A fragments for this warp's tiles (col-major)
                #pragma unroll
                for (int ti = 0; ti < WARP_TILING_X; ti++) {
                    int sA_col = warpIdy * WMMA_M * WARP_TILING_X + ti * WMMA_M;
                    wmma::load_matrix_sync(frag_a[ti], &sA[k][sA_col], ldsA);
                }

                // Load B fragments for this warp's tiles (row-major)
                #pragma unroll
                for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                    int sB_col = warpIdx * WMMA_N * WARP_TILING_Y + tj * WMMA_N;
                    wmma::load_matrix_sync(frag_b[tj], &sB[k][sB_col], ldsB);
                }

                // Compute all tile combinations
                #pragma unroll
                for (int ti = 0; ti < WARP_TILING_X; ti++) {
                    #pragma unroll
                    for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                        wmma::mma_sync(frag_acc[ti][tj], frag_a[ti], frag_b[tj], frag_acc[ti][tj]);
                    }
                }
            }

            __syncthreads();
        }

        // Store results with bounds checking
        #pragma unroll
        for (int ti = 0; ti < WARP_TILING_X; ti++) {
            #pragma unroll
            for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                int mAg = blockIdx.y * BM + warpIdy * WMMA_M * WARP_TILING_X + ti * WMMA_M;
                int nBg = blockIdx.x * BN + warpIdx * WMMA_N * WARP_TILING_Y + tj * WMMA_N;

                if (mAg < M && nBg < N) {
                    // Load existing values and accumulate
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> existing_frag;
                    float* C_ptr = &C_idx.at(mAg, nBg);
                    wmma::load_matrix_sync(existing_frag, C_ptr, N, wmma::mem_row_major);

                    // Accumulate
                    for (int i = 0; i < frag_acc[ti][tj].num_elements; ++i) {
                        frag_acc[ti][tj].x[i] += existing_frag.x[i];
                    }

                    // Store back
                    wmma::store_matrix_sync(C_ptr, frag_acc[ti][tj], N, wmma::mem_row_major);
                }
            }
        }
    }


    // ---------------------------------------------------------------
    // Kernel 4 – WMMA-based grad_Wa computation
    // grad_Wa[a_dim, N*N] += A_t^T[a_dim, B] @ grad_W_eff[B, N*N]
    // ---------------------------------------------------------------
    template<int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int WARP_TILING_X, int WARP_TILING_Y>
    __global__ void __launch_bounds__((BN / WMMA_N / WARP_TILING_Y) * WARPSIZE * (BM / WMMA_M / WARP_TILING_X))
    grad_W_eff_gemm_kernel(
        const __nv_bfloat16* A, // [M, K]
        const float* grad_re, // [B, N]
        const float* r_prev,  // [B, N]
        const __nv_bfloat16* B, // [B, NN]
        float* C, // [M, NN] with N^2 == NN
        int M, int NN, int K, int N
    ){
        IndexWrapper<const __nv_bfloat16, 2> A_idx(A, K, M);  // [K, M] col-major
        IndexWrapper<const __nv_bfloat16, 2> B_idx(B, K, NN);  // [K, N] col-major
        IndexWrapper<              float, 2> C_idx(C, M, NN);  // [M, N] row-major
        IndexWrapper<        const float, 2> grad_re_T_idx(grad_re, N, K);
        IndexWrapper<        const float, 2> r_prev_idx(r_prev, K, N);

        constexpr int padd = 8;
        constexpr int ldsA = BM + padd;
        constexpr int ldsB = BN + padd;
        __shared__ __nv_bfloat16 sA[BK][ldsA];
        __shared__ __nv_bfloat16 sB[BK][ldsB];

        // Dynamically allocate shared memory for entire grad_re row
        extern __shared__ float sGrad_re[];  // Size: K (batch_size) floats

        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int numThreads = blockDim.x * blockDim.y;

        // Determine which row of grad_re we need (the 'i' in the N×N matrix)
        int globalNN_start = blockIdx.x * BN;
        int globalNNrow = globalNN_start / N;

        // Load entire grad_re row for all batches once (OUTSIDE the ctak loop)
        {
            constexpr int GRAD_VEC_SIZE = 4;
            int numLoads = (K + GRAD_VEC_SIZE - 1) / GRAD_VEC_SIZE;

            for (int i = tid; i < numLoads; i += numThreads) {
                int k = i * GRAD_VEC_SIZE;

                if (k + GRAD_VEC_SIZE - 1 < K) {
                    // Vectorized load from grad_re_T[globalNNrow, k]
                    float4 tmp = *reinterpret_cast<const float4*>(&grad_re_T_idx.at(globalNNrow, k));
                    *reinterpret_cast<float4*>(&sGrad_re[k]) = tmp;
                } else {
                    // Scalar fallback for boundary
                    for (int j = 0; j < GRAD_VEC_SIZE; j++) {
                        if (k + j < K) {
                            sGrad_re[k + j] = grad_re_T_idx.at(globalNNrow, k + j);
                        }
                    }
                }
            }
        }

        __syncthreads();

        // 2D array of accumulator fragments
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[WARP_TILING_X][WARP_TILING_Y];

        // Initialize all accumulators to zero
        // #pragma unroll
        for (int ti = 0; ti < WARP_TILING_X; ti++) {
            // #pragma unroll
            for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                wmma::fill_fragment(frag_acc[ti][tj], 0.f);
            }
        }

        size_t warpIdx = threadIdx.x / WARPSIZE;
        size_t warpIdy = threadIdx.y;

        for (int ctak = 0; ctak < K; ctak += BK) {
            // Load sA
            {
                constexpr int VEC_SIZE = 8;
                int numLoads = (BK * BM) / VEC_SIZE;
                for (int i = tid; i < numLoads; i += numThreads) {
                    int k = i / (BM / VEC_SIZE);
                    int m = (i % (BM / VEC_SIZE)) * VEC_SIZE;
                    int globalK = ctak + k;
                    int globalM = blockIdx.y * BM + m;

                    if (globalK < K && globalM < M) {
                        float4 tmp = *reinterpret_cast<const float4*>(&A_idx.at(globalK, globalM));
                        *reinterpret_cast<float4*>(&sA[k][m]) = tmp;
                    } else {
                        for (int j = 0; j < VEC_SIZE; j++) {
                            sA[k][m + j] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }

            // Load sB: Compute grad_W_eff on the fly
            {
                constexpr int VEC_SIZE = 4;
                int numLoads = (BK * BN) / VEC_SIZE;

                for (int i = tid; i < numLoads; i += numThreads) {
                    int k = i / (BN / VEC_SIZE);
                    int n = (i % (BN / VEC_SIZE)) * VEC_SIZE;
                    int globalK = ctak + k;
                    int globalNN = blockIdx.x * BN + n;
                    int globalNNcol = globalNN % N;

                    // Vectorized load from r_prev: load 4 consecutive columns
                    float4 temp = *reinterpret_cast<const float4*>(&r_prev_idx.at(globalK, globalNNcol));

                    // Multiply with sGrad_re (now indexed by globalK, not k)
                    __nv_bfloat16 gradWeffTmp[4];
                    gradWeffTmp[0] = __float2bfloat16(temp.x * sGrad_re[globalK]);
                    gradWeffTmp[1] = __float2bfloat16(temp.y * sGrad_re[globalK]);
                    gradWeffTmp[2] = __float2bfloat16(temp.z * sGrad_re[globalK]);
                    gradWeffTmp[3] = __float2bfloat16(temp.w * sGrad_re[globalK]);

                    // Vectorized store to sB (4 bfloat16 = 1 float2)
                    *reinterpret_cast<float2*>(&sB[k][n]) =
                        *reinterpret_cast<float2*>(&gradWeffTmp[0]);
                }
            }

            __syncthreads();

            // WMMA computation
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> frag_a[WARP_TILING_X];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_b[WARP_TILING_Y];

            for (int k = 0; k < BK; k += WMMA_K) {
                // #pragma unroll
                for (int ti = 0; ti < WARP_TILING_X; ti++) {
                    int sA_col = warpIdy * WMMA_M * WARP_TILING_X + ti * WMMA_M;
                    wmma::load_matrix_sync(frag_a[ti], &sA[k][sA_col], ldsA);
                }

                // #pragma unroll
                for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                    int sB_col = warpIdx * WMMA_N * WARP_TILING_Y + tj * WMMA_N;
                    wmma::load_matrix_sync(frag_b[tj], &sB[k][sB_col], ldsB);
                }

                // #pragma unroll
                for (int ti = 0; ti < WARP_TILING_X; ti++) {
                    // #pragma unroll
                    for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                        wmma::mma_sync(frag_acc[ti][tj], frag_a[ti], frag_b[tj], frag_acc[ti][tj]);
                    }
                }
            }

            __syncthreads();
        }

        // Store results
        #pragma unroll
        for (int ti = 0; ti < WARP_TILING_X; ti++) {
            #pragma unroll
            for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                int mAg = blockIdx.y * BM + warpIdy * WMMA_M * WARP_TILING_X + ti * WMMA_M;
                int nBg = blockIdx.x * BN + warpIdx * WMMA_N * WARP_TILING_Y + tj * WMMA_N;

                if (mAg < M && nBg < NN) {
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> existing_frag;
                    float* C_ptr = &C_idx.at(mAg, nBg);
                    wmma::load_matrix_sync(existing_frag, C_ptr, NN, wmma::mem_row_major);

                    for (int i = 0; i < frag_acc[ti][tj].num_elements; ++i) {
                        frag_acc[ti][tj].x[i] += existing_frag.x[i];
                    }

                    wmma::store_matrix_sync(C_ptr, frag_acc[ti][tj], NN, wmma::mem_row_major);
                }
            }
        }
    }

    template<int TILE_DIM = 32, int BLOCK_ROWS = 8>
    __global__ void transpose_BN_to_NB(
        const float* __restrict__ input,   // [B, N]
        float* __restrict__ output,        // [N, B]
        int B,
        int N
    ) {
        __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

        int x = blockIdx.x * TILE_DIM + threadIdx.x;
        int y = blockIdx.y * TILE_DIM + threadIdx.y;

        // Load input tile (from B x N)
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            int row = y + j;
            if (row < B && x < N) {
                tile[threadIdx.y + j][threadIdx.x] = input[row * N + x];
            }
        }

        __syncthreads();

        // Transpose block indices
        x = blockIdx.y * TILE_DIM + threadIdx.x;
        y = blockIdx.x * TILE_DIM + threadIdx.y;

        // Write transposed output (to N x B)
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            int row = y + j;
            if (row < N && x < B) {
                output[row * B + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }

template<Activation ACT, int TILE_DIM = 32, int BLOCK_ROWS = 8>
__global__ void compute_and_transpose_grad_re_kernel(
    const float* __restrict__ re_ctx_t,      // [B, N]
    const float* __restrict__ grad_output_t, // [B, N]
    const float* __restrict__ grad_r,        // [B, N] - input
    float* __restrict__ grad_re_T,           // [N, B] - transposed output
    float alpha,
    int N,
    int B)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;  // N dimension
    int y = blockIdx.y * TILE_DIM + threadIdx.y;  // B dimension

    // Load input tile, compute grad_re, and store in shared memory
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;  // batch index
        int col = x;      // neuron index

        if (row < B && col < N) {
            int idx = row * N + col;

            // Compute grad_re on the fly
            float grad_r_current = grad_output_t[idx] + grad_r[idx];
            float re_val = re_ctx_t[idx];
            float dphi = activation_derivative<ACT>(re_val);
            float grad_re = dphi * grad_r_current * alpha;

            // Store in shared memory tile for transpose
            tile[threadIdx.y + j][threadIdx.x] = grad_re;
        }
    }

    __syncthreads();

    // Transpose block indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // B dimension
    y = blockIdx.x * TILE_DIM + threadIdx.y;  // N dimension

    // Write transposed output (to N x B)
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;  // neuron index
        int col = x;      // batch index

        if (row < N && col < B) {
            grad_re_T[row * B + col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}


    template<Activation ACT>
    void bwd_wmma_impl(
        const float* grad_output,   // [S, B, N]
        const float* A,             // [B, S, A_dim]
        const float* Wa,            // [A_dim, N, N]
        float J0,
        float J1,
        const float* Wo,            // [N, N]
        const float* bump_history,  // [S, B, N]
        const float* r_init,        // [B, N]
        const float* re_ctx,        // [S, B, N] - saved from forward
        float* grad_Wa,             // [A_dim, N*N]
        float* grad_Wo,             // [N, N]
        float alpha,
        int N,
        int a_dim,
        int seq_len,
        int batch_size,
        // NEW: workspace buffers passed from PyTorch
        float* grad_r,              // [B, N]
        float* W_eff_temp,          // [B, N, N]
        float* grad_re_temp,        // [B, N]
        __nv_bfloat16* grad_W_eff_temp_bf16,  // [B, N, N]
        __nv_bfloat16* A_t_temp_bf16,         // [B, a_dim]
        cublasHandle_t handle                  // NEW: reuse cuBLAS handle
    )
    {


        float *grad_re_temp_T;
        cudaMalloc(&grad_re_temp_T, batch_size * N * sizeof(float));
        // -----------------------------------------------------------------
        // Back‑propagation through time (reverse order)
        // -----------------------------------------------------------------
        for (int t = seq_len - 1; t >= 0; --t) {
            const float* r_prev = (t == 0) ? r_init : (bump_history + (t - 1) * batch_size * N);

            // Kernel 2A: Compute grad_re
            {
                dim3 block(128);
                dim3 grid(batch_size, (N + block.x - 1) / block.x);

                compute_grad_re_kernel<ACT><<<grid, block>>>(
                    re_ctx + t * batch_size * N,
                    grad_output + t * batch_size * N,
                    grad_r,
                    grad_re_temp,
                    alpha, N, batch_size);
                CHECK_CUDA_ERROR(cudaGetLastError());
            }

            // Tranpose grad_re_temp
            {


                int B = batch_size;
                constexpr int TILE_DIM = 32;
                constexpr int BLOCK_ROWS = 8;

                dim3 blockDim(TILE_DIM, BLOCK_ROWS);
                dim3 gridDim(
                    (N + TILE_DIM - 1) / TILE_DIM,
                    (B + TILE_DIM - 1) / TILE_DIM
                );

                transpose_BN_to_NB<TILE_DIM, BLOCK_ROWS><<<gridDim, blockDim>>>(
                    grad_re_temp, grad_re_temp_T, B, N
                );
            }

            // -------------------------------------------------------------
            // Kernel 1 – recompute W_eff
            // -------------------------------------------------------------
            {
                constexpr int BM = 64;
                constexpr int BN = 64;
                constexpr int BK = 16;
                constexpr int WMMA_M = 16;
                constexpr int WMMA_N = 16;
                constexpr int WMMA_K = 16;

                dim3 blockSize((BN / WMMA_N) * WARPSIZE, BM / WMMA_M);
                dim3 gridSize(N, (batch_size + BM - 1) / BM);

                recompute_Weff_kernel<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K><<<gridSize, blockSize>>>(
                        A, Wa, J0, J1, Wo,
                        W_eff_temp, A_t_temp_bf16,
                        N, a_dim, seq_len, t, batch_size);
                CHECK_CUDA_ERROR(cudaGetLastError());
            }

            // Kernel 2B: Update grad_r
            {
                dim3 block(128);
                dim3 grid(batch_size, (N + block.x - 1) / block.x);

                update_grad_r_kernel<<<grid, block>>>(
                    W_eff_temp,
                    grad_re_temp,
                    grad_output + t * batch_size * N,
                    grad_r,
                    alpha, N, batch_size);
                CHECK_CUDA_ERROR(cudaGetLastError());
            }

            // -------------------------------------------------------------
            // Kernel 3 – Custom batched outer product with BF16 output
            // -------------------------------------------------------------


            // {
            //     constexpr int TILE_M = 64;
            //     constexpr int TILE_N = 64;
            //     constexpr int THREAD_TILE_M = 4;
            //     constexpr int THREAD_TILE_N = 4;
            //     dim3 block3(TILE_N / THREAD_TILE_N, TILE_M / THREAD_TILE_M);
            //     dim3 grid3((N + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M, batch_size);
            //
            //     BMM_kernel_4a<TILE_M, TILE_N, THREAD_TILE_M, THREAD_TILE_N>
            //         <<<grid3, block3>>>(
            //             grad_re_temp, r_prev, grad_W_eff_temp_bf16, N, batch_size);
            //     CHECK_CUDA_ERROR(cudaGetLastError());
            // }

            // -------------------------------------------------------------
            // Kernel 4 (cuBLAS) – grad_Wo += J1 * r_prev^T @ grad_re
            // -------------------------------------------------------------
            {
                float beta = 1.0f;
                cublasSgemm(handle,
                            CUBLAS_OP_N,        // r_prev (no transpose)
                            CUBLAS_OP_T,        // grad_re^T
                            N,                  // m
                            N,                  // n
                            batch_size,         // k
                            &J1,                // alpha
                            r_prev,             // A: [B, N]
                            N,                  // lda
                            grad_re_temp,       // B: [B, N]
                            N,                  // ldb
                            &beta,              // beta
                            grad_Wo,            // C: [N, N]
                            N);                 // ldc
                CHECK_CUDA_ERROR(cudaGetLastError());
            }

            // -------------------------------------------------------------
            // Kernel 5 – WMMA grad_Wa
            // -------------------------------------------------------------
            // {
            //     int M = a_dim;
            //     int N_ = N * N;
            //     int K = batch_size;
            //
            //     constexpr int BM = 32;
            //     constexpr int BN = 128;
            //     constexpr int BK = 64;
            //     constexpr int WMMA_M = 16;
            //     constexpr int WMMA_N = 16;
            //     constexpr int WMMA_K = 16;
            //     constexpr int WARP_TILING_X = 1;
            //     constexpr int WARP_TILING_Y = 1;
            //     constexpr int WARPS_M = BM / WMMA_M / WARP_TILING_X;
            //     constexpr int WARPS_N = BN / WMMA_N / WARP_TILING_Y;
            //
            //     dim3 blockSize(WARPS_N * WARPSIZE, WARPS_M);
            //     dim3 gridSize((N_ + BN - 1) / BN, (M + BM - 1) / BM);
            //
            //     static_assert(BM % (WMMA_M * WARP_TILING_X) == 0, "BM must be multiple of WMMA_M * WARP_TILING_X");
            //     static_assert(BN % (WMMA_N * WARP_TILING_Y) == 0, "BN must be multiple of WMMA_N * WARP_TILING_Y");
            //     static_assert(BK % WMMA_K == 0, "BK must be multiple of WMMA_K");
            //     static_assert(WARPS_M * WARPS_N * WARPSIZE <= 1024, "Too many threads per block");
            //
            //     gemm_kernel<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARP_TILING_X, WARP_TILING_Y>
            //         <<<gridSize, blockSize>>>(A_t_temp_bf16, grad_W_eff_temp_bf16, grad_Wa, M, N_, K);
            //     CHECK_CUDA_ERROR(cudaGetLastError());
            // }





            {
                int M = a_dim;
                int NN = N * N;
                int K = batch_size;

                constexpr int BM = 32;
                constexpr int BN = 128;
                constexpr int BK = 128;
                constexpr int WMMA_M = 16;
                constexpr int WMMA_N = 16;
                constexpr int WMMA_K = 16;
                constexpr int WARP_TILING_X = 1;
                constexpr int WARP_TILING_Y = 1;
                constexpr int WARPS_M = BM / WMMA_M / WARP_TILING_X;
                constexpr int WARPS_N = BN / WMMA_N / WARP_TILING_Y;

                dim3 blockSize(WARPS_N * WARPSIZE, WARPS_M);
                dim3 gridSize((NN + BN - 1) / BN, (M + BM - 1) / BM);

                static_assert(BM % (WMMA_M * WARP_TILING_X) == 0, "BM must be multiple of WMMA_M * WARP_TILING_X");
                static_assert(BN % (WMMA_N * WARP_TILING_Y) == 0, "BN must be multiple of WMMA_N * WARP_TILING_Y");
                static_assert(BK % WMMA_K == 0, "BK must be multiple of WMMA_K");
                static_assert(WARPS_M * WARPS_N * WARPSIZE <= 1024, "Too many threads per block");

                size_t shared_mem_size = K * sizeof(float);
                grad_W_eff_gemm_kernel<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARP_TILING_X, WARP_TILING_Y>
                    <<<gridSize, blockSize, shared_mem_size>>>(A_t_temp_bf16, grad_re_temp_T, r_prev, grad_W_eff_temp_bf16, grad_Wa, M, NN, K, N);

                CHECK_CUDA_ERROR(cudaGetLastError());
            }


        }
        cudaFree(grad_re_temp_T);
    }



    // Main launcher
    void bwd_wmma_launcher(
        const float* grad_output,
        const float* A,
        const float* Wa,
        float J0,
        float J1,
        const float* Wo,
        const float* bump_history,
        const float* r_init,
        const float* re_ctx,
        float* grad_Wa,
        float* grad_Wo,
        float alpha,
        int N,
        int a_dim,
        int seq_len,
        int batch_size,
        int activation_type,
        float* grad_r,
        float* W_eff_temp,
        float* grad_re_temp,
        __nv_bfloat16* grad_W_eff_temp_bf16,
        __nv_bfloat16* A_t_temp_bf16,
        cublasHandle_t handle  // NEW: pass handle from outside
    ){
        switch(activation_type){
            case 0: bwd_wmma_impl<Activation::RELU>(grad_output, A, Wa, J0, J1, Wo, bump_history, r_init, re_ctx, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size, grad_r, W_eff_temp, grad_re_temp, grad_W_eff_temp_bf16, A_t_temp_bf16, handle); break;
            case 1: bwd_wmma_impl<Activation::GELU>(grad_output, A, Wa, J0, J1, Wo, bump_history, r_init, re_ctx, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size, grad_r, W_eff_temp, grad_re_temp, grad_W_eff_temp_bf16, A_t_temp_bf16, handle); break;
            case 2: bwd_wmma_impl<Activation::TANH>(grad_output, A, Wa, J0, J1, Wo, bump_history, r_init, re_ctx, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size, grad_r, W_eff_temp, grad_re_temp, grad_W_eff_temp_bf16, A_t_temp_bf16, handle); break;
            case 3: bwd_wmma_impl<Activation::SILU>(grad_output, A, Wa, J0, J1, Wo, bump_history, r_init, re_ctx, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size, grad_r, W_eff_temp, grad_re_temp, grad_W_eff_temp_bf16, A_t_temp_bf16, handle); break;
        }
    }
}// namespace cuda_wmma_re