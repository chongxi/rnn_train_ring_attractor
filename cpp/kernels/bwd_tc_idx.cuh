// #include "cuda_common.cuh"
#include "../cuda_common.cuh"

namespace cuda_wmma {
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

                    frag_Wa_weighted.x[regID] = J0 + J1 * Wo[colData] + frag_Wa_weighted.x[regID];
                }
            }
        }
    }

    
    // ---------------------------------------------------------------
    // Kernel 1 – recompute W_eff and the recurrent input “re”
    // ---------------------------------------------------------------
    template<Activation ACT, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
    // __global__ void __launch_bounds__((BN / WMMA_N) * WARPSIZE * (BM / WMMA_M)) recompute_Weff_re_kernel(
    __global__ void __launch_bounds__((BN / WMMA_N) * WARPSIZE * (BM / WMMA_M)) recompute_Weff_re_grad_re_kernel(
        const float* A,               // [B, S, A_dim]   (float → cast to half)
        const float* Wa,              // [A_dim, N, N]   (float → cast to half)
        float J0,
        float J1,
        const float* Wo,              // [N, N]
        const float* bump_history,    // [S+1, B, N]    (float)
        const float* grad_output,     // [S, B, N]
        float* grad_r,                // [B, N] – in/out
        float*       W_eff_out,       // [B, N, N]      (float accumulator)
        float*       grad_re_out,     // [B, N]         (float grad_re output)
        float        alpha,
        int          N,               // N
        int          a_dim,
        int          seq_len,
        int          t,               // current time‑step (0‑based)
        int          batch_size)
    {
        // -----------------------------------------------------------------
        // Index wrappers (same as forward kernel)
        // -----------------------------------------------------------------
        IndexWrapper<const float, 3> A_idx (A,  batch_size, seq_len, a_dim);
        IndexWrapper<const float, 3> Wa_idx(Wa, a_dim, N, N);
        IndexWrapper<const float, 2> Wo_idx(Wo, N, N);
        IndexWrapper<const float, 3> bump_history_idx(bump_history, seq_len + 1, batch_size, N);
        IndexWrapper<const float, 3> grad_output_idx(grad_output, seq_len, batch_size, N);
        IndexWrapper<float, 2> grad_r_idx(grad_r, batch_size, N);
        IndexWrapper<float, 2> W_eff_out_idx(W_eff_out, batch_size, N * N);
        IndexWrapper<float, 2> grad_re_out_idx(grad_re_out, batch_size, N);

        auto cta = cg::this_thread_block();

        const size_t warp_idx_x = threadIdx.x / WARPSIZE;
        const size_t warp_idx_y = threadIdx.y;

        const int global_m_base = blockIdx.y * BM;
        const int global_Non = blockIdx.x;

        if (global_m_base >= batch_size || global_Non >= N) return;

        constexpr int ld = BK + 8;
        constexpr int lda = ld;
        constexpr int ldb = ld;

        // TODO: Increase size of tileA to [BN * n + 8][ld] for better smem reuse ???
        __shared__ half tileA[BM][ld];
        __shared__ half tileB[BN][ld];

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_re;
        wmma::fill_fragment(frag_re, 0.f);

        for (int cta_n = 0; cta_n < N; cta_n += BN) {

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_Wa_weighted;
            wmma::fill_fragment(frag_Wa_weighted, 0.f);

            for (int cta_k = 0; cta_k < a_dim; cta_k += BK) {

                //===================== Step 1: Compute Wa_weighted[BM, BN] tile ===========================================
                {
                    const int total_threads = blockDim.x * blockDim.y;
                    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
                    const int num_loads = (BM * BK) / 2;

                    for (int idx = thread_id; idx < num_loads; idx += total_threads) {
                        int load_m = idx / (BK / 2);
                        int load_k_base = (idx % (BK / 2)) * 2;

                        int global_m = global_m_base + load_m;
                        int global_k = cta_k + load_k_base;

                        if (global_m < batch_size && global_k + 1 < a_dim) {
                            const float* src_ptr = &A_idx.at(global_m, t, global_k);
                            float2 data = *reinterpret_cast<const float2*>(src_ptr);

                            tileA[load_m][load_k_base] = __float2half(data.x);
                            tileA[load_m][load_k_base + 1] = __float2half(data.y);
                        } else {
                            tileA[load_m][load_k_base] = __float2half(0.f);
                            tileA[load_m][load_k_base + 1] = __float2half(0.f);
                        }
                    }
                }

                {   // Transpose load into tileB, for faster WMMA frag b load, lds.128
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

                            tileB[load_n_base][load_k] = __float2half(data.x);
                            tileB[load_n_base + 1][load_k] = __float2half(data.y);
                        } else {
                            tileB[load_n_base][load_k] = __float2half(0.f);
                            tileB[load_n_base + 1][load_k] = __float2half(0.f);
                        }
                    }
                }

                __syncthreads();

                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b;

    #pragma unroll
                for (int mma_k = 0; mma_k < BK; mma_k += WMMA_K) {
                    wmma::load_matrix_sync(frag_a, &tileA[0][0] + (warp_idx_y * WMMA_M) * lda + mma_k, lda);
                    wmma::load_matrix_sync(frag_b, &tileB[0][0] + (warp_idx_x * WMMA_N) * ldb + mma_k, ldb);
                    wmma::mma_sync(frag_Wa_weighted, frag_a, frag_b, frag_Wa_weighted);
                }

                __syncthreads();
            } // end for cta_k

            //============================ Step 2: Apply epilogue to get W_eff[BM, BN] =====================================
            //The result of BMxBN matrix mulitplication is Wa_weighted stored in frags, They have the result of
            //a single row in vector r in batches. Wa_weighted stored in frags will be applied an epilogue to create W_eff.
            //frag_Wa_weighted now stores W_eff results
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

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_r;

            //============================= Step 3: Load r[BM, BN] tile and perform partial sum ============================
            int frag_m = global_m_base + warp_idx_y * WMMA_M;
            if (frag_m < batch_size && frag_n < N) {
                const float* pos_frag_r = &bump_history_idx.at(t, frag_m, frag_n);
                wmma::load_matrix_sync(frag_r, pos_frag_r, N, wmma::mem_row_major);
                for (size_t i = 0; i < frag_r.num_elements; ++i) {
                    frag_re.x[i] += frag_Wa_weighted.x[i] * frag_r.x[i];
                }
            }
        } // end for cta_n

        //========= Step 4: Reduce (full sum) frag_re across BN dimension to get a single column of re in batches ==========
        // TODO: Reduce the size of re_temp, it's using too much smem. REVIEW: current impl is already optimized
        constexpr int ldtemp = BN + 2;
        __shared__ float re_temp[BM][ldtemp];

        wmma::store_matrix_sync(
            &re_temp[warp_idx_y * WMMA_M][warp_idx_x * WMMA_N],
            frag_re,
            ldtemp,
            wmma::mem_row_major);

        __syncthreads();

        auto warp_group = cg::tiled_partition<WARPSIZE>(cta);
        size_t lane_id = warp_group.thread_rank();
        size_t warp_id = warp_group.meta_group_rank();

        for (size_t warp_m = 0; warp_m < BM; warp_m += warp_group.meta_group_size()) {
            int global_m = global_m_base + warp_m + warp_id;
            if (global_m >= batch_size) continue;

            float re_sum = 0.f;
            for (int warp_n = 0; warp_n < BN; warp_n += WARPSIZE) {
                re_sum += re_temp[warp_m + warp_id][warp_n + lane_id];
            }
            float re_max = cg::reduce(warp_group, re_sum, cg::plus<float>());

            if (lane_id == 0) {
                re_temp[warp_m + warp_id][0] = re_max;
                // re_out_idx.at(global_m, global_Non) = re_max;
            }
        }

        __syncthreads();

        //========================= Step 5: Compute grad_re from re ==============================================
        if (cta.thread_rank() < BM) {
            int global_m = global_m_base + cta.thread_rank();
            if (global_m < batch_size && global_Non < N) {
                float re_val = re_temp[cta.thread_rank()][0];

                float go = grad_output_idx.at(t, global_m, global_Non);
                float gr = grad_r_idx.at(global_m, global_Non) + go;
                grad_r_idx.at(global_m, global_Non) = gr;

                float dphi = activation_derivative<ACT>(re_val);
                grad_re_out_idx.at(global_m, global_Non) = dphi * gr * alpha;
            }
        }
    }

    // ---------------------------------------------------------------
    // Kernel 2 – grad_re = (grad_r + grad_output) * alpha * activation_derivative(re)
    // ---------------------------------------------------------------
    template<Activation ACT>
    __global__ void compute_grad_re_kernel(
        const float* grad_output,   // [S, B, N]
        float* grad_r,              // [B, N] – in/out
        const float* re,            // [B, N]
        float* grad_re,             // [B, N] – output
        float alpha,
        int N,
        int seq_len,
        int t,
        int batch_size)
    {
        int b = blockIdx.x;
        int lane = threadIdx.x;
        for (int n = lane; n < N; n += blockDim.x) {
            // 1) add upstream gradient
            float go = grad_output[t * batch_size * N + b * N + n];
            float gr = grad_r[b * N + n] + go;
            grad_r[b * N + n] = gr;

            // 2) scale by α and activation derivative
            float re_val = re[b * N + n];
            float dphi = activation_derivative<ACT>(re_val);
            grad_re[b * N + n] = gr * alpha * dphi;
        }
    }

    // // ---------------------------------------------------------------
    // // Kernel 3 – grad_r = W_effᵀ·grad_re + (1‑α)·grad_r (optimized with smem)
    // // ---------------------------------------------------------------
    // template<int CG_TILE_SIZE, int CG_GROUP_SIZE>
    // __global__ void compute_grad_r_kernel(
    //     const float* W_eff,   // [B, N, N]
    //     const float* grad_re, // [B, N]
    //     float* grad_r,        // [B, N] – in/out
    //     float alpha,
    //     int N,
    //     int batch_size)
    // {
    //     IndexWrapper<const float, 3> W_eff_idx(W_eff, batch_size, N, N);
    //     IndexWrapper<const float, 2> grad_re_idx(grad_re, batch_size, N);
    //     IndexWrapper<float, 2> grad_r_idx(grad_r, batch_size, N);
    //
    //     constexpr int TILE_K = CG_TILE_SIZE * 8;
    //     constexpr int CG_GROUP_TILE = 16;
    //
    //     __shared__ float tileA[CG_GROUP_SIZE][TILE_K + 1];
    //     __shared__ float tileB[TILE_K + 1];
    //     __shared__ float final_sum[CG_GROUP_SIZE];
    //
    //     int batch_idx = blockIdx.x;
    //     int row_base = blockIdx.y * CG_GROUP_SIZE;
    //     if (batch_idx >= batch_size) return;
    //
    //     auto cta = cg::this_thread_block();
    //     auto tile = cg::tiled_partition<CG_TILE_SIZE>(cta);
    //
    //     int group_id = tile.meta_group_rank();
    //     int lane = tile.thread_rank();
    //     int row_idx = row_base + group_id;
    //
    //     float partial_sum = 0.0f;
    //
    //     for (int k_base = 0; k_base < N; k_base += TILE_K) {
    //         // Transpose load for tileA using separate cooperative group
    //         {
    //             auto cg_tile_loadA = cg::tiled_partition<CG_GROUP_TILE>(cta);
    //             int load_group_id = cg_tile_loadA.meta_group_rank();
    //             int load_lane = cg_tile_loadA.thread_rank();
    //
    //             for (int k_offset = load_group_id; k_offset < TILE_K; k_offset += CG_GROUP_SIZE) {
    //                 int k = k_base + k_offset;
    //                 int row = row_base + load_lane;
    //
    //                 if (k < N && row < N) {
    //                     float val = W_eff_idx.at(batch_idx, k, row);
    //                     tileA[load_lane][k_offset] = val;
    //                 } else {
    //                     tileA[load_lane][k_offset] = 0.0f;
    //                 }
    //             }
    //         }
    //
    //         // Coalesced load tileB
    //         for (int k_offset = threadIdx.x; k_offset < TILE_K; k_offset += blockDim.x) {
    //             int k = k_base + k_offset;
    //             if (k < N) {
    //                 tileB[k_offset] = grad_re_idx.at(batch_idx, k);
    //             } else {
    //                 tileB[k_offset] = 0.0f;
    //             }
    //         }
    //
    //         __syncthreads();
    //
    //         // Compute partial sum
    //         if (row_idx < N) {
    //             for (int k_offset = lane; k_offset < TILE_K; k_offset += CG_TILE_SIZE) {
    //                 partial_sum += tileA[group_id][k_offset] * tileB[k_offset];
    //             }
    //         }
    //
    //         __syncthreads();
    //     }
    //
    //     // Reduce partial sum across tile
    //     if (row_idx < N) {
    //         float reduced_sum = cg::reduce(tile, partial_sum, cg::plus<float>());
    //         if (lane == 0) {
    //             final_sum[group_id] = reduced_sum;
    //         }
    //     }
    //
    //     __syncthreads();
    //
    //     // Epilogue
    //     if (threadIdx.x < CG_GROUP_SIZE) {
    //         int row = row_base + threadIdx.x;
    //         if (row < N) {
    //             float old_grad_r = grad_r_idx.at(batch_idx, row);
    //             grad_r_idx.at(batch_idx, row) = final_sum[threadIdx.x] + old_grad_r * (1.0f - alpha);
    //         }
    //     }
    // }

    // ---------------------------------------------------------------
    // Kernel 3 – grad_r = W_effᵀ·grad_re + (1‑α)·grad_r
    // ---------------------------------------------------------------
    __global__ void compute_grad_r_kernel(
        const float* W_eff,   // [B, N, N]
        const float* grad_re, // [B, N]
        float* grad_r,        // [B, N] – in/out
        float alpha,
        int N,
        int batch_size)
    {
        int b = blockIdx.x;
        int lane = threadIdx.x;
        for (int i = lane; i < N; i += blockDim.x) {
            // (W_effᵀ·grad_re)_i = Σ_j W_eff[j,i] * grad_re_j
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                float w = W_eff[b * N * N + j * N + i];   // transpose on‑the‑fly
                sum += w * grad_re[b * N + j];
            }
            float old = grad_r[b * N + i];
            grad_r[b * N + i] = sum + old * (1.0f - alpha);
        }
    }

    // ---------------------------------------------------------------
    // Kernel 5 – grad_Wa += A_tᵀ · grad_W_eff   (BF16‑style GEMM)
    // ---------------------------------------------------------------
    template<int BM, int BN, int BK,
             int WMMA_M, int WMMA_N, int WMMA_K>
    __global__ void compute_grad_Wa_kernel(
        const float* A,               // [B, S, A_dim]
        const float* grad_W_eff,      // [B, N, N] (flattened as N*N)
        float* grad_Wa,               // [A_dim, N*N] – accumulated
        int N,
        int a_dim,
        int seq_len,
        int t,
        int batch_size)
    {
        // One thread per (a_dim, N*N) element
        int row = blockIdx.x * blockDim.x + threadIdx.x;   // a_dim index
        int col = blockIdx.y * blockDim.y + threadIdx.y;   // N*N index
        if (row >= a_dim || col >= N * N) return;

        float acc = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            float a_val = A[b * seq_len * a_dim + t * a_dim + row];
            float w_val = grad_W_eff[b * N * N + col];
            acc += a_val * w_val;
        }
        // Accumulate over time‑steps (the launcher zero‑initialises grad_Wa)
        atomicAdd(&grad_Wa[row * N * N + col], acc);
    }


template<Activation ACT>
void bwd_wmma_impl(
    const float* grad_output,   // [B, S, N]
    const float* A,             // [B, S, A_dim]
    const float* Wa,            // [A_dim, N, N]
    float J0,                   // scalar
    float J1,                   // scalar
    const float* Wo,            // [N, N]
    const float* bump_history,  // [B, S+1, N]
    float* grad_Wa,             // [A_dim, N*N] – **must be zeroed** before call
    float* grad_Wo,             // [N, N]    – **must be zeroed** before call
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size)
{
    // -----------------------------------------------------------------
    // Temporary buffers (all FP32 for simplicity)
    // -----------------------------------------------------------------
    float *grad_r, *W_eff_temp, *grad_re_temp, *grad_W_eff_temp;
    cudaMalloc(&grad_r,        batch_size * N * sizeof(float));
    cudaMalloc(&W_eff_temp,   batch_size * N * N * sizeof(float));
    cudaMalloc(&grad_re_temp, batch_size * N * sizeof(float));
    cudaMalloc(&grad_W_eff_temp, batch_size * N * N * sizeof(float));

    cudaMemset(grad_r, 0, batch_size * N * sizeof(float));

    // -----------------------------------------------------------------
    // cuBLAS handle
    // -----------------------------------------------------------------
    cublasHandle_t handle;
    cublasCreate(&handle);

    // -----------------------------------------------------------------
    // Tile sizes
    // -----------------------------------------------------------------
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // -----------------------------------------------------------------
    // Back‑propagation through time (reverse order)
    // -----------------------------------------------------------------
    for (int t = seq_len - 1; t >= 0; --t) {
        // -------------------------------------------------------------
        // Kernel 1+2 – recompute W_eff, re, and compute grad_re
        // -------------------------------------------------------------
        dim3 blockSize((BN / WMMA_N) * WARPSIZE, BM / WMMA_M);
        dim3 gridSize(N, (batch_size + BM - 1) / BM);

        recompute_Weff_re_grad_re_kernel<ACT, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K><<<gridSize, blockSize>>>(
                A, Wa, J0, J1, Wo,
                bump_history,
                grad_output, grad_r,
                W_eff_temp, grad_re_temp,
                alpha, N, a_dim, seq_len, t, batch_size);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // -------------------------------------------------------------
        // Kernel 3 – grad_r
        // -------------------------------------------------------------
        dim3 grid3(batch_size);
        dim3 block3(256);
        compute_grad_r_kernel<<<grid3, block3>>>(
            W_eff_temp, grad_re_temp, grad_r,
            alpha, N, batch_size);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // -------------------------------------------------------------
        // Kernel 4a+4b (fused) – grad_Wo += J1 * grad_re.T @ r_prev
        // GEMM: [N, B] @ [B, N] = [N, N]
        // -------------------------------------------------------------
        const float* r_prev = bump_history + t * batch_size * N;  // [B, N]
        float beta = 1.0f;  // accumulate

        cublasSgemm(handle,
                    CUBLAS_OP_N,        // r_prev (no transpose)
                    CUBLAS_OP_T,        // grad_re^T
                    N,                  // m
                    N,                  // n
                    batch_size,         // k
                    &J1,                // alpha
                    r_prev,             // A: [B, N]
                    N,                  // lda: N (row-major stride)
                    grad_re_temp,       // B: [B, N]
                    N,                  // ldb: N (row-major stride)
                    &beta,              // beta
                    grad_Wo,            // C: [N, N]
                    N);                 // ldc
        CHECK_CUDA_ERROR(cudaGetLastError());

        // -------------------------------------------------------------
        // Kernel 5 – grad_Wa += A_t.T @ grad_W_eff
        // -------------------------------------------------------------

        dim3 block5(16, 16);
        dim3 grid5((a_dim + block5.x - 1) / block5.x,
                   ((N * N) + block5.y - 1) / block5.y);
        compute_grad_Wa_kernel<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>
            <<<grid5, block5>>>(
                A, grad_W_eff_temp, grad_Wa,
                N, a_dim, seq_len, t, batch_size);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    // -----------------------------------------------------------------
    // Clean‑up
    // -----------------------------------------------------------------
    cublasDestroy(handle);

    cudaFree(grad_r);
    cudaFree(W_eff_temp);
    cudaFree(grad_re_temp);
    cudaFree(grad_W_eff_temp);
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
        float* grad_Wa,
        float* grad_Wo,
        float alpha,
        int N,
        int a_dim,
        int seq_len,
        int batch_size,
        int activation_type
    ){
        switch(activation_type){
            case 0: bwd_wmma_impl<Activation::RELU>(grad_output, A, Wa, J0, J1, Wo, bump_history, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size); break;
            case 1: bwd_wmma_impl<Activation::GELU>(grad_output, A, Wa, J0, J1, Wo, bump_history, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size); break;
            case 2: bwd_wmma_impl<Activation::TANH>(grad_output, A, Wa, J0, J1, Wo, bump_history, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size); break;
            case 3: bwd_wmma_impl<Activation::SILU>(grad_output, A, Wa, J0, J1, Wo, bump_history, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size); break;
        }
    }
}// namespace cuda_wmma

