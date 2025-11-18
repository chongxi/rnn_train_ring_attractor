// #include "cuda_common.cuh"
#include <charconv>

#include "../cuda_common.cuh"
#include <cuda_bf16.h>

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
        size_t threadID_in_group = threadID_in_warp % 4;

        for (int regBlockRow = 0; regBlockRow < NUM_REGBLOCK_ROWS; ++regBlockRow) {
            for (int regBlockCol = 0; regBlockCol < NUM_REGBLOCK_COLS; ++regBlockCol) {
                for (int i = 0; i < REGS_PER_BLOCK; ++i) {
                    int regID = (regBlockRow * REGS_PER_BLOCK) +
                           (regBlockCol * NUM_REGBLOCK_ROWS * REGS_PER_BLOCK) + i;
                    size_t colData = regBlockCol * 8 + threadID_in_group * 2 + i;

                    frag_Wa_weighted.x[regID] = J0 + J1 * Wo[colData] + frag_Wa_weighted.x[regID];
                }
            }
        }
    }


    // ---------------------------------------------------------------
    // Kernel 1 – recompute W_eff and the recurrent input "re"
    // ---------------------------------------------------------------
    template<Activation ACT, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
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
        __nv_bfloat16* A_t_temp,      // [B, A_dim] - BF16
        __nv_bfloat16* grad_W_eff,    // [B, N, N] - BF16
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
        IndexWrapper<__nv_bfloat16, 3> grad_W_eff_idx(grad_W_eff, batch_size, N, N);


        auto cta = cg::this_thread_block();

        const size_t warp_idx_x = threadIdx.x / WARPSIZE;
        const size_t warp_idx_y = threadIdx.y;

        const int global_m_base = blockIdx.y * BM;
        const int global_Non = blockIdx.x;

        if (global_m_base >= batch_size || global_Non >= N) return;

        constexpr int ld = BK + 8;
        constexpr int lda = ld;
        constexpr int ldb = ld;

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
                    const int num_loads = (BM * BK) / 4;

                    for (int idx = thread_id; idx < num_loads; idx += total_threads) {
                        int load_m = idx / (BK / 4);
                        int load_k_base = (idx % (BK / 4)) * 4;

                        int global_m = global_m_base + load_m;
                        int global_k = cta_k + load_k_base;

                        if (global_m < batch_size) {
                            const float* src_ptr = &A_idx.at(global_m, t, global_k);
                            float4 data = *reinterpret_cast<const float4*>(src_ptr);

                            tileA[load_m][load_k_base] = __float2half(data.x);
                            tileA[load_m][load_k_base + 1] = __float2half(data.y);
                            tileA[load_m][load_k_base + 2] = __float2half(data.z);
                            tileA[load_m][load_k_base + 3] = __float2half(data.w);

                            // Convert to BF16 and store
                            __nv_bfloat16* a_dest = &A_t_temp[global_m * a_dim + global_k];
                            a_dest[0] = __float2bfloat16(data.x);
                            a_dest[1] = __float2bfloat16(data.y);
                            a_dest[2] = __float2bfloat16(data.z);
                            a_dest[3] = __float2bfloat16(data.w);
                        } else {
                            tileA[load_m][load_k_base] = __float2half(0.f);
                            tileA[load_m][load_k_base + 1] = __float2half(0.f);
                            tileA[load_m][load_k_base + 2] = __float2half(0.f);
                            tileA[load_m][load_k_base + 3] = __float2half(0.f);
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
                float grad_re_val = dphi * gr * alpha;
                grad_re_out_idx.at(global_m, global_Non) = grad_re_val;
            }
        }
    }


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
        int i = blockIdx.y * blockDim.x + threadIdx.x;

        if (b >= batch_size || i >= N) return;

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float w = W_eff[b * N * N + j * N + i];
            sum += w * grad_re[b * N + j];
        }

        grad_r[b * N + i] = sum + grad_r[b * N + i] * (1.0f - alpha);
    }


    // ---------------------------------------------------------------
    // Kernel 4a – Custom Batched GEMM with BF16 output
    // Computes: grad_W_eff[b, i, j] = grad_re[b, i] * r_prev[b, j]
    // This is an outer product for each batch
    // ---------------------------------------------------------------
    template<int TILE_M, int TILE_N, int THREAD_TILE_M, int THREAD_TILE_N>
    __global__ void batched_outer_product_bf16_kernel(
        const float* __restrict__ grad_re,    // [B, N] - rows of output
        const float* __restrict__ r_prev,     // [B, N] - cols of output
        __nv_bfloat16* __restrict__ grad_W_eff, // [B, N, N] output
        int N,
        int batch_size)
    {
        const int b = blockIdx.z;
        if (b >= batch_size) return;

        const int tid_x = threadIdx.x;
        const int tid_y = threadIdx.y;
        const int block_row = blockIdx.y * TILE_M;  // row in output
        const int block_col = blockIdx.x * TILE_N;  // col in output

        // Shared memory for the vectors
        __shared__ float smem_grad_re[TILE_M];
        __shared__ float smem_r_prev[TILE_N];

        // Load grad_re (rows)
        for (int i = tid_y * blockDim.x + tid_x; i < TILE_M; i += blockDim.x * blockDim.y) {
            int global_row = block_row + i;
            smem_grad_re[i] = (global_row < N) ? grad_re[b * N + global_row] : 0.0f;
        }

        // Load r_prev (cols)
        for (int j = tid_y * blockDim.x + tid_x; j < TILE_N; j += blockDim.x * blockDim.y) {
            int global_col = block_col + j;
            smem_r_prev[j] = (global_col < N) ? r_prev[b * N + global_col] : 0.0f;
        }

        __syncthreads();

        // Compute outer product for this thread's tile
        #pragma unroll
        for (int m = 0; m < THREAD_TILE_M; ++m) {
            #pragma unroll
            for (int n = 0; n < THREAD_TILE_N; ++n) {
                int local_row = tid_y * THREAD_TILE_M + m;
                int local_col = tid_x * THREAD_TILE_N + n;

                if (local_row < TILE_M && local_col < TILE_N) {
                    int global_row = block_row + local_row;
                    int global_col = block_col + local_col;

                    if (global_row < N && global_col < N) {
                        float result = smem_grad_re[local_row] * smem_r_prev[local_col];
                        grad_W_eff[b * N * N + global_row * N + global_col] = __float2bfloat16(result);
                    }
                }
            }
        }
    }

// ---------------------------------------------------------------
// Kernel 5 – WMMA-based grad_Wa computation (optimized like Kernel 1)
// grad_Wa[a_dim, N*N] += A_t^T[a_dim, B] @ grad_W_eff[B, N*N]
// ---------------------------------------------------------------
template<int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void __launch_bounds__((BN / WMMA_N) * WARPSIZE * (BM / WMMA_M))
compute_grad_Wa_wmma_kernel(
    const __nv_bfloat16* A_t,        // [B, a_dim] in BF16
    const __nv_bfloat16* grad_W_eff, // [B, N*N] in BF16
    float* grad_Wa,                   // [a_dim, N*N] in FP32
    int a_dim,
    int NN,           // N * N
    int batch_size)
{
    // Index wrappers
    IndexWrapper<const __nv_bfloat16, 2> A_t_idx(A_t, batch_size, a_dim);
    IndexWrapper<const __nv_bfloat16, 2> grad_W_eff_idx(grad_W_eff, batch_size, NN);
    IndexWrapper<float, 2> grad_Wa_idx(grad_Wa, a_dim, NN);

    // Block indices
    const int global_m_base = blockIdx.y * BM;
    const int global_n_base = blockIdx.x * BN;

    if (global_m_base >= a_dim || global_n_base >= NN) return;

    // Warp indices within block
    const size_t warp_idx_x = threadIdx.x / WARPSIZE;
    const size_t warp_idx_y = threadIdx.y;

    // Shared memory tiles (with padding to avoid bank conflicts)
    constexpr int ld = BK + 8;
    __shared__ __nv_bfloat16 tileA[BM][ld];  // A_t transposed: [BM, BK]
    __shared__ __nv_bfloat16 tileB[BN][ld];  // grad_W_eff: [BK, BN] transposed for col-major load

    // Accumulator fragment for this warp's tile
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over batch dimension in chunks of BK
    for (int k_base = 0; k_base < batch_size; k_base += BK) {

        // Load tile A: Coalesced read from A_t[batch, a_dim], transpose to tileA[BM, BK]
        // Each thread reads a row from A_t and writes to a column in tileA
        {
            const int total_threads = blockDim.x * blockDim.y;
            const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
            const int num_loads = (BM * BK) / 4;  // Load 4 BF16s per iteration

            for (int idx = thread_id; idx < num_loads; idx += total_threads) {
                int load_k = idx / (BM / 4);              // Which K element in BK
                int load_m_base = (idx % (BM / 4)) * 4;   // Which M elements in BM

                int global_k = k_base + load_k;

                if (global_k < batch_size) {
                    // Coalesced read: read row from A_t[global_k, global_m_base:global_m_base+4]
                    int global_m = global_m_base + load_m_base;
                    const __nv_bfloat16* src_ptr = &A_t_idx.at(global_k, global_m);

                    // Read 4 consecutive BF16s (coalesced)
                    __nv_bfloat162 data0 = *reinterpret_cast<const __nv_bfloat162*>(src_ptr);
                    __nv_bfloat162 data1 = *reinterpret_cast<const __nv_bfloat162*>(src_ptr + 2);

                    // Transpose write: write to column in shared memory
                    tileA[load_m_base][load_k] = data0.x;
                    tileA[load_m_base + 1][load_k] = data0.y;
                    tileA[load_m_base + 2][load_k] = data1.x;
                    tileA[load_m_base + 3][load_k] = data1.y;
                } else {
                    tileA[load_m_base][load_k] = __float2bfloat16(0.0f);
                    tileA[load_m_base + 1][load_k] = __float2bfloat16(0.0f);
                    tileA[load_m_base + 2][load_k] = __float2bfloat16(0.0f);
                    tileA[load_m_base + 3][load_k] = __float2bfloat16(0.0f);
                }
            }
        }

        // Load tile B: Coalesced read from grad_W_eff[batch, NN], transpose to tileB[BN, BK]
        {
            const int total_threads = blockDim.x * blockDim.y;
            const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
            const int num_loads = (BK * BN) / 4;

            for (int idx = thread_id; idx < num_loads; idx += total_threads) {
                int load_k = idx / (BN / 4);              // Which K element in BK
                int load_n_base = (idx % (BN / 4)) * 4;   // Which N elements in BN

                int global_k = k_base + load_k;

                if (global_k < batch_size) {
                    // Coalesced read: read row from grad_W_eff[global_k, global_n_base:global_n_base+4]
                    int global_n = global_n_base + load_n_base;
                    const __nv_bfloat16* src_ptr = &grad_W_eff_idx.at(global_k, global_n);

                    // Read 4 consecutive BF16s (coalesced)
                    __nv_bfloat162 data0 = *reinterpret_cast<const __nv_bfloat162*>(src_ptr);
                    __nv_bfloat162 data1 = *reinterpret_cast<const __nv_bfloat162*>(src_ptr + 2);

                    // Transpose write: write to column in shared memory
                    tileB[load_n_base][load_k] = data0.x;
                    tileB[load_n_base + 1][load_k] = data0.y;
                    tileB[load_n_base + 2][load_k] = data1.x;
                    tileB[load_n_base + 3][load_k] = data1.y;
                } else {
                    tileB[load_n_base][load_k] = __float2bfloat16(0.0f);
                    tileB[load_n_base + 1][load_k] = __float2bfloat16(0.0f);
                    tileB[load_n_base + 2][load_k] = __float2bfloat16(0.0f);
                    tileB[load_n_base + 3][load_k] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();

        // Perform WMMA operations on tiles
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_a;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> frag_b;

        #pragma unroll
        for (int mma_k = 0; mma_k < BK; mma_k += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &tileA[warp_idx_y * WMMA_M][mma_k], ld);
            wmma::load_matrix_sync(frag_b, &tileB[warp_idx_x * WMMA_N][mma_k], ld);
            wmma::mma_sync(acc_frag, frag_a, frag_b, acc_frag);
        }

        __syncthreads();
    }

    // Store result to grad_Wa with accumulation
    int frag_m = global_m_base + warp_idx_y * WMMA_M;
    int frag_n = global_n_base + warp_idx_x * WMMA_N;

    // Bound check before storing
    if (frag_m < a_dim && frag_n < NN) {
        // Load existing values
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> existing_frag;
        float* grad_Wa_ptr = &grad_Wa_idx.at(frag_m, frag_n);
        wmma::load_matrix_sync(existing_frag, grad_Wa_ptr, NN, wmma::mem_row_major);

        // Accumulate
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            acc_frag.x[i] += existing_frag.x[i];
        }

        // Store back
        wmma::store_matrix_sync(grad_Wa_ptr, acc_frag, NN, wmma::mem_row_major);
    }
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
    // Temporary buffers
    // -----------------------------------------------------------------
    float *grad_r, *W_eff_temp, *grad_re_temp;
    __nv_bfloat16 *grad_W_eff_temp_bf16, *A_t_temp_bf16;
    
    cudaMalloc(&grad_r,        batch_size * N * sizeof(float));
    cudaMalloc(&W_eff_temp,   batch_size * N * N * sizeof(float));
    cudaMalloc(&grad_re_temp, batch_size * N * sizeof(float));
    cudaMalloc(&grad_W_eff_temp_bf16, batch_size * N * N * sizeof(__nv_bfloat16));
    cudaMalloc(&A_t_temp_bf16,     batch_size * a_dim * sizeof(__nv_bfloat16));

    cudaMemset(grad_r, 0, batch_size * N * sizeof(float));

    // -----------------------------------------------------------------
    // cuBLAS handle with tensor core math mode
    // -----------------------------------------------------------------
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Enable tensor core operations
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // -----------------------------------------------------------------
    // Tile sizes
    // -----------------------------------------------------------------
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Kernel 4a tile sizes
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int THREAD_TILE_M = 4;
    constexpr int THREAD_TILE_N = 4;

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
                W_eff_temp, grad_re_temp, A_t_temp_bf16, grad_W_eff_temp_bf16,
                alpha, N, a_dim, seq_len, t, batch_size);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // -------------------------------------------------------------
        // Kernel 3 – grad_r
        // -------------------------------------------------------------
        dim3 block3(128);
        dim3 grid3(batch_size, (N + block3.x - 1) / block3.x);
        compute_grad_r_kernel<<<grid3, block3>>>(
            W_eff_temp, grad_re_temp, grad_r,
            alpha, N, batch_size);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // -------------------------------------------------------------
        // Kernel 4a – Custom batched outer product with BF16 output
        // grad_W_eff[b, i, j] = grad_re[b, i] * r_prev[b, j]
        // -------------------------------------------------------------
        const float* r_prev = bump_history + t * batch_size * N;
        
        dim3 block4a(TILE_N / THREAD_TILE_N, TILE_M / THREAD_TILE_M);
        dim3 grid4a((N + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M, batch_size);
        
        batched_outer_product_bf16_kernel<TILE_M, TILE_N, THREAD_TILE_M, THREAD_TILE_N>
            <<<grid4a, block4a>>>(
                grad_re_temp, r_prev, grad_W_eff_temp_bf16, N, batch_size);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // -------------------------------------------------------------
        // Kernel 4b – grad_Wo += J1 * grad_re.T @ r_prev
        // GEMM: [N, B] @ [B, N] = [N, N]
        // Using FP32 GEMM for this smaller reduction
        // -------------------------------------------------------------
        float one = 1.0f;
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

        // -------------------------------------------------------------
        // Kernel 5 – WMMA-based grad_Wa computation
        // -------------------------------------------------------------
        constexpr int K5_BM = 64;
        constexpr int K5_BN = 64;
        constexpr int K5_BK = 64;
        constexpr int K5_WMMA_M = 16;
        constexpr int K5_WMMA_N = 16;
        constexpr int K5_WMMA_K = 16;

        dim3 block5((K5_BN / K5_WMMA_N) * WARPSIZE, K5_BM / K5_WMMA_M);
        dim3 grid5(
            (N * N + K5_BN - 1) / K5_BN,  // columns of grad_Wa
            (a_dim + K5_BM - 1) / K5_BM   // rows of grad_Wa
        );

        compute_grad_Wa_wmma_kernel<K5_BM, K5_BN, K5_BK, K5_WMMA_M, K5_WMMA_N, K5_WMMA_K>
            <<<grid5, block5>>>(
                A_t_temp_bf16,
                grad_W_eff_temp_bf16,
                grad_Wa,
                a_dim,
                N * N,
                batch_size);

        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    // -----------------------------------------------------------------
    // Clean‑up
    // -----------------------------------------------------------------
    cublasDestroy(handle);

    cudaFree(grad_r);
    cudaFree(W_eff_temp);
    cudaFree(grad_re_temp);
    cudaFree(grad_W_eff_temp_bf16);
    cudaFree(A_t_temp_bf16);
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