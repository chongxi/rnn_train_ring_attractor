#include "cuda_common.cuh"
#include "../cuda_common.cuh"



template<int WMMA_M, int WMMA_N, int WMMA_K, typename T>
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

template<Activation ACT, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, typename T>
__global__ void __launch_bounds__((BN / WMMA_N) * WARPSIZE * (BM / WMMA_M)) fwd_update_r_kernel(
    const float* A,  //[batch_size, seq_len, a_dim]) = [256, 10, 64] TODO: Transposing for better L2 cache hit
    const float* Wa, //[a_dim, n_neur, n_neur] = [64, 128, 128]
    float J0,
    float J1,
    const float* Wo, //[n_neur, n_neur] = [128, 128] TODO: Make Wo persistent in L2 cache. REVIEW: already fits inside L2
    const float* r_init, // [batch_size, n_neur] = [256, 128]
    float* bump_history, // [batch_size, seq_len, n_neur] -> [seq_len, batch_size, n_neur] = [10, 256, 128]
    float alpha,
    int n_neur,
    int a_dim,
    int seq_len,
    int t,
    int batch_size
){
    IndexWrapper<const float, 3> A_idx(A, batch_size, seq_len, a_dim);
    IndexWrapper<const float, 3> Wa_idx(Wa, a_dim, n_neur, n_neur);
    IndexWrapper<const float, 2> Wo_idx(Wo, n_neur, n_neur);
    IndexWrapper<const float, 2> r_init_idx(r_init, batch_size, n_neur);
    IndexWrapper<float, 3> bump_history_idx(bump_history, seq_len, batch_size, n_neur);

    // constexpr int group_size = 16;
    auto cta = cg::this_thread_block();
    // auto group = cg::tiled_partition<group_size>(cta);
    // size_t gr_x = group.thread_rank(); // 0 -> 15
    // size_t gr_y = group.meta_group_rank(); // 0 -> 31

    const size_t warp_idx_x = threadIdx.x / WARPSIZE;
    const size_t warp_idx_y = threadIdx.y;

    const int global_m_base = blockIdx.y * BM;
    const int global_n_neuron = blockIdx.x;

    if (global_m_base >= batch_size || global_n_neuron >= n_neur) return;

    bool is_first = (t == 0);

    constexpr int ld = BK + 8;
    constexpr int lda = ld;
    constexpr int ldb = ld;

    // TODO: Increase size of tileA to [BN * n + 8][ld] for better smem reuse ???
    __shared__ T tileA[BM][ld];
    __shared__ T tileB[BN][ld];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_re;
    wmma::fill_fragment(frag_re, 0.f);

    for (int cta_n = 0; cta_n < n_neur; cta_n += BN) {

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

                        tileA[load_m][load_k_base] = cuda_cast<T>(data.x);
                        tileA[load_m][load_k_base + 1] = cuda_cast<T>(data.y);
                    } else {
                        tileA[load_m][load_k_base] = cuda_cast<T>(0.f);
                        tileA[load_m][load_k_base + 1] = cuda_cast<T>(0.f);
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

                    if (global_k < a_dim && global_n + 1 < n_neur) {
                        const float* src_ptr = &Wa_idx.at(global_k, global_n_neuron, global_n);
                        float2 data = *reinterpret_cast<const float2*>(src_ptr);

                        tileB[load_n_base][load_k] = cuda_cast<T>(data.x);
                        tileB[load_n_base + 1][load_k] = cuda_cast<T>(data.y);
                    } else {
                        tileB[load_n_base][load_k] = cuda_cast<T>(0.f);
                        tileB[load_n_base + 1][load_k] = cuda_cast<T>(0.f);
                    }
                }
            }

            __syncthreads();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> frag_a;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major> frag_b;

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
        if (frag_n < n_neur) {
            const float* pos_Wo = &Wo_idx.at(global_n_neuron, frag_n);
            calc_Weff<WMMA_M, WMMA_N, WMMA_K, T>(frag_Wa_weighted, pos_Wo, J0, J1);
        }

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_r;

        //============================= Step 3: Load r[BM, BN] tile and perform partial sum ============================
        int frag_m = global_m_base + warp_idx_y * WMMA_M;
        if (frag_m < batch_size && frag_n < n_neur) {
            const float* pos_frag_r = is_first ?
                &r_init_idx.at(frag_m, frag_n) :
                &bump_history_idx.at(t - 1, frag_m, frag_n);

            wmma::load_matrix_sync(frag_r, pos_frag_r, n_neur, wmma::mem_row_major);

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
        }
    }

    __syncthreads();

    //========================= Step 5: Update r and save to bump_history ==============================================
    if (cta.thread_rank() < BM) {
        int global_m = global_m_base + cta.thread_rank();
        if (global_m < batch_size && global_n_neuron < n_neur) {
            float r_prev = is_first ?
                r_init_idx.at(global_m, global_n_neuron) :
                bump_history_idx.at(t - 1, global_m, global_n_neuron);
            float r_updated = (1 - alpha) * r_prev + alpha * activation<ACT>(re_temp[cta.thread_rank()][0]);
            bump_history_idx.at(t, global_m, global_n_neuron) = r_updated;
        }
    }
}

template<Activation ACT, typename T>
void fwd_n128_a23_global_launcher_impl(
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
    int batch_size
){
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // constexpr int BLOCKSIZE = (BM / WMMA_M) * (BN / WMMA_N) * WARPSIZE;

    dim3 blockSize((BN / WMMA_N) * WARPSIZE, BM / WMMA_M);

    // TODO: Make blockDix.x = N / SPLIT, meaning each block calculates SPLIT rows in r, we can reuse batches of r in the for loop
    // Currently, each block will calculate just a single row in the resulted r in batches.
    dim3 gridSize(N, (batch_size + BM - 1) / BM);

    for (int t = 0; t < seq_len; ++t){
        fwd_update_r_kernel<ACT, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, T><<<gridSize, blockSize>>>(
            static_cast<const float*>(A),
            static_cast<const float*>(Wa),
            J0, J1,
            static_cast<const float*>(Wo),
            static_cast<const float*>(r_init),
            static_cast<float*>(bump_history),
            alpha, N, a_dim, seq_len, t, batch_size
        );
    }
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
    constexpr bool use_bf16 = false;
    if (use_bf16) {
        switch(activation_type){
            case 0: fwd_n128_a23_global_launcher_impl<Activation::RELU, nv_bfloat16>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 1: fwd_n128_a23_global_launcher_impl<Activation::GELU, nv_bfloat16>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 2: fwd_n128_a23_global_launcher_impl<Activation::TANH, nv_bfloat16>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 3: fwd_n128_a23_global_launcher_impl<Activation::SILU, nv_bfloat16>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
        }
    } else {
        switch(activation_type){
            case 0: fwd_n128_a23_global_launcher_impl<Activation::RELU, half>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 1: fwd_n128_a23_global_launcher_impl<Activation::GELU, half>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 2: fwd_n128_a23_global_launcher_impl<Activation::TANH, half>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 3: fwd_n128_a23_global_launcher_impl<Activation::SILU, half>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
        }
    }
}

