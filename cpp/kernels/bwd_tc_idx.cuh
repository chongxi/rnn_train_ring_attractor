// #include "cuda_common.cuh"
#include "../cuda_common.cuh"



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

// Kernel 1: Recompute W_eff and re for timestep t (FP16 GEMM)
template<Activation ACT, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void __launch_bounds__((BN / WMMA_N) * WARPSIZE * (BM / WMMA_M))
recompute_Weff_re_kernel(
    const float* A,              // [batch_size, seq_len, a_dim]
    const float* Wa,             // [a_dim, N, N]
    float J0,             // [N, N]
    float J1,
    const float* Wo,             // [N, N]
    const float* bump_history,   // [seq_len+1, batch_size, N]
    float* W_eff_out,            // [batch_size, N, N] - temp output
    float* re_out,               // [batch_size, N] - temp output
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int t,
    int batch_size
){
    IndexWrapper<const float, 3> A_idx(A, batch_size, seq_len, a_dim);
    IndexWrapper<const float, 3> Wa_idx(Wa, a_dim, N, N);
    IndexWrapper<const float, 2> Wo_idx(Wo, N, N);
    IndexWrapper<const float, 3> bump_history_idx(bump_history, seq_len+1, batch_size, N);
    IndexWrapper<float, 3> W_eff_idx(W_eff_out, batch_size, N, N);
    IndexWrapper<float, 2> re_idx(re_out, batch_size, N);

    auto cta = cg::this_thread_block();

    const size_t warp_idx_x = threadIdx.x / WARPSIZE;
    const size_t warp_idx_y = threadIdx.y;

    const int global_m_base = blockIdx.y * BM;
    const int global_n_neuron = blockIdx.x;

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

            {
                const int total_threads = blockDim.x * blockDim.y;
                const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
                const int num_loads = (BK * BN) / 2;

                for (int idx = thread_id; idx < num_loads; idx += total_threads) {
                    int load_k = idx / (BN / 2);
                    int load_n_base = (idx % (BN / 2)) * 2;

                    int global_k = cta_k + load_k;
                    int global_n = cta_n + load_n_base;

                    if (global_k < a_dim && global_n + 1 < N) {
                        const float* src_ptr = &Wa_idx.at(global_k, global_n_neuron, global_n);
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
        }

        int frag_n = cta_n + warp_idx_x * WMMA_N;
        const float* pos_Wo = &Wo_idx.at(global_n_neuron, frag_n);
        calc_Weff<WMMA_M, WMMA_N, WMMA_K>(frag_Wa_weighted, pos_Wo, J0, J1);

        int frag_m = global_m_base + warp_idx_y * WMMA_M;
        float* pos_Weff = &W_eff_idx.at(frag_m, global_n_neuron, frag_n);
        wmma::store_matrix_sync(pos_Weff, frag_Wa_weighted, N, wmma::mem_row_major);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_r;
        const float* pos_frag_r = &bump_history_idx.at(t, frag_m, frag_n);
        wmma::load_matrix_sync(frag_r, pos_frag_r, N, wmma::mem_row_major);

        for (size_t i = 0; i < frag_r.num_elements; ++i) {
            frag_re.x[i] += frag_Wa_weighted.x[i] * frag_r.x[i];
        }
    }

    constexpr int ldtemp = BN + 2;
    __shared__ float re_temp[BM][ldtemp];

    wmma::store_matrix_sync(
        &re_temp[warp_idx_y * WMMA_M][warp_idx_x * WMMA_N],
        frag_re, ldtemp, wmma::mem_row_major);

    __syncthreads();

    auto warp_group = cg::tiled_partition<WARPSIZE>(cta);
    size_t lane_id = warp_group.thread_rank();
    size_t warp_id = warp_group.meta_group_rank();

    for (size_t warp_m = 0; warp_m < BM; warp_m += warp_group.meta_group_size()) {
        int global_m = global_m_base + warp_m + warp_id;

        float re_sum = 0.f;
        for (int warp_n = 0; warp_n < BN; warp_n += WARPSIZE) {
            re_sum += re_temp[warp_m + warp_id][warp_n + lane_id];
        }
        float re_reduced = cg::reduce(warp_group, re_sum, cg::plus<float>());

        if (lane_id == 0) {
            re_idx.at(global_m, global_n_neuron) = re_reduced;
        }
    }
}


// Kernel 2: Compute grad_re from grad_output and grad_r
template<Activation ACT>
__global__ void compute_grad_re_kernel(
    const float* grad_output,    // [seq_len, batch_size, N]
    float* grad_r,               // [batch_size, N] - in/out
    const float* re,             // [batch_size, N]
    float* grad_re,              // [batch_size, N] - output
    float alpha,
    int N,
    int seq_len,
    int t,
    int batch_size
){
    IndexWrapper<const float, 3> grad_output_idx(grad_output, seq_len, batch_size, N);
    IndexWrapper<float, 2> grad_r_idx(grad_r, batch_size, N);
    IndexWrapper<const float, 2> re_idx(re, batch_size, N);
    IndexWrapper<float, 2> grad_re_idx(grad_re, batch_size, N);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * N;

    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int batch_idx = i / N;
        int n_idx = i % N;

        float grad_out = grad_output_idx.at(t, batch_idx, n_idx);
        float grad_r_val = grad_r_idx.at(batch_idx, n_idx);

        float grad_r_total = grad_r_val + grad_out;
        float grad_recurrent_activated = grad_r_total * alpha;

        float re_val = re_idx.at(batch_idx, n_idx);
        float grad_re_val = activation_derivative<ACT>(re_val) * grad_recurrent_activated;

        grad_re_idx.at(batch_idx, n_idx) = grad_re_val;
    }
}


// Kernel 3: Compute grad_r from W_eff^T @ grad_re (Batched GEMV)
__global__ void compute_grad_r_kernel(
    const float* W_eff,          // [batch_size, N, N]
    const float* grad_re,        // [batch_size, N]
    float* grad_r,               // [batch_size, N] - in/out
    float alpha,
    int N,
    int batch_size
){
    IndexWrapper<const float, 3> W_eff_idx(W_eff, batch_size, N, N);
    IndexWrapper<const float, 2> grad_re_idx(grad_re, batch_size, N);
    IndexWrapper<float, 2> grad_r_idx(grad_r, batch_size, N);

    int batch_idx = blockIdx.x;
    int n_out = blockIdx.y;

    auto cta = cg::this_thread_block();
    auto tile = cg::tiled_partition<128>(cta);

    float sum = 0.f;
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        sum += W_eff_idx.at(batch_idx, k, n_out) * grad_re_idx.at(batch_idx, k);
    }

    sum = cg::reduce(tile, sum, cg::plus<float>());

    if (threadIdx.x == 0) {
        float old_grad_r = grad_r_idx.at(batch_idx, n_out);
        float new_grad_r = sum + old_grad_r * (1.0f - alpha);
        grad_r_idx.at(batch_idx, n_out) = new_grad_r;
    }
}


// Kernel 4a: Compute grad_W_eff only
__global__ void compute_grad_Weff_kernel(
    const float* grad_re,        // [batch_size, N]
    const float* r_prev,         // [seq_len+1, batch_size, N]
    __nv_bfloat16* grad_W_eff,   // [batch_size, N, N] - output in bf16
    int N,
    int seq_len,
    int t,
    int batch_size
){
    IndexWrapper<const float, 2> grad_re_idx(grad_re, batch_size, N);
    IndexWrapper<const float, 3> r_prev_idx(r_prev, seq_len+1, batch_size, N);
    IndexWrapper<__nv_bfloat16, 3> grad_W_eff_idx(grad_W_eff, batch_size, N, N);

    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && batch_idx < batch_size) {
        float grad_re_val = grad_re_idx.at(batch_idx, row);
        float r_val = r_prev_idx.at(t, batch_idx, col);
        float grad_W_eff_val = grad_re_val * r_val;

        grad_W_eff_idx.at(batch_idx, row, col) = __float2bfloat16(grad_W_eff_val);
    }
}


// Kernel 4b: Reduce grad_W_eff over batch and accumulate to grad_Wo
__global__ void accumulate_grad_Wo_kernel(
    const __nv_bfloat16* grad_W_eff, // [batch_size, N, N]
    float* grad_Wo,                  // [N, N] - accumulated
    float J1,
    int N,
    int batch_size
){
    IndexWrapper<const __nv_bfloat16, 3> grad_W_eff_idx(grad_W_eff, batch_size, N, N);
    IndexWrapper<float, 2> grad_Wo_idx(grad_Wo, N, N);

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.f;
        for (int b = 0; b < batch_size; ++b) {
            __nv_bfloat16 val = grad_W_eff_idx.at(b, row, col);
            sum += __bfloat162float(val);
        }

        grad_Wo_idx.at(row, col) += J1 * sum;
    }
}


// Kernel 5: Compute grad_Wa using tensor cores (BF16 GEMM)
template<int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void __launch_bounds__((BN / WMMA_N) * WARPSIZE * (BM / WMMA_M))
compute_grad_Wa_kernel(
    const float* A_t,            // [seq_len, batch_size, a_dim]
    const __nv_bfloat16* grad_W_eff, // [batch_size, N, N]
    float* grad_Wa_flat,         // [a_dim, N*N] - flattened last 2 dims
    int N,
    int a_dim,
    int seq_len,
    int t,
    int batch_size
){
    IndexWrapper<const float, 3> A_idx(A_t, batch_size, seq_len, a_dim);
    IndexWrapper<const __nv_bfloat16, 3> grad_W_eff_idx(grad_W_eff, batch_size, N, N);
    IndexWrapper<float, 2> grad_Wa_flat_idx(grad_Wa_flat, a_dim, N * N);

    auto cta = cg::this_thread_block();

    const size_t warp_idx_x = threadIdx.x / WARPSIZE;
    const size_t warp_idx_y = threadIdx.y;

    const int global_m_base = blockIdx.y * BM;
    const int global_n_base = blockIdx.x * BN;

    constexpr int ld = BK + 8;
    constexpr int lda = ld;
    constexpr int ldb = ld;

    __shared__ nv_bfloat16 tileA[BM][ld];
    __shared__ nv_bfloat16 tileB[BN][ld];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc;
    wmma::fill_fragment(frag_acc, 0.f);

    int N_sq = N * N;

    for (int cta_k = 0; cta_k < batch_size; cta_k += BK) {
        {
            const int total_threads = blockDim.x * blockDim.y;
            const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
            const int num_loads = (BM * BK) / 2;

            for (int idx = thread_id; idx < num_loads; idx += total_threads) {
                int load_m = idx / (BK / 2);
                int load_k_base = (idx % (BK / 2)) * 2;

                int global_m = global_m_base + load_m;
                int global_k = cta_k + load_k_base;

                const float* src_ptr = &A_idx.at(global_k, t, global_m);
                float val0 = *src_ptr;
                float val1 = *(src_ptr + a_dim);

                tileA[load_m][load_k_base] = __float2bfloat16(val0);
                tileA[load_m][load_k_base + 1] = __float2bfloat16(val1);
            }
        }

        {
            const int total_threads = blockDim.x * blockDim.y;
            const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
            const int num_loads = (BK * BN) / 2;

            for (int idx = thread_id; idx < num_loads; idx += total_threads) {
                int load_k = idx / (BN / 2);
                int load_n_base = (idx % (BN / 2)) * 2;

                int global_k = cta_k + load_k;
                int global_n = global_n_base + load_n_base;

                int row0 = global_n / N;
                int col0 = global_n % N;
                int row1 = (global_n + 1) / N;
                int col1 = (global_n + 1) % N;

                __nv_bfloat16 val0 = grad_W_eff_idx.at(global_k, row0, col0);
                __nv_bfloat16 val1 = grad_W_eff_idx.at(global_k, row1, col1);

                tileB[load_n_base][load_k] = val0;
                tileB[load_n_base + 1][load_k] = val1;
            }
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, nv_bfloat16, wmma::row_major> frag_a;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, nv_bfloat16, wmma::row_major> frag_b;

#pragma unroll
        for (int mma_k = 0; mma_k < BK; mma_k += WMMA_K) {
            wmma::load_matrix_sync(frag_a, &tileA[0][0] + (warp_idx_y * WMMA_M) * lda + mma_k, lda);
            wmma::load_matrix_sync(frag_b, &tileB[0][0] + (warp_idx_x * WMMA_N) * ldb + mma_k, ldb);
            wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        }

        __syncthreads();
    }

    // Store directly to global memory with accumulation
    int frag_m = global_m_base + warp_idx_y * WMMA_M;
    int frag_n = global_n_base + warp_idx_x * WMMA_N;

    if (frag_m < a_dim && frag_n < N_sq) {
        // Load existing, accumulate, store
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_existing;
        float* grad_Wa_ptr = &grad_Wa_flat_idx.at(frag_m, frag_n);

        wmma::load_matrix_sync(frag_existing, grad_Wa_ptr, N_sq, wmma::mem_row_major);

        for (int i = 0; i < frag_acc.num_elements; ++i) {
            frag_existing.x[i] += frag_acc.x[i];
        }

        wmma::store_matrix_sync(grad_Wa_ptr, frag_existing, N_sq, wmma::mem_row_major);
    }
}


// Implementation launcher
template<Activation ACT>
void bwd_wmma_impl(
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
    int batch_size
){
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    float* grad_r;
    float* W_eff_temp;
    float* re_temp;
    float* grad_re_temp;
    __nv_bfloat16* grad_W_eff_temp;

    cudaMalloc(&grad_r, batch_size * N * sizeof(float));
    cudaMalloc(&W_eff_temp, batch_size * N * N * sizeof(float));
    cudaMalloc(&re_temp, batch_size * N * sizeof(float));
    cudaMalloc(&grad_re_temp, batch_size * N * sizeof(float));
    cudaMalloc(&grad_W_eff_temp, batch_size * N * N * sizeof(__nv_bfloat16));

    cudaMemset(grad_r, 0, batch_size * N * sizeof(float));

    for (int t = seq_len - 1; t >= 0; --t) {
        // Kernel 1: Recompute W_eff and re
        {
            dim3 blockSize((BN / WMMA_N) * WARPSIZE, BM / WMMA_M);
            dim3 gridSize(N, (batch_size + BM - 1) / BM);

            recompute_Weff_re_kernel<ACT, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K><<<gridSize, blockSize>>>(
                A, Wa, J0, J1, Wo, bump_history, W_eff_temp, re_temp,
                alpha, N, a_dim, seq_len, t, batch_size
            );
        }

        // Kernel 2: Compute grad_re
        {
            int total_elements = batch_size * N;
            int blockSize = 256;
            int gridSize = (total_elements + blockSize - 1) / blockSize;

            compute_grad_re_kernel<ACT><<<gridSize, blockSize>>>(
                grad_output, grad_r, re_temp, grad_re_temp,
                alpha, N, seq_len, t, batch_size
            );
        }

        // Kernel 3: Compute grad_r (Batched GEMV)
        {
            dim3 gridSize(batch_size, N);
            int blockSize = 128;

            compute_grad_r_kernel<<<gridSize, blockSize>>>(
                W_eff_temp, grad_re_temp, grad_r,
                alpha, N, batch_size
            );
        }

        // Kernel 4a: Compute grad_W_eff
        {
            dim3 blockSize(16, 16);
            dim3 gridSize((N + 15) / 16, (N + 15) / 16, batch_size);

            compute_grad_Weff_kernel<<<gridSize, blockSize>>>(
                grad_re_temp, bump_history, grad_W_eff_temp,
                N, seq_len, t, batch_size
            );
        }

        // Kernel 4b: Accumulate grad_Wo
        {
            dim3 blockSize(16, 16);
            dim3 gridSize((N + 15) / 16, (N + 15) / 16);

            accumulate_grad_Wo_kernel<<<gridSize, blockSize>>>(
                grad_W_eff_temp, grad_Wo,
                J1, N, batch_size
            );
        }

        // Kernel 5: Compute grad_Wa
        {
            int N_sq = N * N;
            dim3 blockSize((BN / WMMA_N) * WARPSIZE, BM / WMMA_M);
            dim3 gridSize((N_sq + BN - 1) / BN, (a_dim + BM - 1) / BM);

            compute_grad_Wa_kernel<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K><<<gridSize, blockSize>>>(
                A, grad_W_eff_temp, grad_Wa,
                N, a_dim, seq_len, t, batch_size
            );
        }
    }

    cudaFree(grad_r);
    cudaFree(W_eff_temp);
    cudaFree(re_temp);
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
