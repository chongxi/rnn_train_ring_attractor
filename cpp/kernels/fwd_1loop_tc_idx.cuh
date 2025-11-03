// #include "cuda_common.cuh"
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
__global__ void __launch_bounds__((BN / WMMA_N) * WARPSIZE * (BM / WMMA_M)) fwd_wmma_kernel(
    const T* A_t,    //[batch_size, a_dim]
    const T* Wa_16b, //[a_dim, n_neur, n_neur] = [64, 128, 128]
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
    Tensor gA_t = make_tensor(A_t, make_layout(make_shape(batch_size, a_dim), LayoutRight{}));
    Tensor gWa = make_tensor(Wa_16b, make_layout(make_shape(a_dim, n_neur, n_neur), LayoutRight{}));
    Tensor gWo = make_tensor(Wo, make_layout(make_shape(n_neur, n_neur), LayoutRight{}));
    Tensor gR_innit = make_tensor(r_init, make_layout(make_shape(batch_size, n_neur), LayoutRight{}));
    Tensor gBump_hist = make_tensor(bump_history, make_layout(make_shape(seq_len, batch_size, n_neur), LayoutRight{}));

    auto cta = cg::this_thread_block();

    const size_t warp_idx_x = threadIdx.x / WARPSIZE;
    const size_t warp_idx_y = threadIdx.y;

    const int global_m_base = blockIdx.y * BM;
    const int global_n_neuron = blockIdx.x;

    if (global_m_base >= batch_size || global_n_neuron >= n_neur) return;

    bool is_first = (t == 0);

    constexpr int ld = BK + 8;
    constexpr int lda = ld;
    constexpr int ldb = ld;

    __shared__ T tileA[BM][ld];
    __shared__ T tileB[BN][ld];

    auto sA = make_tensor(make_smem_ptr(tileA),
        make_layout(make_shape(Int<BM>{}, Int<BK>{}), make_stride(Int<ld>{}, Int<1>{})));

    auto sB = make_tensor(make_smem_ptr(tileB),
        make_layout(make_shape(Int<BK>{}, Int<BN>{}), make_stride(Int<1>{}, Int<ld>{})));

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
                        const T* src_ptr = &gA_t(global_m, global_k);
                        uint32_t data = *reinterpret_cast<const uint32_t*>(src_ptr);

                        tileA[load_m][load_k_base] = reinterpret_cast<const T*>(&data)[0];
                        tileA[load_m][load_k_base + 1] = reinterpret_cast<const T*>(&data)[1];
                    } else {
                        tileA[load_m][load_k_base] = cuda_cast<T>(0.f);
                        tileA[load_m][load_k_base + 1] = cuda_cast<T>(0.f);
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

                    if (global_k < a_dim && global_n + 1 < n_neur) {
                        const T* src_ptr = &gWa(global_k, global_n_neuron, global_n);
                        uint32_t data = *reinterpret_cast<const uint32_t*>(src_ptr);

                        tileB[load_n_base][load_k] = reinterpret_cast<const T*>(&data)[0];
                        tileB[load_n_base + 1][load_k] = reinterpret_cast<const T*>(&data)[1];
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
        int frag_n = cta_n + warp_idx_x * WMMA_N;
        if (frag_n < n_neur) {
            const float* pos_Wo = &gWo(global_n_neuron, frag_n);
            calc_Weff<WMMA_M, WMMA_N, WMMA_K, T>(frag_Wa_weighted, pos_Wo, J0, J1);
        }

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_r;

        //============================= Step 3: Load r[BM, BN] tile and perform partial sum ============================
        int frag_m = global_m_base + warp_idx_y * WMMA_M;
        if (frag_m < batch_size && frag_n < n_neur) {
            const float* pos_frag_r = is_first ?
                &gR_innit(frag_m, frag_n) :
                &gBump_hist(t - 1, frag_m, frag_n);

            wmma::load_matrix_sync(frag_r, pos_frag_r, n_neur, wmma::mem_row_major);

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

    //========================= Step 5: Update r and save to bump_history ==============================================
    if (cta.thread_rank() < BM) {
        int global_m = global_m_base + cta.thread_rank();
        if (global_m < batch_size && global_n_neuron < n_neur) {
            float r_prev = is_first ?
                gR_innit(global_m, global_n_neuron) :
                gBump_hist(t - 1, global_m, global_n_neuron);
            float r_updated = (1 - alpha) * r_prev + alpha * activation<ACT>(re_temp[cta.thread_rank()][0]);
            gBump_hist(t, global_m, global_n_neuron) = r_updated;
        }
    }
}

// Kernel to convert a slice of A at time t from float to half/bfloat16
template<typename T>
__global__ void cast_b32_b16_A_t(
    const float* A,          // [batch_size, seq_len, a_dim]
    T* A_t,                  // [batch_size, a_dim]
    int batch_size,
    int a_dim,
    int seq_len,
    int t
) {
    int total_elements = batch_size * a_dim;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Process 4 floats (becomes 4 halfs/bfloat16s) per iteration
    for (size_t idx = tid * 4; idx < total_elements; idx += stride * 4) {
        size_t batch_idx = idx / a_dim;
        size_t a_idx = idx % a_dim;

        // Load 4 floats from global memory
        const float* src_ptr = &A[batch_idx * (seq_len * a_dim) + t * a_dim + a_idx];
        float4 data = *reinterpret_cast<const float4*>(src_ptr);

        // Convert each element individually
        T* dst_ptr = &A_t[batch_idx * a_dim + a_idx];
        dst_ptr[0] = cuda_cast<T>(data.x);
        dst_ptr[1] = cuda_cast<T>(data.y);
        dst_ptr[2] = cuda_cast<T>(data.z);
        dst_ptr[3] = cuda_cast<T>(data.w);
    }
}

// Kernel to convert Wa from float to half/bfloat16
template<typename T>
__global__ void cast_b32_b16_Wa(
    const float* Wa_32b,     // [a_dim, N, N]
    T* Wa_16b,               // [a_dim, N, N]
    int a_dim,
    int N
) {
    int total_elements = a_dim * N * N;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Process 4 floats (becomes 4 halfs/bfloat16s) per iteration
    for (size_t idx = tid * 4; idx < total_elements; idx += stride * 4) {
        // Load 4 floats from global memory
        const float* src_ptr = &Wa_32b[idx];
        float4 data = *reinterpret_cast<const float4*>(src_ptr);

        // Convert each element individually
        T* dst_ptr = &Wa_16b[idx];
        dst_ptr[0] = cuda_cast<T>(data.x);
        dst_ptr[1] = cuda_cast<T>(data.y);
        dst_ptr[2] = cuda_cast<T>(data.z);
        dst_ptr[3] = cuda_cast<T>(data.w);
    }
}

template<Activation ACT, typename T>
void fwd_wmma_impl(
    void* A,
    void* Wa,
    float J0,
    float J1,
    void* Wo,
    void* r_init,
    void* bump_history,
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

    dim3 blockSize((BN / WMMA_N) * WARPSIZE, BM / WMMA_M);
    dim3 gridSize(N, (batch_size + BM - 1) / BM);

    // Allocate 16-bit buffers
    T* Wa_16b;
    T* A_t;

    size_t Wa_size = a_dim * N * N * sizeof(T);
    size_t A_t_size = batch_size * a_dim * sizeof(T);

    cudaMalloc(&Wa_16b, Wa_size);
    cudaMalloc(&A_t, A_t_size);

    // Convert Wa once (outside the time loop)
    {
        int total_elements = a_dim * N * N;
        int threads = 256;
        int blocks = (total_elements + (threads * 4) - 1) / (threads * 4);

        cast_b32_b16_Wa<T><<<blocks, threads>>>(
            static_cast<const float*>(Wa),
            Wa_16b,
            a_dim,
            N
        );
        cudaDeviceSynchronize();
    }

    // Time loop
    for (int t = 0; t < seq_len; ++t) {
        // Convert A slice for time t
        {
            int total_elements = batch_size * a_dim;
            int threads = 256;
            int blocks = (total_elements + (threads * 4) - 1) / (threads * 4);

            cast_b32_b16_A_t<T><<<blocks, threads>>>(
                static_cast<const float*>(A),
                A_t,
                batch_size,
                a_dim,
                seq_len,
                t
            );
            cudaDeviceSynchronize();
        }

        // Launch main kernel with 16-bit inputs
        fwd_wmma_kernel<ACT, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, T><<<gridSize, blockSize>>>(
            A_t,
            Wa_16b,
            J0, J1,
            static_cast<const float*>(Wo),
            static_cast<const float*>(r_init),
            static_cast<float*>(bump_history),
            alpha, N, a_dim, seq_len, t, batch_size
        );
        cudaDeviceSynchronize();
    }

    // Free allocated memory
    cudaFree(Wa_16b);
    cudaFree(A_t);
}

void fwd_wmma_launcher(
    void* A,
    void* Wa,
    float J0,
    float J1,
    void* Wo,
    void* r_init,
    // void* W_delta7,
    void* bump_history,
    // void* r_history,
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
            case 0: fwd_wmma_impl<Activation::RELU, nv_bfloat16>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 1: fwd_wmma_impl<Activation::GELU, nv_bfloat16>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 2: fwd_wmma_impl<Activation::TANH, nv_bfloat16>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 3: fwd_wmma_impl<Activation::SILU, nv_bfloat16>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
        }
    } else {
        switch(activation_type){
            case 0: fwd_wmma_impl<Activation::RELU, half>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 1: fwd_wmma_impl<Activation::GELU, half>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 2: fwd_wmma_impl<Activation::TANH, half>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 3: fwd_wmma_impl<Activation::SILU, half>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
        }
    }
}

