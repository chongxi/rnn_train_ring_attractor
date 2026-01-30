// #include "cuda_common.cuh"
#include "../cuda_common.cuh"


namespace fwd_fp32 {
    template<Activation ACT, int BM, int BN, int BK>
    __global__ void __launch_bounds__(256) fwd_fp32_kernel(
        const float* A,
        const float* Wa,
        float J0,
        float J1,
        const float* Wo,
        const float* r_init,
        float* bump_history,
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

        const int tid = threadIdx.x;
        const int thread_m = tid / 32;
        const int thread_n = tid % 32;

        const int global_m_base = blockIdx.y * BM;
        const int global_n_neuron = blockIdx.x;

        if (global_n_neuron >= n_neur) return;

        bool is_first = (t == 0);

        __shared__ float tileA[BM][BK + 2];
        __shared__ float tileB[BN][BK + 2];

        float re_accum[4] = {0.f, 0.f, 0.f, 0.f};

        for (int cta_n = 0; cta_n < n_neur; cta_n += BN) {
            float Wa_weighted[4] = {0.f, 0.f, 0.f, 0.f};

            for (int cta_k = 0; cta_k < a_dim; cta_k += BK) {
                //===================== Step 1: Compute Wa_weighted[BM, BN] tile ===========================================
                {   // Load tileA
                    const int total_loads_A = BM * BK / 4;

                    if (tid < total_loads_A) {
                        int load_m = tid / (BK / 4);
                        int load_k_base = (tid % (BK / 4)) * 4;

                        int global_m = global_m_base + load_m;
                        int global_k = cta_k + load_k_base;

                        if (global_m < batch_size && global_k + 4 <= a_dim) {
                            const float* src_ptr = &A_idx.at(global_m, t, global_k);
                            uintptr_t addr = reinterpret_cast<uintptr_t>(src_ptr);

                            if (addr % 16 == 0) {
                                float data[4];
                                *reinterpret_cast<float4*>(data) = *reinterpret_cast<const float4*>(src_ptr);

                                for (int i = 0; i < 4; ++i) {
                                    tileA[load_m][load_k_base + i] = data[i];
                                }
                            } else {
                                for (int i = 0; i < 4; ++i) {
                                    tileA[load_m][load_k_base + i] = A_idx.at(global_m, t, global_k + i);
                                }
                            }
                        } else {
                            for (int i = 0; i < 4; ++i) {
                                int gk = global_k + i;
                                tileA[load_m][load_k_base + i] = (global_m < batch_size && gk < a_dim) ?
                                    A_idx.at(global_m, t, gk) : 0.f;
                            }
                        }
                    }

                    // Load tileB, transposed load from global memory
                    if (tid >= total_loads_A) {
                        int load_idx_wa = tid - total_loads_A;
                        const int total_loads_Wa = BK * BN / 4;

                        if (load_idx_wa < total_loads_Wa) {
                            int load_k = load_idx_wa / (BN / 4);
                            int load_n_base = (load_idx_wa % (BN / 4)) * 4;

                            int global_k = cta_k + load_k;
                            int global_n = cta_n + load_n_base;

                            if (global_k < a_dim && global_n + 4 <= n_neur) {
                                const float* src_ptr = &Wa_idx.at(global_k, global_n_neuron, global_n);
                                uintptr_t addr = reinterpret_cast<uintptr_t>(src_ptr);

                                if (addr % 16 == 0) {
                                    float data[4];
                                    *reinterpret_cast<float4*>(data) = *reinterpret_cast<const float4*>(src_ptr);

                                    for (int i = 0; i < 4; ++i) {
                                        tileB[load_n_base + i][load_k] = data[i];
                                    }
                                } else {
                                    for (int i = 0; i < 4; ++i) {
                                        tileB[load_n_base + i][load_k] = Wa_idx.at(global_k, global_n_neuron, global_n + i);
                                    }
                                }
                            } else {
                                for (int i = 0; i < 4; ++i) {
                                    int gn = global_n + i;
                                    tileB[load_n_base + i][load_k] = (global_k < a_dim && gn < n_neur) ?
                                        Wa_idx.at(global_k, global_n_neuron, gn) : 0.f;
                                }
                            }
                        }
                    }
                }

                __syncthreads();

                int m0 = thread_m * 4;
                int m1 = thread_m * 4 + 1;
                int m2 = thread_m * 4 + 2;
                int m3 = thread_m * 4 + 3;
                int n = thread_n;

                float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

                for (int k = 0; k < BK; ++k) {
                    float a0 = tileA[m0][k];
                    float a1 = tileA[m1][k];
                    float a2 = tileA[m2][k];
                    float a3 = tileA[m3][k];
                    float b = tileB[n][k];

                    c0 += a0 * b;
                    c1 += a1 * b;
                    c2 += a2 * b;
                    c3 += a3 * b;
                }

                Wa_weighted[0] += c0;
                Wa_weighted[1] += c1;
                Wa_weighted[2] += c2;
                Wa_weighted[3] += c3;

                __syncthreads();
            } // end for cta_k

            //============================ Step 2: Apply epilogue to get W_eff[BM, BN] =====================================
            int n = thread_n;
            int global_n = cta_n + n;

            if (global_n < n_neur) {
                float wo = Wo_idx.at(global_n_neuron, global_n);

#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    Wa_weighted[i] += J0 + J1 * wo;
                }

                //============================= Step 3: Load r[BM, BN] tile and perform partial sum ============================
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int m = thread_m * 4 + i;
                    int global_m = global_m_base + m;

                    if (global_m < batch_size) {
                        float r_val;
                        if (is_first) {
                            r_val = r_init_idx.at(global_m, global_n);
                        } else {
                            r_val = bump_history_idx.at(t - 1, global_m, global_n);
                        }

                        re_accum[i] += Wa_weighted[i] * r_val;
                    }
                }
            }
        } // end for cta_n

        //====================== Step 4: Reduce across BN dimension, update r and save to bump_history =====================
        auto block = cg::this_thread_block();
        auto tile = cg::tiled_partition<32>(block);

#pragma unroll
        for (int row_offset = 0; row_offset < 4; ++row_offset) {
            int m = thread_m * 4 + row_offset;
            int global_m = global_m_base + m;

            if (global_m < batch_size) {
                float row_sum = re_accum[row_offset];
                row_sum = cg::reduce(tile, row_sum, cg::plus<float>());

                if (thread_n == 0) {
                    float r_prev = is_first ?
                        r_init_idx.at(global_m, global_n_neuron) :
                        bump_history_idx.at(t - 1, global_m, global_n_neuron);

                    float r_updated = (1 - alpha) * r_prev + alpha * activation<ACT>(row_sum);
                    bump_history_idx.at(t, global_m, global_n_neuron) = r_updated;
                }
            }
        }
    }

    __global__ void permute_021_kernel_vec4(
        const float* input,   // [seq_len, batch_size, N]
        float* output,        // [batch_size, seq_len, N]
        int seq_len,
        int batch_size,
        int N
    ) {
        int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        int n = idx % N;
        int b = (idx / N) % batch_size;
        int t = idx / (N * batch_size);
        float4 data = *reinterpret_cast<const float4*>(&input[idx]);
        int out_idx = b * (seq_len * N) + t * N + n;
        *reinterpret_cast<float4*>(&output[out_idx]) = data;
    }

    template<Activation ACT>
    void fwd_fp32_impl(
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
        constexpr int BM = 32;
        constexpr int BN = 32;
        constexpr int BK = 16;

        float* bump_history_temp;
        cudaMalloc(&bump_history_temp, seq_len * batch_size * N * sizeof(float));

        dim3 blockSize(256);
        dim3 gridSize(N, (batch_size + BM - 1) / BM);

        for (int t = 0; t < seq_len; ++t){
            fwd_fp32_kernel<ACT, BM, BN, BK><<<gridSize, blockSize>>>(
                static_cast<const float*>(A),
                static_cast<const float*>(Wa),
                J0, J1,
                static_cast<const float*>(Wo),
                static_cast<const float*>(r_init),
                static_cast<float*>(bump_history_temp),
                alpha, N, a_dim, seq_len, t, batch_size
            );
        }

        int total_elements = seq_len * batch_size * N;
        constexpr int THREADS = 256;
        int blocks = (total_elements / 4 + THREADS - 1) / THREADS;

        permute_021_kernel_vec4<<<blocks, THREADS>>>(
            bump_history_temp,
            static_cast<float*>(bump_history),
            seq_len, batch_size, N
        );

        cudaFree(bump_history_temp);
    }

    void fwd_fp32_launcher(
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
        int batch_size,
        int activation_type
    ){
        switch(activation_type){
            case 0: fwd_fp32_impl<Activation::RELU>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 1: fwd_fp32_impl<Activation::GELU>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 2: fwd_fp32_impl<Activation::TANH>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
            case 3: fwd_fp32_impl<Activation::SILU>(A, Wa, J0, J1, Wo, r_init, bump_history, alpha, N, a_dim, seq_len, batch_size); break;
        }
    }
} // namespace fwd_fp32