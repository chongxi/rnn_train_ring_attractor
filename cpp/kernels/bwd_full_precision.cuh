// #include "cuda_common.cuh"
#include "../cuda_common.cuh"

namespace cuda_wmma {
    __global__ void compute_grad_r_kernel(
        const float* W_eff,
        const float* grad_re,
        float* grad_r,
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

    template<Activation ACT, int BM, int BN, int BK>
    __global__ void __launch_bounds__(256) recompute_Weff_re_grad_re_kernel(
        const float* A,
        const float* Wa,
        float J0,
        float J1,
        const float* Wo,
        const float* bump_history,
        const float* grad_output,
        float* grad_r,
        float* W_eff_out,
        float* grad_re_out,
        float* A_t_temp,
        float* grad_W_eff,
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
        IndexWrapper<const float, 3> bump_history_idx(bump_history, seq_len + 1, batch_size, n_neur);
        IndexWrapper<const float, 3> grad_output_idx(grad_output, seq_len, batch_size, n_neur);
        IndexWrapper<float, 2> grad_r_idx(grad_r, batch_size, n_neur);
        IndexWrapper<float, 2> W_eff_out_idx(W_eff_out, batch_size, n_neur * n_neur);
        IndexWrapper<float, 2> grad_re_out_idx(grad_re_out, batch_size, n_neur);

        const int tid = threadIdx.x;
        const int thread_m = tid / 32;
        const int thread_n = tid % 32;

        const int global_m_base = blockIdx.y * BM;
        const int global_n_neuron = blockIdx.x;

        if (global_n_neuron >= n_neur) return;

        __shared__ float tileA[BM][BK + 2];
        __shared__ float tileB[BN][BK + 2];

        float re_accum[4] = {0.f, 0.f, 0.f, 0.f};

        for (int cta_n = 0; cta_n < n_neur; cta_n += BN) {
            float Wa_weighted[4] = {0.f, 0.f, 0.f, 0.f};

            for (int cta_k = 0; cta_k < a_dim; cta_k += BK) {
                {
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
                                float4 data = *reinterpret_cast<const float4*>(src_ptr);

                                tileA[load_m][load_k_base] = data.x;
                                tileA[load_m][load_k_base + 1] = data.y;
                                tileA[load_m][load_k_base + 2] = data.z;
                                tileA[load_m][load_k_base + 3] = data.w;

                                float* a_dest = &A_t_temp[global_m * a_dim + global_k];
                                *reinterpret_cast<float4*>(a_dest) = data;

                            } else {
                                for (int i = 0; i < 4; ++i) {
                                    float val = A_idx.at(global_m, t, global_k + i);
                                    tileA[load_m][load_k_base + i] = val;
                                    A_t_temp[global_m * a_dim + global_k + i] = val;
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
            }

            int n = thread_n;
            int global_n = cta_n + n;

            if (global_n < n_neur) {
                float wo = Wo_idx.at(global_n_neuron, global_n);

#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    Wa_weighted[i] += J0 + J1 * wo;
                }

                int offset = global_n_neuron * n_neur + global_n;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int m = thread_m * 4 + i;
                    int global_m = global_m_base + m;

                    if (global_m < batch_size) {
                        W_eff_out_idx.at(global_m, offset) = Wa_weighted[i];
                    }
                }

#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int m = thread_m * 4 + i;
                    int global_m = global_m_base + m;

                    if (global_m < batch_size) {
                        float r_val = bump_history_idx.at(t, global_m, global_n);
                        re_accum[i] += Wa_weighted[i] * r_val;
                    }
                }
            }
        }

        auto block = cg::this_thread_block();
        auto tile = cg::tiled_partition<32>(block);

// #pragma unroll
        for (int row_offset = 0; row_offset < 4; ++row_offset) {
            int m = thread_m * 4 + row_offset;
            int global_m = global_m_base + m;

            if (global_m < batch_size) {
                float row_sum = re_accum[row_offset];
                row_sum = cg::reduce(tile, row_sum, cg::plus<float>());

                if (thread_n == 0) {
                    float go = grad_output_idx.at(t, global_m, global_n_neuron);
                    float gr = grad_r_idx.at(global_m, global_n_neuron) + go;
                    grad_r_idx.at(global_m, global_n_neuron) = gr;

                    float dphi = activation_derivative<ACT>(row_sum);
                    float grad_re_val = dphi * gr * alpha;
                    grad_re_out_idx.at(global_m, global_n_neuron) = grad_re_val;
                }
            }
        }
    }

    template<Activation ACT>
    void bwd_fp32_impl(
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
        float *grad_r, *W_eff_temp, *grad_re_temp, *grad_W_eff_temp, *A_t_temp, *grad_r_from_recurrent;
        cudaMalloc(&grad_r, batch_size * N * sizeof(float));
        cudaMalloc(&W_eff_temp, batch_size * N * N * sizeof(float));
        cudaMalloc(&grad_re_temp, batch_size * N * sizeof(float));
        cudaMalloc(&grad_W_eff_temp, batch_size * N * N * sizeof(float));
        cudaMalloc(&A_t_temp, batch_size * a_dim * sizeof(float));
        cudaMalloc(&grad_r_from_recurrent, batch_size * N * sizeof(float));

        cudaMemset(grad_r, 0, batch_size * N * sizeof(float));

        cublasHandle_t handle;
        cublasCreate(&handle);

        constexpr int BM = 32;
        constexpr int BN = 32;
        constexpr int BK = 16;

        for (int t = seq_len - 1; t >= 0; --t) {
            dim3 blockSize(256);
            dim3 gridSize(N, (batch_size + BM - 1) / BM);

            recompute_Weff_re_grad_re_kernel<ACT, BM, BN, BK><<<gridSize, blockSize>>>(
                A, Wa, J0, J1, Wo,
                bump_history,
                grad_output, grad_r,
                W_eff_temp, grad_re_temp, A_t_temp, grad_W_eff_temp,
                alpha, N, a_dim, seq_len, t, batch_size);
            CHECK_CUDA_ERROR(cudaGetLastError());

            dim3 block3(128);
            dim3 grid3(batch_size, (N + block3.x - 1) / block3.x);
            compute_grad_r_kernel<<<grid3, block3>>>(
                W_eff_temp, grad_re_temp, grad_r,
                alpha, N, batch_size);

            float one = 1.0f;
            float zero = 0.0f;

            CHECK_CUDA_ERROR(cudaGetLastError());

            const float* r_prev = bump_history + t * batch_size * N;

            cublasSgemmStridedBatched(handle,
                                      CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      N,
                                      N,
                                      1,
                                      &one,
                                      r_prev,
                                      1,
                                      N,
                                      grad_re_temp,
                                      N,
                                      N,
                                      &zero,
                                      grad_W_eff_temp,
                                      N,
                                      N * N,
                                      batch_size);
            CHECK_CUDA_ERROR(cudaGetLastError());

            float beta = 1.0f;

            cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_T,
                        N,
                        N,
                        batch_size,
                        &J1,
                        r_prev,
                        N,
                        grad_re_temp,
                        N,
                        &beta,
                        grad_Wo,
                        N);
            CHECK_CUDA_ERROR(cudaGetLastError());

            float beta_wa = 1.0f;

            cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_T,
                        N * N,
                        a_dim,
                        batch_size,
                        &one,
                        grad_W_eff_temp,
                        N * N,
                        A_t_temp,
                        a_dim,
                        &beta_wa,
                        grad_Wa,
                        N * N);

            CHECK_CUDA_ERROR(cudaGetLastError());
        }

        cublasDestroy(handle);

        cudaFree(grad_r);
        cudaFree(W_eff_temp);
        cudaFree(grad_re_temp);
        cudaFree(grad_W_eff_temp);
        cudaFree(A_t_temp);
    }

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
            case 0: bwd_fp32_impl<Activation::RELU>(grad_output, A, Wa, J0, J1, Wo, bump_history, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size); break;
            case 1: bwd_fp32_impl<Activation::GELU>(grad_output, A, Wa, J0, J1, Wo, bump_history, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size); break;
            case 2: bwd_fp32_impl<Activation::TANH>(grad_output, A, Wa, J0, J1, Wo, bump_history, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size); break;
            case 3: bwd_fp32_impl<Activation::SILU>(grad_output, A, Wa, J0, J1, Wo, bump_history, grad_Wa, grad_Wo, alpha, N, a_dim, seq_len, batch_size); break;
        }
    }
}