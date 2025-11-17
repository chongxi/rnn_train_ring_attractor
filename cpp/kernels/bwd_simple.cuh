// #include "cuda_common.cuh"
#include "../cuda_common.cuh"

namespace cuda_fp32 {
    // ---------------------------------------------------------------
    // Kernel 1 – recompute W_eff and the recurrent input “re”
    // ---------------------------------------------------------------
    __global__ void recompute_Weff_re_kernel(
        const float* A,               // [B, S, A_dim]
        const float* Wa,              // [A_dim, N, N]
        float J0,                     // scalar
        float J1,                     // scalar
        const float* Wo,              // [N, N]
        const float* bump_history,    // [S+1, B, N]
        float* W_eff_out,             // [B, N, N] – temporary
        float* re_out,                // [B, N]   – temporary
        float /*alpha*/,              // not used here
        int N,
        int a_dim,
        int seq_len,
        int t,
        int batch_size)
    {
        int b = blockIdx.x;
        if (b >= batch_size) return;

        extern __shared__ float shmem[];
        float* r_prev = shmem;

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            r_prev[i] = bump_history[t * batch_size * N + b * N + i];
        }
        __syncthreads();

        // Compute W_eff
        int idx = threadIdx.x + blockDim.x * blockIdx.y;
        int total = N * N;
        for (int linear = idx; linear < total; linear += blockDim.x * gridDim.y) {
            int i = linear / N;
            int j = linear % N;

            float sum = 0.0f;
            for (int k = 0; k < a_dim; ++k) {
                float a_val = A[b * seq_len * a_dim + t * a_dim + k];
                float wa_val = Wa[k * N * N + i * N + j];
                sum += a_val * wa_val;
            }

            float w_eff = J0 + J1 * Wo[i * N + j] + sum;
            W_eff_out[b * N * N + i * N + j] = w_eff;
        }
        __syncthreads();

        // Compute re = W_eff @ r_prev (each thread computes one output element)
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                sum += W_eff_out[b * N * N + i * N + j] * r_prev[j];
            }
            re_out[b * N + i] = sum;  // Direct write, no atomicAdd
        }
    }

    // ---------------------------------------------------------------
    // Kernel 2 – grad_re = (grad_r + grad_output) * α * φ'(re)
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
    // Kernel 4a – grad_W_eff = grad_re ⊗ r_prev   (outer product)
    // ---------------------------------------------------------------
    __global__ void compute_grad_Weff_kernel(
        const float* grad_re,
        const float* bump_history,
        float* grad_W_eff,
        int N,
        int seq_len,
        int t,
        int batch_size)
    {
        IndexWrapper<const float, 2> grad_re_idx(grad_re, batch_size, N);
        IndexWrapper<const float, 3> bump_idx(bump_history, seq_len + 1, batch_size, N);
        IndexWrapper<float, 3> grad_W_eff_idx(grad_W_eff, batch_size, N, N);

        int b = blockIdx.x;
        if (b >= batch_size) return;

        for (int idx = threadIdx.x; idx < N * N; idx += blockDim.x) {
            int i = idx / N;
            int j = idx % N;

            grad_W_eff_idx.at(b, i, j) = grad_re_idx.at(b, i) * bump_idx.at(t, b, j);
        }
    }

    // ---------------------------------------------------------------
    // Kernel 4b – reduce over batch and accumulate into grad_Wo
    // ---------------------------------------------------------------
    __global__ void accumulate_grad_Wo_kernel(
        const float* grad_W_eff, // [B, N, N]
        float* grad_Wo,          // [N, N] – accumulated
        float J1,
        int N,
        int batch_size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= N || j >= N) return;

        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            sum += grad_W_eff[b * N * N + i * N + j];
        }
        // atomicAdd is safe because each (i,j) is unique per thread
        atomicAdd(&grad_Wo[i * N + j], J1 * sum);
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
        const float* grad_output,   // [S, B, N]
        const float* A,             // [B, S, A_dim]
        const float* Wa,            // [A_dim, N, N]
        float J0,                   // scalar (demo)
        float J1,                   // scalar
        const float* Wo,            // [N, N]
        const float* bump_history,  // [S+1, B, N]
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
        float *grad_r, *W_eff_temp, *re_temp, *grad_re_temp, *grad_W_eff_temp;
        cudaMalloc(&grad_r,        batch_size * N * sizeof(float));
        cudaMalloc(&W_eff_temp,   batch_size * N * N * sizeof(float));
        cudaMalloc(&re_temp,      batch_size * N * sizeof(float));
        cudaMalloc(&grad_re_temp, batch_size * N * sizeof(float));
        cudaMalloc(&grad_W_eff_temp, batch_size * N * N * sizeof(float));

        cudaMemset(grad_r, 0, batch_size * N * sizeof(float));
        cudaMemset(grad_Wa, 0, a_dim * N * N * sizeof(float));
        cudaMemset(grad_Wo, 0, N * N * sizeof(float));

        // -----------------------------------------------------------------
        // Tile sizes – feel free to tune them for your GPU
        // -----------------------------------------------------------------
        constexpr int BM = 128, BN = 128, BK = 16;
        constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

        // -----------------------------------------------------------------
        // Back‑propagation through time (reverse order)
        // -----------------------------------------------------------------
        for (int t = seq_len - 1; t >= 0; --t) {
            // -------------------------------------------------------------
            // Kernel 1 – recompute W_eff and re
            // -------------------------------------------------------------
            dim3 grid1(batch_size);
            dim3 block1(256);
            size_t shmem_bytes = N * sizeof(float);   // r_prev
            recompute_Weff_re_kernel<<<grid1, block1, shmem_bytes>>>(
                    A, Wa, J0, J1, Wo,
                    bump_history,
                    W_eff_temp, re_temp,
                    alpha, N, a_dim, seq_len, t, batch_size);
            cudaGetLastError();

            // -------------------------------------------------------------
            // Kernel 2 – grad_re
            // -------------------------------------------------------------
            dim3 grid2(batch_size);
            dim3 block2(256);
            compute_grad_re_kernel<ACT><<<grid2, block2>>>(
                grad_output, grad_r, re_temp, grad_re_temp,
                alpha, N, seq_len, t, batch_size);
            cudaGetLastError();

            // -------------------------------------------------------------
            // Kernel 3 – grad_r (batched GEMV)
            // -------------------------------------------------------------
            dim3 grid3(batch_size);
            dim3 block3(256);
            compute_grad_r_kernel<<<grid3, block3>>>(
                W_eff_temp, grad_re_temp, grad_r,
                alpha, N, batch_size);
            cudaGetLastError();

            // -------------------------------------------------------------
            // Kernel 4a – grad_W_eff = grad_re ⊗ r_prev
            // -------------------------------------------------------------
            dim3 grid4a(batch_size);
            dim3 block4a(256);  // 1D block
            compute_grad_Weff_kernel<<<grid4a, block4a>>>(
                grad_re_temp, bump_history,
                grad_W_eff_temp, N, seq_len, t, batch_size);
            cudaGetLastError();

            // -------------------------------------------------------------
            // Kernel 4b – reduce over batch → grad_Wo
            // -------------------------------------------------------------
            dim3 block4b(16, 16);
            dim3 grid4b((N + block4b.x - 1) / block4b.x,
                        (N + block4b.y - 1) / block4b.y);
            accumulate_grad_Wo_kernel<<<grid4b, block4b>>>(
                grad_W_eff_temp, grad_Wo, J1, N, batch_size);
            cudaGetLastError();

            // -------------------------------------------------------------
            // Kernel 5 – grad_Wa += A_tᵀ · grad_W_eff
            // -------------------------------------------------------------
            dim3 block5(16, 16);
            dim3 grid5((a_dim + block5.x - 1) / block5.x,
                       ((N * N) + block5.y - 1) / block5.y);
            compute_grad_Wa_kernel<BM, BN, BK,
                                   WMMA_M, WMMA_N, WMMA_K>
                <<<grid5, block5>>>(
                    A, grad_W_eff_temp, grad_Wa,
                    N, a_dim, seq_len, t, batch_size);
            cudaGetLastError();
        }

        // -----------------------------------------------------------------
        // Clean‑up
        // -----------------------------------------------------------------
        cudaFree(grad_r);
        cudaFree(W_eff_temp);
        cudaFree(re_temp);
        cudaFree(grad_re_temp);
        cudaFree(grad_W_eff_temp);
    }


    // Main launcher
    void bwd_fp32_launcher(
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
}// namespace cuda_fp32

