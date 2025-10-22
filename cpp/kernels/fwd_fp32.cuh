#include "cuda_common.cuh"
#include "../cuda_common.cuh"

template<Activation ACT, int BM, int BN, int BK>
__global__ void __launch_bounds__(256) fwd_update_r_kernel_fp32(
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
    const int thread_m = tid / 32;  // 0-7
    const int thread_n = tid % 32;  // 0-31

    bool is_first = (t == 0);

    __shared__ float tileA[BM][BK + 2];
    __shared__ float tileB[BN][BK + 2];  // Transposed: [BN][BK]

    float re_accum[4] = {0.f, 0.f, 0.f, 0.f};

    for (int cta_n = 0; cta_n < n_neur; cta_n += BN) {
        float Wa_weighted[4] = {0.f, 0.f, 0.f, 0.f};

        for (int cta_k = 0; cta_k < a_dim; cta_k += BK) {

            // Load A tile: vectorized load using float data[4]
            {
                int load_idx = tid;
                int load_m = load_idx / (BK / 4);  // Which row
                int load_k_base = (load_idx % (BK / 4)) * 4;  // Which column (base)

                if (load_m < BM) {
                    int global_m = blockIdx.y * BM + load_m;
                    int global_k = cta_k + load_k_base;

                    // Get pointer to global memory for vectorized load
                    const float* src_ptr = &A_idx.at(global_m, t, global_k);
                    float data[4];
                    *reinterpret_cast<float4*>(data) = *reinterpret_cast<const float4*>(src_ptr);

                    for (int i = 0; i < 4; ++i) {
                        tileA[load_m][load_k_base + i] = data[i];
                    }
                }
            }

            // Load Wa tile TRANSPOSED with vectorized load using float data[4]
            // We load 4 consecutive elements along the N dimension and transpose to tileB
            {
                int load_idx = tid;
                int load_k = load_idx / (BN / 4);  // Which k index
                int load_n_base = (load_idx % (BN / 4)) * 4;  // Which n index (base of 4)

                if (load_k < BK && load_n_base < BN) {
                    int global_k = cta_k + load_k;
                    int global_n = cta_n + load_n_base;

                    // Load 4 consecutive elements along N dimension
                    const float* src_ptr = &Wa_idx.at(global_k, blockIdx.x, global_n);
                    float data[4];
                    *reinterpret_cast<float4*>(data) = *reinterpret_cast<const float4*>(src_ptr);

                    for (int i = 0; i < 4; ++i) {
                        tileB[load_n_base + i][load_k] = data[i];
                    }
                }
            }

            __syncthreads();

            // Compute: each thread computes 4x1 tile (4 elements)
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

        // Apply epilogue: W_eff = Wa_weighted + J0 + J1 * Wo
        int n = thread_n;
        float wo = Wo_idx.at(blockIdx.x, cta_n + n);

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            Wa_weighted[i] += J0 + J1 * wo;
        }

        // Load r and accumulate
        float r[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int m = thread_m * 4 + i;
            int global_m = blockIdx.y * BM + m;

            if (is_first) {
                r[i] = r_init_idx.at(global_m, cta_n + n);
            } else {
                r[i] = bump_history_idx.at(t - 1, global_m, cta_n + n);
            }

            re_accum[i] += Wa_weighted[i] * r[i];
        }
    }

    // Reduce across N dimension using cooperative groups
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    // Each row needs to be reduced independently
    for (int row_offset = 0; row_offset < 4; ++row_offset) {
        int m = thread_m * 4 + row_offset;
        int global_m = blockIdx.y * BM + m;

        // Sum across the warp (32 threads, each holding one n-column)
        float row_sum = re_accum[row_offset];
        row_sum = cg::reduce(tile, row_sum, cg::plus<float>());

        // Only the first thread in each warp writes the result
        if (thread_n == 0) {
            float r_prev = is_first ?
                r_init_idx.at(global_m, blockIdx.x) :
                bump_history_idx.at(t - 1, global_m, blockIdx.x);

            float r_updated = (1 - alpha) * r_prev + alpha * activation<ACT>(row_sum);
            bump_history_idx.at(t, global_m, blockIdx.x) = r_updated;
        }
    }
}

template<Activation ACT>
void fwd_n128_a23_global_launcher_fp32_impl(
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
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;

    dim3 blockSize(256);
    dim3 gridSize(N, batch_size / BM);

    for (int t = 0; t < seq_len; ++t){
        fwd_update_r_kernel_fp32<ACT, BM, BN, BK><<<gridSize, blockSize>>>(
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
    switch(activation_type){
        case 0: fwd_n128_a23_global_launcher_fp32_impl<Activation::RELU>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
        case 1: fwd_n128_a23_global_launcher_fp32_impl<Activation::GELU>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
        case 2: fwd_n128_a23_global_launcher_fp32_impl<Activation::TANH>(A, Wa, J0, J1, Wo, r_init, W_delta7, bump_history, r_history, alpha, N, a_dim, seq_len, batch_size); break;
    }
}