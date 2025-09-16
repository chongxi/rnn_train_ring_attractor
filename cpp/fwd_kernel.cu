void fwd_loop(
    void* A,
    void* J0,
    float J1,
    void* Wo,
    void* Wa,
    void* W_d7,

    void* r, // init value for r
    void* bump_history,
    void* r_history,
    
    void* Wa_weighted, // internal use
    void* re_inp, // internal use
    void* r_d7, // internal use

    int seq_len,
    int N,
    int a_dim
);

__global__ void fwd_loop_kernel(
    const float* A,
    const float* J0,
    float J1,
    const float* Wo,
    const float* Wa,
    const float* W_d7,

    float* r, // init value for r
    float* bump_history,
    float* r_history,
    
    float* Wa_weighted, // internal use
    float* re_inp, // internal use
    float* r_d7, // internal use

    int seq_len,
    int N,
    int a_dim
){
    constexpr float ALPHA = 1.5;
    
    int row_idx = blockIdx.x;
    int col_idx = threadIdx.x;

    // for (int t = 0; t < seq_len; ++t){

    //     // Step 1:

    //     Wa_weighted[row_idx * N + col_idx] = Wa[][row_idx][col_idx]

    // } // end for t

}

void fwd_loop(
    void* A,
    void* J0,
    float J1,
    void* Wo,
    void* Wa,
    void* W_d7,

    void* r, // init value for r
    void* bump_history,
    void* r_history,
    
    void* Wa_weighted, // internal use
    void* re_inp, // internal use
    void* r_d7, // internal use

    int seq_len,
    int N,
    int a_dim
){
    dim3 blockSize, gridSize;

    blockSize = N; gridSize = N;
    fwd_loop_kernel<<<gridSize, blockSize>>>(
        static_cast<const float*>(A),
        static_cast<const float*>(J0),
        J1,
        static_cast<const float*>(Wo),
        static_cast<const float*>(Wa),
        static_cast<const float*>(W_d7),

        static_cast<float *>(r), // init value for r
        static_cast<float *>(bump_history),
        static_cast<float *>(r_history),
        
        static_cast<float *>(Wa_weighted), // internal use
        static_cast<float *>(re_inp), // internal use
        static_cast<float *>(r_d7), // internal use

        seq_len,
        N,
        a_dim        
    );
}