// #include <ATen/Tensor.h>
// #include <ATen/Functions.h>
// #include <torch/csrc/utils/pybind.h>

// void torch_sum_cuda(
//     void* A_t,
//     void* Wa,
//     void* J0,
//     float J1,
    
//     void* r, // init value for r

//     void* Wa_weighted, // internal use
//     // void* re_inp, // internal use
//     void* Wo,

//     int N,
//     int a_dim
// );

// void torch_sum(
//     const at::Tensor& A_t, 
//     const at::Tensor& Wa,
//     const at::Tensor& J0,
//     float J1,
//     const at::Tensor& Wo,
    
//     at::Tensor& r,

//     at::Tensor& Wa_weighted
//     // at::Tensor& r_inp
// ) {
//     int a_dim = Wa.sizes()[0];
//     int N = Wa.sizes()[1];

//     torch_sum_cuda(
//         A_t.data_ptr(), 
//         Wa.data_ptr(),
//         J0.data_ptr(),
//         J1,
//         Wo.data_ptr(),
        
//         r.data_ptr(),
        
//         Wa_weighted.data_ptr(), 

//         N, 
//         a_dim);
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("torch_sum", &torch_sum, "torch sum (CUDA)");
// }

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <torch/csrc/utils/pybind.h>

void torch_sum_cuda(
    void* A_t,
    void* Wa,
    
    void* Wa_weighted, // internal use



    int N,
    int a_dim
);

void torch_sum(const at::Tensor& A_t, const at::Tensor& Wa, at::Tensor& Wa_weighted) {
    int a_dim = Wa.sizes()[0];
    int N = Wa.sizes()[1];

    torch_sum_cuda(A_t.data_ptr(), Wa.data_ptr(), Wa_weighted.data_ptr(), N, a_dim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_sum", &torch_sum, "torch sum (CUDA)");
}