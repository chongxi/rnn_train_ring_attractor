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