#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <torch/csrc/utils/pybind.h>

void matmul_cuda(void* a, void* b, void* c, int M, int N, int K);

void matmul(const at::Tensor& a, const at::Tensor& b, at::Tensor& c) {
    int M = a.sizes()[0];
    int K = b.sizes()[0];
    int N = c.sizes()[1];

    matmul_cuda(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Matmul (CUDA)");
}