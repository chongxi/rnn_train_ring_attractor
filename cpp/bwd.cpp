#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <torch/csrc/utils/pybind.h>

void bwd_cuda(
    void* grad_output,
    void* A,
    void* Wa,
    float J0,
    float J1,
    void* Wo,
    void* bump_history,
    void* r_init,
    void* grad_Wa,
    void* grad_Wo,
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size,
    int activation_type
);

void bwd(
    const at::Tensor& grad_output,
    const at::Tensor& A,
    const at::Tensor& Wa,
    float J0,
    float J1,
    const at::Tensor& Wo,
    const at::Tensor& bump_history,
    const at::Tensor& r_init,
    at::Tensor& grad_Wa,
    at::Tensor& grad_Wo,
    float alpha,
    int activation_type
) {
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(Wa.is_contiguous(), "Wa must be contiguous");
    TORCH_CHECK(Wo.is_contiguous(), "Wo must be contiguous");
    TORCH_CHECK(bump_history.is_contiguous(), "bump_history must be contiguous");
    TORCH_CHECK(r_init.is_contiguous(), "r_init must be contiguous");
    TORCH_CHECK(grad_Wa.is_contiguous(), "grad_Wa must be contiguous");
    TORCH_CHECK(grad_Wo.is_contiguous(), "grad_Wo must be contiguous");

    auto sizes = A.sizes();
    int batch_size = sizes[0], seq_len = sizes[1], a_dim = sizes[2];
    int N = Wa.size(1);

    bwd_cuda(
        grad_output.data_ptr(),
        A.data_ptr(),
        Wa.data_ptr(),
        J0,
        J1,
        Wo.data_ptr(),
        bump_history.data_ptr(),
        r_init.data_ptr(),
        grad_Wa.data_ptr(),
        grad_Wo.data_ptr(),
        alpha, N, a_dim, seq_len, batch_size, activation_type
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bwd", &bwd, "bwd (CUDA)",
          py::arg("grad_output"),
          py::arg("A"),
          py::arg("Wa"),
          py::arg("J0"),
          py::arg("J1"),
          py::arg("Wo"),
          py::arg("bump_history"),
          py::arg("r_init"),
          py::arg("grad_Wa"),
          py::arg("grad_Wo"),
          py::arg("alpha"),
          py::arg("activation_type"));
}