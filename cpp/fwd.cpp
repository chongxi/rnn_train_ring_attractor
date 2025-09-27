#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <torch/csrc/utils/pybind.h>

void fwd_cuda(
    void* A,
    void* Wa,
    // void* J0,
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
);

void fwd(
    const at::Tensor& A, 
    const at::Tensor& Wa,
    // const at::Tensor& J0,
    float J0,
    float J1,
    const at::Tensor& Wo,
    at::Tensor& r_init,
    at::Tensor& W_delta7,
    at::Tensor& bump_history,
    at::Tensor& r_history,
    float alpha
) {
    auto sizes = A.sizes();
    int batch_size = sizes[0], seq_len = sizes[1], a_dim = sizes[2];
    int N = Wa.size(1);

    fwd_cuda(
        A.data_ptr(), 
        Wa.data_ptr(),
        // J0.data_ptr(),
        J0,
        J1,
        Wo.data_ptr(),
        r_init.data_ptr(),
        W_delta7.data_ptr(), 
        bump_history.data_ptr(),
        r_history.data_ptr(),
        alpha,
        N, 
        a_dim,
        seq_len,
        batch_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd, "fwd (CUDA)",
          py::arg("A"), py::arg("Wa"), py::arg("J0"), py::arg("J1"), py::arg("Wo"), py::arg("r_init"), py::arg("W_delta7"), py::arg("bump_history"), py::arg("r_history"), py::arg("alpha"));
}