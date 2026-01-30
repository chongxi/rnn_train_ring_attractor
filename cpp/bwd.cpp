#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <torch/csrc/utils/pybind.h>
#include <cublas_v2.h>

void bwd_cuda(
    void* grad_output,
    void* A,
    void* Wa,
    float J0,
    float J1,
    void* Wo,
    void* bump_history,
    void* re_history,
    void* r_init,
    void* grad_Wa,
    void* grad_Wo,
    float alpha,
    int N,
    int a_dim,
    int seq_len,
    int batch_size,
    int activation_type,
    // Workspace buffers
    void* grad_r,
    void* W_eff_temp,
    void* grad_re_temp,
    void* grad_W_eff_temp_bf16,
    void* A_t_temp_bf16,
    void* grad_re_temp_T,
    void* cublas_handle
);

void bwd(
    const at::Tensor& grad_output,
    const at::Tensor& A,
    const at::Tensor& Wa,
    float J0,
    float J1,
    const at::Tensor& Wo,
    const at::Tensor& bump_history,
    const at::Tensor& re_history,
    const at::Tensor& r_init,
    at::Tensor& grad_Wa,
    at::Tensor& grad_Wo,
    float alpha,
    int activation_type,
    at::Tensor& grad_r,
    at::Tensor& W_eff_temp,
    at::Tensor& grad_re_temp,
    at::Tensor& grad_W_eff_temp_bf16,
    at::Tensor& A_t_temp_bf16,
    at::Tensor& grad_re_temp_T,
    int64_t cublas_handle_ptr
) {
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(Wa.is_contiguous(), "Wa must be contiguous");
    TORCH_CHECK(Wo.is_contiguous(), "Wo must be contiguous");
    TORCH_CHECK(bump_history.is_contiguous(), "bump_history must be contiguous");
    TORCH_CHECK(re_history.is_contiguous(), "re_history must be contiguous");
    TORCH_CHECK(r_init.is_contiguous(), "r_init must be contiguous");
    TORCH_CHECK(grad_Wa.is_contiguous(), "grad_Wa must be contiguous");
    TORCH_CHECK(grad_Wo.is_contiguous(), "grad_Wo must be contiguous");

    // NEW: workspace buffer checks
    TORCH_CHECK(grad_r.is_contiguous(), "grad_r must be contiguous");
    TORCH_CHECK(W_eff_temp.is_contiguous(), "W_eff_temp must be contiguous");
    TORCH_CHECK(grad_re_temp.is_contiguous(), "grad_re_temp must be contiguous");
    TORCH_CHECK(grad_W_eff_temp_bf16.is_contiguous(), "grad_W_eff_temp_bf16 must be contiguous");
    TORCH_CHECK(A_t_temp_bf16.is_contiguous(), "A_t_temp_bf16 must be contiguous");
    TORCH_CHECK(grad_re_temp_T.is_contiguous(), "grad_re_temp_T must be contiguous");

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
        re_history.data_ptr(),
        r_init.data_ptr(),
        grad_Wa.data_ptr(),
        grad_Wo.data_ptr(),
        alpha, N, a_dim, seq_len, batch_size, activation_type,
        grad_r.data_ptr(),
        W_eff_temp.data_ptr(),
        grad_re_temp.data_ptr(),
        grad_W_eff_temp_bf16.data_ptr(),
        A_t_temp_bf16.data_ptr(),
        grad_re_temp_T.data_ptr(),
        reinterpret_cast<void*>(cublas_handle_ptr)
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
          py::arg("re_history"),
          py::arg("r_init"),
          py::arg("grad_Wa"),
          py::arg("grad_Wo"),
          py::arg("alpha"),
          py::arg("activation_type"),
          py::arg("grad_r"),
          py::arg("W_eff_temp"),
          py::arg("grad_re_temp"),
          py::arg("grad_W_eff_temp_bf16"),
          py::arg("A_t_temp_bf16"),
          py::arg("grad_re_temp_T"),
          py::arg("cublas_handle"));
}