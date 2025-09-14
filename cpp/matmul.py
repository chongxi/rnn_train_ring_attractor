import torch
from torch.utils.cpp_extension import load
import pathlib
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

dir_path = pathlib.Path(__file__).parent.absolute()
print(f"dir_path: {dir_path}")

build_dir = f"{dir_path}/build"

if not pathlib.Path(build_dir).exists():
    pathlib.Path(build_dir).mkdir(parents=True)
else:
    # Clean the build directory
    for file in pathlib.Path(build_dir).glob("*"):
        file.unlink()

module = load(
    name='matmul',
    sources=[f"{dir_path}/matmul_kernel.cu", f"{dir_path}/matmul.cpp"],
    verbose=True,
    build_directory=build_dir 
)

M = 1024
N = 2048
K = 512

if M % 128 != 0 or N % 128 != 0:
    raise ValueError("M and N must be divisible by 128")
if K % 64 != 0:
    raise ValueError("K must be divisible by 64")

A_torch = torch.randn(M, K, device='cuda', dtype=torch.float32)
B_torch = torch.randn(N, K, device='cuda', dtype=torch.float32)
C_torch = torch.zeros(M, N, device='cuda', dtype=torch.float32)

print("A_torch shape: ", A_torch.shape, "device: ", A_torch.device)
print("B_torch shape: ", B_torch.shape)
print("C_torch shape: ", C_torch.shape)

module.matmul(A_torch, B_torch.t(), C_torch)

print("My results:")
print(C_torch[0][:10])
print("Torch result:")
C_torch_native = torch.matmul(A_torch, B_torch.t())
print(C_torch_native[0][:10])

is_close = torch.allclose(C_torch, C_torch_native, atol=0.1, rtol=0.001)
print("Results match:", is_close)