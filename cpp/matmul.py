import torch
from torch.utils.cpp_extension import load
import pathlib
import os

print("----------------------------------------------")
print("---------------- CHECK GPU -------------------")
print("----------------------------------------------")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

capability = torch.cuda.get_device_capability(torch.cuda.current_device())
if capability[0] < 8:
    raise RuntimeError(f"GPU compute capability {capability[0]}.{capability[1]} is below minimum required (8.0)")

os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
print(f"Using GPU with compute capability {capability[0]}.{capability[1]}")
# os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

print("--------------------------------------------------------")
print("---------------- COMPILE CUDA MODULE -------------------")
print("--------------------------------------------------------")


dir_path = pathlib.Path(__file__).parent.absolute()
print(f"dir_path: {dir_path}")

force_rebuild = False
build_dir = f"{dir_path}/build"

build_path = pathlib.Path(build_dir)
build_path.mkdir(parents=True, exist_ok=True)
if force_rebuild:
    for file in build_path.glob("*"):
        file.unlink()

module = load(
    name='matmul',
    sources=[f"{dir_path}/matmul_kernel.cu", f"{dir_path}/matmul.cpp"],
    verbose=True,
    build_directory=build_dir 
)

print("------------------------------------------------------")
print("---------------- MAIN PYTORCH CODE -------------------")
print("------------------------------------------------------")

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