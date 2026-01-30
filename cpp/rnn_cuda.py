import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import pathlib
import os
import sys

print()
print("[WARNING] If dimensions (batch_size, a_dim, num_neurons) must be divisible by 16 to activate tensor core kernel.")
print("[WARNING] Currently compiles for sm_120, change `extra_cuda_cflags` to sm_80 for ampere GPU")
print()

print("========================================================") 

"""
Example usage:
from rnn_cuda import * 
...
        rnn_cuda.fwd(
            A=action_signal, 
            Wa=Wa,
            J0=J0,
            J1=J1,
            Wo=Wo,        
            r_init=r,
            W_delta7=W_delta7,  
            bump_history=bump_history,
            r_history=r_history,
            alpha=alpha
        ) 
...
"""

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not found")

force_rebuild = False
capability = torch.cuda.get_device_capability(torch.cuda.current_device())
name = torch.cuda.get_device_name(torch.cuda.current_device())

if capability[0] < 8:
    raise RuntimeError(f"GPU compute capability {capability[0]}.{capability[1]} is below minimum required (8.0)")

os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
print(f"GPU: {name}, compute capability: {capability[0]}.{capability[1]}")

dir_path = pathlib.Path(__file__).parent.absolute()
print(f"dir_path: {dir_path}")


build_dir = f"{dir_path}/build"

build_path = pathlib.Path(build_dir)
build_path.mkdir(parents=True, exist_ok=True)
if force_rebuild:
    for file in build_path.glob("*"):
        file.unlink()

fwd_cuda = load(
    name='fwd',
    sources=[f"{dir_path}/fwd.cu", f"{dir_path}/fwd.cpp"],
    verbose=True,
    build_directory=build_dir,
    extra_cuda_cflags=[
        # "-lineinfo",          # useful for profiling
        # "-Xptxas=-v",         # print register/shared memory usage
        # # "--ptxas-options=-v", # alternative syntax
        # "-keep",               # keep intermediate files (including .ptx and .cubin)
        
        # "-arch=sm_120" # support ampere tensor core sm_80 and above
        f"-arch=sm_{capability[0]}{capability[1]}"
    ]
)

extra_ldflags = []
if sys.platform == "win32":
    extra_ldflags = ["cublas.lib"]
else:
    extra_ldflags = ["-lcublas"]

bwd_cuda = load(
    name='bwd',
    sources=[f"{dir_path}/bwd.cu", f"{dir_path}/bwd.cpp"],
    verbose=True,
    build_directory=build_dir,
    extra_cuda_cflags=[
        # "-arch=sm_120"
        f"-arch=sm_{capability[0]}{capability[1]}"
    ],
    extra_ldflags=extra_ldflags
)