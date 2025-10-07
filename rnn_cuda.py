import torch
from torch.utils.cpp_extension import load
import pathlib
import os

print("========================================================") 

"""
Example usage:
from rnn_cuda import * 
...
        rnn_cuda.fwd(
            A=action_signal, 
            Wa=self.Wa,
            J0=self.J0,
            J1=self.J1,
            Wo=self.Wo,        
            r_init=r,
            W_delta7=self.W_delta7,  
            bump_history=self.bump_history,
            r_history=self.r_history,
            alpha=alpha
        ) 
...
"""


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not found")

force_rebuild = False
capability = torch.cuda.get_device_capability(torch.cuda.current_device())
name = torch.cuda.get_device_name(torch.cuda.current_device())

if capability[0] < 9:
    raise RuntimeError(f"GPU compute capability {capability[0]}.{capability[1]} is below minimum required (9.0)")

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

rnn_cuda = load(
    name='fwd',
    sources=[f"{dir_path}/cpp/fwd.cu", f"{dir_path}/cpp/fwd.cpp"],
    verbose=True,
    build_directory=build_dir,
    with_cuda=True
)

# module = load(
#     name='fwd',
#     sources=[f"{dir_path}/cpp/fwd.cu", f"{dir_path}/cpp/fwd.cpp"],
#     verbose=True,
#     build_directory=build_dir,
#     extra_cuda_cflags=[
#         # "-lineinfo",          # useful for profiling
#         "-Xptxas=-v",         # print register/shared memory usage
#         # "--ptxas-options=-v", # alternative syntax
#         "-keep"               # keep intermediate files (including .ptx and .cubin)
#     ]
# )
