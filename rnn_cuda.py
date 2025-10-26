import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import pathlib
import os

print()
print("WARNING: If dimensions (batch_size, a_dim, num_neurons) must be divisible by 16 to activate tensor core kernel, otherwise it will fall back to non tensor core version.")
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

# fwd_cuda = load(
#     name='fwd',
#     sources=[f"{dir_path}/cpp/fwd.cu", f"{dir_path}/cpp/fwd.cpp"],
#     verbose=True,
#     build_directory=build_dir,
#     with_cuda=True
# )

fwd_cuda = load(
    name='fwd',
    sources=[f"{dir_path}/cpp/fwd.cu", f"{dir_path}/cpp/fwd.cpp"],
    verbose=True,
    build_directory=build_dir,
    extra_cuda_cflags=[
        # "-lineinfo",          # useful for profiling
        # "-Xptxas=-v",         # print register/shared memory usage
        # # "--ptxas-options=-v", # alternative syntax
        # "-keep",               # keep intermediate files (including .ptx and .cubin)
        "-arch=sm_120"
    ]
)

# class RnnCudaFunction(torch.autograd.Function):
#     """
#     Custom autograd function for RNN CUDA kernel.
#     """
    
#     @staticmethod
#     def forward(ctx, A, Wa, J0, J1, Wo, r_init, W_delta7, alpha, activation_type):
#         """
#         Forward pass using CUDA kernel.
        
#         Args:
#             A: action_signal (batch_size, seq_len, action_dim)
#             Wa: (action_dim, num_neurons, num_neurons)
#             J0: scalar
#             J1: scalar
#             Wo: (num_neurons, num_neurons)
#             r_init: (batch_size, num_neurons)
#             W_delta7: (num_neurons, num_neurons)
#             alpha: scalar
#             activation_type: int (0=relu, 1=gelu, 2=tanh)
#         """
#         batch_size, seq_len, action_dim = A.shape
#         num_neurons = Wa.shape[1]
        
#         # Allocate output tensors
#         bump_history = torch.empty(
#             batch_size, seq_len, num_neurons, 
#             device=A.device, 
#             dtype=torch.float32
#         )
#         r_history = torch.empty(
#             batch_size, seq_len, num_neurons,
#             device=A.device,
#             dtype=torch.float32
#         )
        
#         # Call CUDA kernel
#         fwd_cuda.fwd(
#             A=A,
#             Wa=Wa,
#             J0=J0,
#             J1=J1,
#             Wo=Wo,
#             r_init=r_init,
#             W_delta7=W_delta7,
#             bump_history=bump_history,
#             r_history=r_history,
#             alpha=alpha,
#             activation_type=activation_type
#         )
        
#         # Save for backward
#         ctx.save_for_backward(A, Wa, Wo, W_delta7, bump_history, r_init)
#         ctx.alpha = alpha
#         ctx.J0 = J0
#         ctx.J1 = J1
#         ctx.activation_type = activation_type
        
#         return bump_history, r_history
    
#     @staticmethod
#     def backward(ctx, grad_bump_history, grad_r_history):
#         """
#         Backward pass using BPTT in PyTorch.
#         """
#         A, Wa, Wo, W_delta7, bump_history, r_init = ctx.saved_tensors
#         alpha = ctx.alpha
#         J0 = ctx.J0
#         J1 = ctx.J1
#         activation_type = ctx.activation_type
        
#         batch_size, seq_len, action_dim = A.shape
#         num_neurons = bump_history.shape[2]
        
#         # Initialize gradient accumulators
#         grad_Wa = torch.zeros_like(Wa)
#         grad_Wo = torch.zeros_like(Wo)
#         grad_A = torch.zeros_like(A)
        
#         # Total gradient w.r.t. bump_history (from both outputs if needed)
#         grad_bump = grad_bump_history.clone()
        
#         # BPTT through time
#         grad_r = torch.zeros(batch_size, num_neurons, device=A.device)
        
#         for t in range(seq_len - 1, -1, -1):
#             # Accumulate gradient from current timestep
#             grad_r = grad_r + grad_bump[:, t]
            
#             # Get state from previous timestep
#             r_prev = bump_history[:, t-1] if t > 0 else r_init
            
#             # Backprop through: r = r_prev * (1 - alpha) + recurrent_input * alpha
#             grad_recurrent_input = grad_r * alpha
            
#             # Reconstruct forward pass to get pre-activation values
#             A_t = A[:, t]  # (batch, action_dim)
            
#             # Compute Wa_weighted
#             A_t_expanded = A_t.unsqueeze(-1).unsqueeze(-1)  # (batch, action_dim, 1, 1)
#             Wa_weighted = torch.sum(A_t_expanded * Wa.unsqueeze(0), dim=1)  # (batch, N, N)
            
#             # Effective weight matrix
#             W_eff = J0 + J1 * Wo + Wa_weighted
            
#             # Recurrent input before activation
#             recurrent_input_pre_act = (W_eff @ r_prev.unsqueeze(2)).squeeze(2)
            
#             # Backprop through activation function
#             if activation_type == 0:  # relu
#                 grad_pre_act = grad_recurrent_input * (recurrent_input_pre_act > 0).float()
#             elif activation_type == 1:  # gelu
#                 x = recurrent_input_pre_act
#                 tanh_arg = 0.797885 * (x + 0.044715 * x ** 3)
#                 tanh_val = torch.tanh(tanh_arg)
#                 sech_sq = 1 - tanh_val ** 2
#                 grad_pre_act = grad_recurrent_input * (
#                     0.5 * (1 + tanh_val) + 
#                     0.5 * x * sech_sq * 0.797885 * (1 + 3 * 0.044715 * x ** 2)
#                 )
#             elif activation_type == 2:  # tanh
#                 grad_pre_act = grad_recurrent_input * (1 - torch.tanh(recurrent_input_pre_act) ** 2)
#             else:
#                 grad_pre_act = grad_recurrent_input
            
#             # Backprop through: recurrent_input_pre_act = W_eff @ r_prev
#             # Gradient w.r.t. W_eff
#             grad_W_eff = torch.bmm(
#                 grad_pre_act.unsqueeze(2),  # (batch, N, 1)
#                 r_prev.unsqueeze(1)  # (batch, 1, N)
#             )  # (batch, N, N)
            
#             # Accumulate gradients for learnable parameters
#             # grad w.r.t. Wo: sum over batch of J1 * grad_W_eff
#             grad_Wo += J1 * grad_W_eff.sum(0)
            
#             # grad w.r.t. Wa: sum over batch of A_t[k] * grad_W_eff for each k
#             for k in range(action_dim):
#                 grad_Wa[k] += (grad_W_eff * A_t[:, k].view(-1, 1, 1)).sum(0)
            
#             # grad w.r.t. action_signal at time t
#             for k in range(action_dim):
#                 grad_A[:, t, k] = (grad_W_eff * Wa[k].unsqueeze(0)).sum(dim=(1, 2))
            
#             # Backprop to r_prev through W_eff @ r_prev
#             grad_r_prev = torch.bmm(
#                 W_eff.transpose(1, 2),  # (batch, N, N)
#                 grad_pre_act.unsqueeze(2)  # (batch, N, 1)
#             ).squeeze(2)  # (batch, N)
            
#             # Add gradient from: r = r_prev * (1 - alpha) + ...
#             grad_r = grad_r_prev + grad_r * (1 - alpha)
        
#         # Return gradients for all inputs (None for non-differentiable inputs)
#         return grad_A, grad_Wa, None, None, grad_Wo, None, None, None, None
    
# class RnnCuda(nn.Module):
#     """
#     Standalone CUDA RNN layer that can be used independently.
#     This layer wraps the CUDA kernel and provides autograd support.
#     """
    
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, A, Wa, J0, J1, Wo, r_init, W_delta7, alpha, activation_type):
#         """
#         Forward pass through the RNN CUDA layer.
        
#         Args:
#             A: action_signal (batch_size, seq_len, action_dim)
#             Wa: (action_dim, num_neurons, num_neurons) - learnable
#             J0: scalar - fixed baseline connectivity
#             J1: scalar - fixed scaling factor
#             Wo: (num_neurons, num_neurons) - learnable recurrent weights
#             r_init: (batch_size, num_neurons) - initial state
#             W_delta7: (num_neurons, num_neurons) - output projection
#             alpha: scalar - integration rate
#             activation_type: int (0=relu, 1=gelu, 2=tanh)
            
#         Returns:
#             bump_history: (batch_size, seq_len, num_neurons)
#             r_history: (batch_size, seq_len, num_neurons) - filled by caller
#         """
#         return RnnCudaFunction.apply(
#             A, Wa, J0, J1, Wo, r_init, W_delta7, alpha, activation_type
#         )