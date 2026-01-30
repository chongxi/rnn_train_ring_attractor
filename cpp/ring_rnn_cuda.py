import torch
from torch.autograd import Function
import sys
from pathlib import Path

build_dir = Path(__file__).parent / "build"

try:
    sys.path.insert(0, str(build_dir))
    import fwd as fwd_cuda
    import bwd as bwd_cuda
    print("rnn_cuda Imported from prebuilt in cpp/build")
except ImportError:
    print("Prebuilt not found, falling back to JIT compilation...")
    from .rnn_cuda import fwd_cuda, bwd_cuda
    print("Successfully imported via JIT from rnn_cuda.py")

def ring_rnn_cuda_func(
        action_signal,
        Wa,
        J0,
        J1,
        Wo,
        r_init,
        alpha,
        activation='relu',
):
    """Ring RNN forward pass with CUDA acceleration.

    Args:
        action_signal: (batch_size, seq_len, action_dim) - input action sequence
        Wa: (action_dim, num_neurons, num_neurons) - action-modulated weight bank
        J0: scalar or (num_neurons, num_neurons) - baseline connectivity
        J1: scalar - scaling factor for Wo
        Wo: (num_neurons, num_neurons) - recurrent weight matrix
        r_init: (batch_size, num_neurons) - initial state
        alpha: float - update rate
        activation: str - activation function ('relu', 'gelu', 'tanh', 'silu')

    Returns:
        bump_history: (seq_len, batch_size, num_neurons) - neural state history
    """
    return RingRnnCudaFunc.apply(
        action_signal,
        Wa,
        J0,
        J1,
        Wo,
        r_init,
        alpha,
        activation,
    )


class RingRnnCudaFunc(Function):
    @staticmethod
    def forward(ctx, action_signal, Wa, J0, J1, Wo, r_init, alpha, activation_name):
        batch_size, seq_len, a_dim = action_signal.shape
        N = Wa.shape[1]

        r = r_init.clone()
        bump_history = torch.empty(seq_len, batch_size, N, device=action_signal.device, dtype=action_signal.dtype)
        activation_map = {'relu': 0, 'gelu': 1, 'tanh': 2, 'silu': 3}
        if activation_name not in activation_map:
            raise ValueError(f"Invalid activation '{activation_name}'. Must be one of {list(activation_map.keys())}.")
        activation_type = activation_map[activation_name]

        fwd_cuda.fwd(
            A=action_signal,
            Wa=Wa,
            J0=J0,
            J1=J1,
            Wo=Wo,
            r_init=r,
            bump_history=bump_history,
            alpha=alpha,
            activation_type=activation_type
        )

        ctx.save_for_backward(action_signal, Wa, Wo, bump_history, r_init)
        ctx.J0 = J0
        ctx.J1 = J1
        ctx.alpha = alpha
        ctx.activation_name = activation_name

        return bump_history

    @staticmethod
    def backward(ctx, grad_output):
        action_signal, Wa, Wo, bump_history, r_init = ctx.saved_tensors
        alpha = ctx.alpha
        activation_name = ctx.activation_name
        J0 = ctx.J0
        J1 = ctx.J1
        batch_size, seq_len, a_dim = action_signal.shape
        N = Wa.shape[1]

        grad_Wa = torch.zeros_like(Wa)
        grad_Wo = torch.zeros_like(Wo)

        activation_map = {'relu': 0, 'gelu': 1, 'tanh': 2, 'silu': 3}
        if activation_name not in activation_map:
            raise ValueError(f"Invalid activation '{activation_name}'. Must be one of {list(activation_map.keys())}.")
        activation_type = activation_map[activation_name]

        recurrent_input_history_dummy = torch.empty(1, device=grad_output.device, dtype=torch.bfloat16)

        # Workspace buffers - use empty since they're completely overwritten
        grad_r = torch.zeros(batch_size, N, device=grad_output.device, dtype=grad_output.dtype)
        W_eff_temp = torch.empty(batch_size, N, N, device=grad_output.device, dtype=grad_output.dtype)
        grad_re_temp = torch.empty(batch_size, N, device=grad_output.device, dtype=grad_output.dtype)
        grad_W_eff_temp_bf16 = torch.empty(batch_size, N, N, device=grad_output.device, dtype=torch.bfloat16)
        A_t_temp_bf16 = torch.empty(batch_size, a_dim, device=grad_output.device, dtype=torch.bfloat16)
        grad_re_temp_T = torch.empty(batch_size, N, device=grad_output.device, dtype=grad_output.dtype)  # NEW: added this buffer

        grad_output_ctg = grad_output.contiguous()

        handle = torch.cuda.current_blas_handle()

        bwd_cuda.bwd(
            grad_output=grad_output_ctg,
            A=action_signal,
            Wa=Wa,
            J0=J0,
            J1=J1,
            Wo=Wo,
            bump_history=bump_history,
            re_history=recurrent_input_history_dummy,
            r_init=r_init,
            grad_Wa=grad_Wa,
            grad_Wo=grad_Wo,
            alpha=alpha,
            activation_type=activation_type,
            # Workspace buffers
            grad_r=grad_r,
            W_eff_temp=W_eff_temp,
            grad_re_temp=grad_re_temp,
            grad_W_eff_temp_bf16=grad_W_eff_temp_bf16,
            A_t_temp_bf16=A_t_temp_bf16,
            grad_re_temp_T=grad_re_temp_T,
            cublas_handle=handle
        )

        return None, grad_Wa, None, None, grad_Wo, None, None, None

