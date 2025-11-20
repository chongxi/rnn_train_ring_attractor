import torch
from torch.autograd import Function
from rnn_cuda import fwd_cuda, bwd_cuda
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
    # activation_map = {'relu': 0, 'gelu': 1, 'tanh': 2, 'silu': 3}
    # if activation not in activation_map:
    #     raise ValueError(f"Invalid activation '{activation}'. Must be one of {list(activation_map.keys())}.")
    # activation_type = activation_map[activation]

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

        # for t in range(seq_len):
        #     A_t = action_signal[:, t, :]
        #     Wa_flat = Wa.view(a_dim, N * N)
        #     Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, N, N)
        #     W_eff = J0 + J1 * Wo + Wa_weighted
        #     recurrent_input = (W_eff @ r.unsqueeze(2)).squeeze(2)
        #     recurrent_input_activated = non_linear(recurrent_input, activation_name)
        #     r = r * (1 - alpha) + recurrent_input_activated * alpha
        #     bump_history[t] = r


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
            # W_delta7=self.W_delta7,
            bump_history=bump_history,
            # r_history=self.r_history,
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

        bwd_cuda.bwd(
            grad_output=grad_output.contiguous(),
            A=action_signal,
            Wa=Wa,
            J0=J0,
            J1=J1,
            Wo=Wo,
            bump_history=bump_history,
            r_init=r_init,
            grad_Wa=grad_Wa,
            grad_Wo=grad_Wo,
            alpha=alpha,
            activation_type=activation_type
        )
        return None, grad_Wa, None, None, grad_Wo, None, None, None

