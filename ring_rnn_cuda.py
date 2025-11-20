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
    activation_map = {'relu': 0, 'gelu': 1, 'tanh': 2, 'silu': 3}
    if activation not in activation_map:
        raise ValueError(f"Invalid activation '{activation}'. Must be one of {list(activation_map.keys())}.")
    activation_type = activation_map[activation]

    return RingRnnCudaFunc.apply(
        action_signal,
        Wa,
        J0,
        J1,
        Wo,
        r_init,
        alpha,
        activation_type,
    )


class RingRnnCudaFunc(Function):
    @staticmethod
    def forward(
            ctx,
            action_signal,
            Wa,
            J0,
            J1,
            Wo,
            r_init,
            alpha,
            activation_type,
    ):
        batch_size, seq_len, action_dim = action_signal.shape
        num_neurons = Wa.shape[1]

        bump_history = torch.empty(
            seq_len + 1, batch_size, num_neurons,
            device=action_signal.device,
            dtype=torch.float32
        )

        bump_history[0, :, :] = r_init

        _ring_rnn_forward(
            action_signal,
            Wa,
            J0,
            J1,
            Wo,
            r_init,
            bump_history,
            alpha,
            activation_type,
        )

        ctx.save_for_backward(action_signal, Wa, Wo, bump_history)
        ctx.J0 = J0
        ctx.J1 = J1
        ctx.alpha = alpha
        ctx.activation_type = activation_type

        return bump_history[1:, :, :]
    @staticmethod
    def backward(ctx, grad_bump_history):
        print("grad_output.shape: ", grad_bump_history.shape)
        action_signal, Wa, Wo, bump_history = ctx.saved_tensors
        J0 = ctx.J0
        J1 = ctx.J1
        alpha = ctx.alpha
        activation_type = ctx.activation_type

        grad_Wa, grad_Wo = _ring_rnn_backward(
            grad_bump_history,
            action_signal,
            Wa,
            J0,
            J1,
            Wo,
            bump_history,
            alpha,
            activation_type,
        )

        return None, grad_Wa, None, None, grad_Wo, None, None, None


def _ring_rnn_forward(
        action_signal,
        Wa,
        J0,
        J1,
        Wo,
        r_init,
        bump_history,
        alpha,
        activation_type,
):
    fwd_cuda.fwd(
        A=action_signal,
        Wa=Wa,
        J0=J0,
        J1=J1,
        Wo=Wo,
        r_init=r_init,
        bump_history=bump_history,
        alpha=alpha,
        activation_type=activation_type,
    )


def _ring_rnn_backward(
        grad_output,
        action_signal,
        Wa,
        J0,
        J1,
        Wo,
        bump_history,
        alpha,
        activation_type,
):
    grad_Wa = torch.zeros_like(Wa)
    grad_Wo = torch.zeros_like(Wo)

    bwd_cuda.bwd(
        grad_output=grad_output,
        A=action_signal,
        Wa=Wa,
        J0=J0,
        J1=J1,
        Wo=Wo,
        bump_history=bump_history,
        grad_Wa=grad_Wa,
        grad_Wo=grad_Wo,
        alpha=alpha,
        activation_type=activation_type
    )

    return grad_Wa, grad_Wo