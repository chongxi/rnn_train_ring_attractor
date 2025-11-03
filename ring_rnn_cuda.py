import torch
from torch.autograd import Function
from rnn_cuda import fwd_cuda

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
            seq_len, batch_size, num_neurons,
            device=action_signal.device,
            dtype=torch.float32
        )

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

        ctx.save_for_backward(action_signal, Wa, Wo, r_init)
        ctx.J0 = J0
        ctx.J1 = J1
        ctx.alpha = alpha
        ctx.activation_type = activation_type

        return bump_history

    @staticmethod
    def backward(ctx, grad_bump_history):
        action_signal, Wa, Wo, r_init = ctx.saved_tensors
        J0 = ctx.J0
        J1 = ctx.J1
        alpha = ctx.alpha
        activation_type = ctx.activation_type

        grad_action, grad_Wa, grad_Wo = _ring_rnn_backward(
            grad_bump_history,
            action_signal,
            Wa,
            J0,
            J1,
            Wo,
            r_init,
            alpha,
            activation_type,
        )

        return grad_action, grad_Wa, None, None, grad_Wo, None, None, None


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
        grad_bump_history,
        action_signal,
        Wa,
        J0,
        J1,
        Wo,
        r_init,
        alpha,
        activation_type,
):
    """Pure PyTorch backward pass - fully recompute forward pass."""
    seq_len, batch_size, num_neurons = grad_bump_history.shape
    action_dim = Wa.shape[0]

    # Recompute entire forward pass
    bump_history_recomputed = torch.zeros(seq_len, batch_size, num_neurons, device=action_signal.device)
    r = r_init.clone()

    for t in range(seq_len):
        A_t = action_signal[:, t, :]

        Wa_flat = Wa.view(action_dim, num_neurons * num_neurons)
        Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, num_neurons, num_neurons)
        W_eff = J0 + J1 * Wo + Wa_weighted
        recurrent_input_pre_act = (W_eff @ r.unsqueeze(2)).squeeze(2)

        # Apply activation
        if activation_type == 0:  # relu
            recurrent_input = torch.relu(recurrent_input_pre_act)
        elif activation_type == 1:  # gelu
            recurrent_input = torch.nn.functional.gelu(recurrent_input_pre_act)
        elif activation_type == 2:  # tanh
            recurrent_input = torch.tanh(recurrent_input_pre_act)
        elif activation_type == 3:  # silu
            recurrent_input = torch.nn.functional.silu(recurrent_input_pre_act)

        r = r * (1 - alpha) + recurrent_input * alpha
        bump_history_recomputed[t] = r

    # Now do backward pass
    grad_action = torch.zeros_like(action_signal)
    grad_Wa = torch.zeros_like(Wa)
    grad_Wo = torch.zeros_like(Wo)
    grad_r = torch.zeros(batch_size, num_neurons, device=action_signal.device)

    for t in reversed(range(seq_len)):
        grad_r = grad_r + grad_bump_history[t]

        A_t = action_signal[:, t, :]

        if t > 0:
            r_prev = bump_history_recomputed[t-1]
        else:
            r_prev = r_init

        Wa_flat = Wa.view(action_dim, num_neurons * num_neurons)
        Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, num_neurons, num_neurons)
        W_eff = J0 + J1 * Wo + Wa_weighted
        recurrent_input_pre_act = (W_eff @ r_prev.unsqueeze(2)).squeeze(2)

        grad_recurrent_input = grad_r * alpha
        grad_r_prev = grad_r * (1 - alpha)

        grad_recurrent_pre_act = grad_recurrent_input * activation_derivative(
            recurrent_input_pre_act, activation_type
        )

        grad_W_eff = torch.bmm(grad_recurrent_pre_act.unsqueeze(2), r_prev.unsqueeze(1))
        grad_r_prev = grad_r_prev + torch.bmm(W_eff.transpose(1, 2), grad_recurrent_pre_act.unsqueeze(2)).squeeze(2)

        grad_Wo = grad_Wo + (grad_W_eff * J1).sum(dim=0)

        grad_A_t = torch.matmul(grad_W_eff.view(batch_size, -1), Wa_flat.T)
        grad_action[:, t, :] = grad_A_t

        grad_Wa_flat = torch.matmul(A_t.T, grad_W_eff.view(batch_size, -1))
        grad_Wa = grad_Wa + grad_Wa_flat.view_as(Wa)

        grad_r = grad_r_prev

    return grad_action, grad_Wa, grad_Wo


def activation_derivative(x, activation_type):
    """Compute activation derivative."""
    import math

    if activation_type == 0:  # relu
        return (x > 0).float()
    elif activation_type == 1:  # gelu
        # GELU(x) = x * Φ(x)
        # Derivative: Φ(x) + x * φ(x)
        # where Φ(x) = CDF, φ(x) = PDF of standard normal
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return cdf + x * pdf
    elif activation_type == 2:  # tanh
        return 1 - torch.tanh(x) ** 2
    elif activation_type == 3:  # silu
        # SiLU(x) = x * sigmoid(x)
        # Derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sigmoid = torch.sigmoid(x)
        return sigmoid * (1 + x * (1 - sigmoid))
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")