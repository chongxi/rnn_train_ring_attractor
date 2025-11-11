import torch
from torch.autograd import Function
from utils.benchmark import *

import torch
from torch.autograd import Function

class CalcBumpFunction(Function):
    @staticmethod
    def forward(ctx, action_signal, Wa, J0, J1, Wo, r_init, alpha, activation_name):
        """
        Forward pass for calc_bump.

        Args:
            action_signal: (batch_size, seq_len, a_dim)
            Wa: (a_dim, N, N) - differentiable
            J0: (N, N) - not differentiable
            J1: scalar - not differentiable
            Wo: (N, N) - differentiable
            r_init: (batch_size, N) - not differentiable
            alpha: scalar - not differentiable
            activation_name: string - activation function name

        Returns:
            bump_history: (batch_size, seq_len, N)
        """
        batch_size, seq_len, a_dim = action_signal.shape
        N = Wa.shape[1]

        r = r_init.clone()
        # Start with r_init so we have [r_0, r_1, r_2, ..., r_T]
        bump_history = [r_init]

        for t in range(seq_len):
            A_t = action_signal[:, t, :]

            # Compute Wa contribution
            Wa_flat = Wa.view(a_dim, N * N)
            Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, N, N)

            # Effective weight matrix
            W_eff = J0 + J1 * Wo + Wa_weighted

            # Recurrent computation
            recurrent_input = (W_eff @ r.unsqueeze(2)).squeeze(2)

            # Apply activation
            recurrent_input_activated = non_linear(recurrent_input, activation_name)

            # Update state
            r = r * (1 - alpha) + recurrent_input_activated * alpha
            bump_history.append(r)

        # Stack: bump_history_full = [r_0, r_1, ..., r_T] shape (batch_size, seq_len+1, N)
        bump_history_full = torch.stack(bump_history, dim=1)

        # Save full history for backward (includes r_init at index 0)
        ctx.save_for_backward(action_signal, Wa, J0, Wo, bump_history_full)
        ctx.J1 = J1
        ctx.alpha = alpha
        ctx.activation_name = activation_name
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.a_dim = a_dim
        ctx.N = N

        # Return only [r_1, r_2, ..., r_T] - exclude r_init
        return bump_history_full[:, 1:, :]

    @staticmethod
    def backward(ctx, grad_output):
        action_signal, Wa, J0, Wo, bump_history_full = ctx.saved_tensors
        J1 = ctx.J1
        alpha = ctx.alpha
        activation_name = ctx.activation_name
        batch_size = ctx.batch_size
        seq_len = ctx.seq_len
        a_dim = ctx.a_dim
        N = ctx.N

        # Initialize gradients
        grad_Wa = torch.zeros_like(Wa)
        grad_Wo = torch.zeros_like(Wo)
        grad_r = torch.zeros(batch_size, N, device=grad_output.device, dtype=grad_output.dtype)

        # Backward pass through time
        for t in reversed(range(seq_len)):
            # Gradient from output at time t
            grad_r = grad_r + grad_output[:, t, :]

            # Gradient w.r.t. recurrent_input_activated
            grad_recurrent_activated = grad_r * alpha

            # Get r_{t-1} from bump_history_full
            # bump_history_full[:, 0] = r_0 (r_init)
            # bump_history_full[:, t] = r_t
            # So r_{t-1} is at bump_history_full[:, t]
            r_prev = bump_history_full[:, t, :]

            # Recompute W_eff for this timestep
            A_t = action_signal[:, t, :]
            Wa_flat = Wa.view(a_dim, N * N)
            Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, N, N)
            W_eff = J0 + J1 * Wo + Wa_weighted

            # Recompute recurrent_input for gradient through activation
            recurrent_input = (W_eff @ r_prev.unsqueeze(2)).squeeze(2)

            # Gradient through activation function
            grad_recurrent_input = non_linear_derivative(
                recurrent_input,
                grad_recurrent_activated,
                activation_name
            )

            # Gradient w.r.t. r_{t-1}
            grad_r_from_recurrent = (W_eff.transpose(1, 2) @ grad_recurrent_input.unsqueeze(2)).squeeze(2)

            # Gradient w.r.t. W_eff
            grad_W_eff = torch.bmm(
                grad_recurrent_input.unsqueeze(2),
                r_prev.unsqueeze(1)
            )

            # Gradient w.r.t. Wo
            grad_Wo = grad_Wo + J1 * grad_W_eff.sum(dim=0)

            # Gradient w.r.t. Wa
            grad_Wa_weighted_flat = grad_W_eff.view(batch_size, N * N)
            grad_Wa_flat = torch.matmul(A_t.t(), grad_Wa_weighted_flat)
            grad_Wa = grad_Wa + grad_Wa_flat.view(a_dim, N, N)

            # Propagate gradient to r_{t-1}
            grad_r = grad_r_from_recurrent + grad_r * (1 - alpha)

        return None, grad_Wa, None, None, grad_Wo, None, None, None

class CalcBumpFunctionSave(Function):
    @staticmethod
    def forward(ctx, action_signal, Wa, J0, J1, Wo, r_init, alpha, activation_name):
        """
        Forward pass for calc_bump - saves everything to avoid recomputation in backward.

        Args:
            action_signal: (batch_size, seq_len, a_dim)
            Wa: (a_dim, N, N) - differentiable
            J0: (N, N) - not differentiable
            J1: scalar - not differentiable
            Wo: (N, N) - differentiable
            r_init: (batch_size, N) - not differentiable
            alpha: scalar - not differentiable
            activation_name: string - activation function name

        Returns:
            bump_history: (batch_size, seq_len, N)
        """
        batch_size, seq_len, a_dim = action_signal.shape
        N = Wa.shape[1]

        r = r_init.clone()
        bump_history = [r_init]
        W_eff_history = []
        recurrent_input_history = []
        Wa_weighted_history = []

        for t in range(seq_len):
            A_t = action_signal[:, t, :]

            # Compute Wa contribution
            Wa_flat = Wa.view(a_dim, N * N)
            Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, N, N)
            Wa_weighted_history.append(Wa_weighted)

            # Effective weight matrix
            W_eff = J0 + J1 * Wo + Wa_weighted
            W_eff_history.append(W_eff)

            # Recurrent computation
            recurrent_input = (W_eff @ r.unsqueeze(2)).squeeze(2)
            recurrent_input_history.append(recurrent_input)

            # Apply activation
            recurrent_input_activated = non_linear(recurrent_input, activation_name)

            # Update state
            r = r * (1 - alpha) + recurrent_input_activated * alpha
            bump_history.append(r)

        # Stack all histories
        # bump_history_full: [r_0, r_1, ..., r_T] shape (batch_size, seq_len+1, N)
        bump_history_full = torch.stack(bump_history, dim=1)
        # W_eff_history: [W_0, W_1, ..., W_{T-1}] shape (batch_size, seq_len, N, N)
        W_eff_history_stacked = torch.stack(W_eff_history, dim=1)
        # recurrent_input_history: [h_0, h_1, ..., h_{T-1}] shape (batch_size, seq_len, N)
        recurrent_input_history_stacked = torch.stack(recurrent_input_history, dim=1)
        # Wa_weighted_history: [Wa_0, Wa_1, ..., Wa_{T-1}] shape (batch_size, seq_len, N, N)
        Wa_weighted_history_stacked = torch.stack(Wa_weighted_history, dim=1)

        # Save everything needed for backward
        ctx.save_for_backward(
            action_signal,
            bump_history_full,
            W_eff_history_stacked,
            recurrent_input_history_stacked,
            Wa_weighted_history_stacked
        )
        ctx.J1 = J1
        ctx.alpha = alpha
        ctx.activation_name = activation_name
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.a_dim = a_dim
        ctx.N = N

        # Return only [r_1, r_2, ..., r_T] - exclude r_init
        return bump_history_full[:, 1:, :]

    @staticmethod
    def backward(ctx, grad_output):
        (action_signal, bump_history_full, W_eff_history,
         recurrent_input_history, Wa_weighted_history) = ctx.saved_tensors
        J1 = ctx.J1
        alpha = ctx.alpha
        activation_name = ctx.activation_name
        batch_size = ctx.batch_size
        seq_len = ctx.seq_len
        a_dim = ctx.a_dim
        N = ctx.N

        # Initialize gradients
        grad_Wa = torch.zeros(a_dim, N, N, device=grad_output.device, dtype=grad_output.dtype)
        grad_Wo = torch.zeros(N, N, device=grad_output.device, dtype=grad_output.dtype)
        grad_r = torch.zeros(batch_size, N, device=grad_output.device, dtype=grad_output.dtype)

        # Backward pass through time
        for t in reversed(range(seq_len)):
            # Gradient from output at time t
            grad_r = grad_r + grad_output[:, t, :]

            # Gradient w.r.t. recurrent_input_activated
            grad_recurrent_activated = grad_r * alpha

            # Get saved values (no recomputation needed)
            r_prev = bump_history_full[:, t, :]
            W_eff = W_eff_history[:, t, :, :]
            recurrent_input = recurrent_input_history[:, t, :]

            # Gradient through activation function
            grad_recurrent_input = non_linear_derivative(
                recurrent_input,
                grad_recurrent_activated,
                activation_name
            )

            # Gradient w.r.t. r_{t-1}
            grad_r_from_recurrent = (W_eff.transpose(1, 2) @ grad_recurrent_input.unsqueeze(2)).squeeze(2)

            # Gradient w.r.t. W_eff
            grad_W_eff = torch.bmm(
                grad_recurrent_input.unsqueeze(2),
                r_prev.unsqueeze(1)
            )

            # Gradient w.r.t. Wo
            grad_Wo = grad_Wo + J1 * grad_W_eff.sum(dim=0)

            # Gradient w.r.t. Wa
            # Since W_eff = J0 + J1 * Wo + Wa_weighted
            # and Wa_weighted = A_t @ Wa_flat reshaped
            # grad_Wa_weighted = grad_W_eff (from the chain rule)
            A_t = action_signal[:, t, :]
            grad_Wa_weighted_flat = grad_W_eff.view(batch_size, N * N)
            grad_Wa_flat = torch.matmul(A_t.t(), grad_Wa_weighted_flat)
            grad_Wa = grad_Wa + grad_Wa_flat.view(a_dim, N, N)

            # Propagate gradient to r_{t-1}
            grad_r = grad_r_from_recurrent + grad_r * (1 - alpha)

        return None, grad_Wa, None, None, grad_Wo, None, None, None

def non_linear(x, activation_name):
    """Apply activation function."""
    if activation_name == 'tanh':
        return torch.tanh(x)
    elif activation_name == 'relu':
        return torch.relu(x)
    elif activation_name == 'sigmoid':
        return torch.sigmoid(x)
    elif activation_name == 'linear' or activation_name is None:
        return x
    else:
        raise ValueError(f"Unknown activation: {activation_name}")


def non_linear_derivative(x, grad_output, activation_name):
    """
    Compute gradient through activation function.

    Args:
        x: pre-activation input
        grad_output: gradient from upstream
        activation_name: name of activation function

    Returns:
        gradient w.r.t. x
    """
    if activation_name == 'tanh':
        # d/dx tanh(x) = 1 - tanh(x)^2
        return grad_output * (1 - torch.tanh(x) ** 2)
    elif activation_name == 'relu':
        # d/dx relu(x) = 1 if x > 0 else 0
        return grad_output * (x > 0).float()
    elif activation_name == 'sigmoid':
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        sig = torch.sigmoid(x)
        return grad_output * sig * (1 - sig)
    elif activation_name == 'linear' or activation_name is None:
        return grad_output
    else:
        raise ValueError(f"Unknown activation: {activation_name}")


def calc_bump_impl(action_signal, Wa, J0, J1, Wo, r_init, alpha, activation_name):
    """
    Wrapper function using custom autograd.

    Args:
        action_signal: (batch_size, seq_len, a_dim)
        Wa: (a_dim, N, N) - differentiable
        J0: (N, N) - not differentiable
        J1: scalar - not differentiable
        Wo: (N, N) - differentiable
        r_init: (batch_size, N) - not differentiable
        alpha: scalar (0 to 1) - not differentiable
        activation_name: str - activation function name

    Returns:
        bump_history: (batch_size, seq_len, N)
    """
    return CalcBumpFunction.apply(
        action_signal, Wa, J0, J1, Wo, r_init, alpha, activation_name
    )

def calc_bump_save_impl(action_signal, Wa, J0, J1, Wo, r_init, alpha, activation_name):
    return CalcBumpFunctionSave.apply(
        action_signal, Wa, J0, J1, Wo, r_init, alpha, activation_name
    )


def param_prep(num_neurons, a_dim, device='cpu'):
    Wa = torch.randn(a_dim, num_neurons, num_neurons, device=device, requires_grad=True)
    J0 = torch.randn(num_neurons, num_neurons, device=device, requires_grad=False)
    J1 = torch.rand(1, device=device).item()
    Wo = torch.randn(num_neurons, num_neurons, device=device, requires_grad=True)

    return Wa, J0, J1, Wo


def input_prep(num_neurons, batch_size, a_dim, seq_len, device='cpu'):
    action_signal = torch.randn(batch_size, seq_len, a_dim, device=device)
    r_init = torch.randn(batch_size, num_neurons, device=device)
    bump_history = torch.empty(batch_size, seq_len, num_neurons, device=device)

    return action_signal, r_init, bump_history

def calc_bump_ref(action_signal, Wa, J0, J1, Wo, r_init, alpha, activation_name):
    """Original calc_bump for reference."""
    batch_size, seq_len, a_dim = action_signal.shape
    N = Wa.shape[1]
    r = r_init.clone()
    bump_history = []

    for t in range(seq_len):
        A_t = action_signal[:, t, :]
        Wa_flat = Wa.view(a_dim, N * N)
        Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, N, N)

        W_eff = J0 + J1 * Wo + Wa_weighted
        recurrent_input = (W_eff @ r.unsqueeze(2)).squeeze(2)
        recurrent_input = non_linear(recurrent_input, activation_name)
        r = r * (1 - alpha) + recurrent_input * alpha
        bump_history.append(r)

    return torch.stack(bump_history, dim=1)

if __name__ == "__main__":

    # Small test case for gradcheck
    num_neur = 8
    a_dim = 4
    seq_len = 10
    bs = 4
    act = "tanh"
    alpha = 0.15

    set_seed(42)
    Wa, J0, J1, Wo = param_prep(num_neurons=num_neur, a_dim=a_dim, device='cpu')
    action_signal, r_init, _ = input_prep(num_neurons=num_neur, batch_size=bs, a_dim=a_dim, seq_len=seq_len, device='cpu')

    # Convert to double for gradcheck
    Wa = Wa.double().requires_grad_(True)
    Wo = Wo.double().requires_grad_(True)
    action_signal = action_signal.double()
    r_init = r_init.double()
    J0 = J0.double()

    def calc_bump_wrapper(wa, wo):
        return calc_bump_impl(action_signal, wa, J0, J1, wo, r_init, alpha, act)

    test = torch.autograd.gradcheck(calc_bump_wrapper, (Wa, Wo))

    print(f"torch.autograd.gradcheck passed: {test}")

    num_neur = 512
    a_dim = 32
    act = "relu"
    bs = 4
    seq_len = 2
    alpha = 0.15
    device = "cuda"

    set_seed(42)
    Wa_impl, J0_impl, J1_impl, Wo_impl = param_prep(num_neurons=num_neur, a_dim=a_dim, device=device)
    action_signal_impl, r_init_impl, _ = input_prep(num_neurons=num_neur, batch_size=bs, a_dim=a_dim, seq_len=seq_len, device=device)

    set_seed(42)
    Wa_ref, J0_ref, J1_ref, Wo_ref = param_prep(num_neurons=num_neur, a_dim=a_dim, device=device)
    action_signal_ref, r_init_ref, _ = input_prep(num_neurons=num_neur, batch_size=bs, a_dim=a_dim, seq_len=seq_len, device=device)


    result_impl = calc_bump_ref(action_signal_impl, Wa_impl, J0_impl, J1_impl, Wo_impl, r_init_impl, alpha, act)
    result_ref = calc_bump_impl(action_signal_ref, Wa_ref, J0_ref, J1_ref, Wo_ref, r_init_ref, alpha, act)

    gt = torch.randn_like(result_impl, device=device)
    gt_impl = gt.clone().detach()
    check_tensor_match(tsr_ref=result_ref, tsr_impl=result_impl, name="Forward")

    criterion = torch.nn.MSELoss()

    loss_ref = criterion(result_ref, gt)
    loss_ref.backward()
    grad_Wa_ref = Wa_ref.grad
    grad_Wo_ref = Wo_ref.grad

    loss_impl = criterion(result_impl, gt_impl)
    loss_impl.backward()
    grad_Wa_impl = Wa_impl.grad
    grad_Wo_impl = Wo_impl.grad

    check_tensor_match(tsr_ref=grad_Wa_ref, tsr_impl=grad_Wa_impl, name="Backward Wa", atol=1e-6, max_print=10)
    check_tensor_match(tsr_ref=grad_Wo_ref, tsr_impl=grad_Wo_impl, name="Backward Wo", atol=1e-6, max_print=10)

    # ============ LATENCY BENCHMARKING ============
    print("\n" + "="*60)
    print("LATENCY BENCHMARKING")
    print("="*60)

    # Forward pass latency
    print("\n--- Forward Pass ---")
    latency_ref_fwd = measure_latency_cuda(
        calc_bump_ref,
        action_signal_ref, Wa_ref, J0_ref, J1_ref,
        Wo_ref, r_init_ref, alpha, act,
        n_warmup=5, n_iters=50
    )
    print(f"Reference:      {latency_ref_fwd}")

    latency_impl_fwd = measure_latency_cuda(
        calc_bump_impl,
        action_signal_impl, Wa_impl, J0_impl, J1_impl,
        Wo_impl, r_init_impl, alpha, act,
        n_warmup=5, n_iters=50
    )
    print(f"Implementation: {latency_impl_fwd}")

    # Forward + Backward pass latency
    print("\n--- Forward + Backward Pass ---")

    def forward_backward_ref():
        if Wa_ref.grad is not None:
            Wa_ref.grad.zero_()
        if Wo_ref.grad is not None:
            Wo_ref.grad.zero_()

        result = calc_bump_ref(action_signal_ref, Wa_ref, J0_ref,
                               J1_ref, Wo_ref, r_init_ref, alpha, act)
        loss = criterion(result, gt)
        loss.backward()

    def forward_backward_impl():
        if Wa_impl.grad is not None:
            Wa_impl.grad.zero_()
        if Wo_impl.grad is not None:
            Wo_impl.grad.zero_()

        result = calc_bump_impl(action_signal_impl, Wa_impl, J0_impl,
                                J1_impl, Wo_impl, r_init_impl, alpha, act)
        loss = criterion(result, gt_impl)
        loss.backward()

    latency_ref_bwd = measure_latency_cuda(forward_backward_ref, n_warmup=5, n_iters=50)
    print(f"Reference:      {latency_ref_bwd}")

    latency_impl_bwd = measure_latency_cuda(forward_backward_impl, n_warmup=5, n_iters=50)
    print(f"Implementation: {latency_impl_bwd}")
