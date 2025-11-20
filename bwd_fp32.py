import torch
from torch.autograd import Function
from utils.benchmark import *
from rnn_cuda import bwd_cuda, fwd_cuda

from ring_rnn_cuda import ring_rnn_cuda_func

import torch
from torch.autograd import Function
torch.set_printoptions(linewidth=200)
# np.set_printoptions(linewidth=200, precision=6, suppress=True)

np.set_printoptions(
    precision=4,      # 6 decimal places
    suppress=True,    # Don't use scientific notation for small numbers
    linewidth=200,    # Wider lines
    formatter={'float': lambda x: f'{x:>12.4f}'})

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
        bump_history = []
        recurrent_input_history = []  # 👈 NEW: Save pre-activation values

        for t in range(seq_len):
            A_t = action_signal[:, t, :]
            Wa_flat = Wa.view(a_dim, N * N)
            Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, N, N)
            W_eff = J0 + J1 * Wo + Wa_weighted
            recurrent_input = (W_eff @ r.unsqueeze(2)).squeeze(2)
            recurrent_input_history.append(recurrent_input)  # 👈 Save it
            recurrent_input_activated = non_linear(recurrent_input, activation_name)
            r = r * (1 - alpha) + recurrent_input_activated * alpha
            bump_history.append(r)

        bump_history = torch.stack(bump_history, dim=1)
        recurrent_input_history = torch.stack(recurrent_input_history, dim=1)  # [batch, seq_len, N]
        ctx.save_for_backward(action_signal, Wa, Wo, bump_history, recurrent_input_history, r_init)
        ctx.J0 = J0
        ctx.J1 = J1
        ctx.alpha = alpha
        ctx.activation_name = activation_name

        return bump_history

    @staticmethod
    def backward(ctx, grad_output):
        action_signal, Wa, Wo, bump_history, recurrent_input_history, r_init = ctx.saved_tensors
        alpha = ctx.alpha
        activation_name = ctx.activation_name
        J0 = ctx.J0
        J1 = ctx.J1
        batch_size, seq_len, a_dim = action_signal.shape
        N = Wa.shape[1]

        grad_Wa = torch.zeros_like(Wa)
        grad_Wo = torch.zeros_like(Wo)

        # Initialize grad_r for the NEXT timestep (starts at 0 after the sequence)
        grad_r_next = torch.zeros(batch_size, N, device=grad_output.device, dtype=grad_output.dtype)

        for t in reversed(range(seq_len)):
            # Get r_prev: either from bump_history[t-1] or r_init if t==0
            if t == 0:
                r_prev = r_init
            else:
                r_prev = bump_history[:, t - 1, :]
            recurrent_input = recurrent_input_history[:, t, :]
            A_t = action_signal[:, t, :]
            Wa_flat = Wa.view(a_dim, N * N)
            Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, N, N)
            W_eff = J0 + J1 * Wo + Wa_weighted
            grad_r_current = grad_output[:, t, :] + grad_r_next
            activation_derivative = non_linear_derivative(recurrent_input, activation_name)
            grad_recurrent_input = activation_derivative * grad_r_current * alpha
            grad_W_eff = torch.bmm(
                grad_recurrent_input.unsqueeze(2),
                r_prev.unsqueeze(1)
            )
            grad_Wo = grad_Wo + J1 * grad_W_eff.sum(dim=0)
            grad_Wa_weighted_flat = grad_W_eff.view(batch_size, N * N)
            grad_Wa_flat = torch.matmul(A_t.t(), grad_Wa_weighted_flat)
            grad_Wa = grad_Wa + grad_Wa_flat.view(a_dim, N, N)
            grad_r_from_recurrent = (W_eff.transpose(1, 2) @ grad_recurrent_input.unsqueeze(2)).squeeze(2)
            grad_r_next = grad_r_from_recurrent + grad_r_current * (1 - alpha)

        return None, grad_Wa, None, None, grad_Wo, None, None, None

class CalcBumpFunction_permute_re(Function):
    @staticmethod
    def forward(ctx, action_signal, Wa, J0, J1, Wo, r_init, alpha, activation_name):
        batch_size, seq_len, a_dim = action_signal.shape
        N = Wa.shape[1]

        r = r_init.clone()
        bump_history = torch.empty(seq_len, batch_size, N, device=action_signal.device, dtype=action_signal.dtype)
        recurrent_input_history = torch.empty(seq_len, batch_size, N, device=action_signal.device, dtype=action_signal.dtype)

        for t in range(seq_len):
            A_t = action_signal[:, t, :]
            Wa_flat = Wa.view(a_dim, N * N)
            Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, N, N)
            W_eff = J0 + J1 * Wo + Wa_weighted
            recurrent_input = (W_eff @ r.unsqueeze(2)).squeeze(2)
            recurrent_input_history[t] = recurrent_input
            recurrent_input_activated = non_linear(recurrent_input, activation_name)
            r = r * (1 - alpha) + recurrent_input_activated * alpha
            bump_history[t] = r

        ctx.save_for_backward(action_signal, Wa, Wo, bump_history, recurrent_input_history, r_init)
        ctx.J0 = J0
        ctx.J1 = J1
        ctx.alpha = alpha
        ctx.activation_name = activation_name

        return bump_history

    @staticmethod
    def backward(ctx, grad_output):
        action_signal, Wa, Wo, bump_history, recurrent_input_history, r_init = ctx.saved_tensors
        alpha = ctx.alpha
        activation_name = ctx.activation_name
        J0 = ctx.J0
        J1 = ctx.J1
        batch_size, seq_len, a_dim = action_signal.shape
        N = Wa.shape[1]

        grad_Wa = torch.zeros_like(Wa)
        grad_Wo = torch.zeros_like(Wo)

        grad_r_next = torch.zeros(batch_size, N, device=grad_output.device, dtype=grad_output.dtype)

        for t in reversed(range(seq_len)):
            if t == 0:
                r_prev = r_init
            else:
                r_prev = bump_history[t - 1]

            recurrent_input = recurrent_input_history[t]

            A_t = action_signal[:, t, :]
            Wa_flat = Wa.view(a_dim, N * N)
            Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, N, N)
            W_eff = J0 + J1 * Wo + Wa_weighted

            grad_r_current = grad_output[t] + grad_r_next

            activation_derivative = non_linear_derivative(recurrent_input, activation_name)
            grad_recurrent_input = activation_derivative * grad_r_current * alpha

            grad_W_eff = torch.bmm(
                grad_recurrent_input.unsqueeze(2),
                r_prev.unsqueeze(1)
            )

            grad_Wo = grad_Wo + J1 * grad_W_eff.sum(dim=0)

            grad_Wa_weighted_flat = grad_W_eff.view(batch_size, N * N)
            grad_Wa_flat = torch.matmul(A_t.t(), grad_Wa_weighted_flat)
            grad_Wa = grad_Wa + grad_Wa_flat.view(a_dim, N, N)

            grad_r_from_recurrent = (W_eff.transpose(1, 2) @ grad_recurrent_input.unsqueeze(2)).squeeze(2)
            grad_r_next = grad_r_from_recurrent + grad_r_current * (1 - alpha)

        return None, grad_Wa, None, None, grad_Wo, None, None, None


class CalcBumpFunction_permute(Function):
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
# PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#     m.def("bwd", &bwd, "bwd (CUDA)",
#           py::arg("grad_output"),
# py::arg("A"),
# py::arg("Wa"),
# py::arg("J0"),
# py::arg("J1"),
# py::arg("Wo"),
# py::arg("bump_history"),
# py::arg("r_init"),
# py::arg("grad_Wa"),
# py::arg("grad_Wo"),
# py::arg("alpha"),
# py::arg("activation_type"));
# }

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

        # grad_r = torch.zeros(batch_size, N, device=grad_output.device, dtype=grad_output.dtype)
        #
        # for t in reversed(range(seq_len)):
        #     if t == 0:
        #         r_prev = r_init
        #     else:
        #         r_prev = bump_history[t - 1]
        #
        #     A_t = action_signal[:, t, :]
        #     Wa_flat = Wa.view(a_dim, N * N)
        #     Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, N, N)
        #     W_eff = J0 + J1 * Wo + Wa_weighted
        #
        #     recurrent_input = (W_eff @ r_prev.unsqueeze(2)).squeeze(2)
        #
        #     grad_r = grad_r + grad_output[t]
        #
        #     activation_derivative = non_linear_derivative(recurrent_input, activation_name)
        #     grad_recurrent_input = activation_derivative * grad_r * alpha
        #
        #     grad_W_eff = torch.bmm(
        #         grad_recurrent_input.unsqueeze(2),
        #         r_prev.unsqueeze(1)
        #     )
        #
        #     grad_Wo = grad_Wo + J1 * grad_W_eff.sum(dim=0)
        #
        #     grad_Wa_weighted_flat = grad_W_eff.view(batch_size, N * N)
        #     grad_Wa_flat = torch.matmul(A_t.t(), grad_Wa_weighted_flat)
        #     grad_Wa = grad_Wa + grad_Wa_flat.view(a_dim, N, N)
        #
        #     grad_r_from_recurrent = (W_eff.transpose(1, 2) @ grad_recurrent_input.unsqueeze(2)).squeeze(2)
        #     grad_r = grad_r_from_recurrent + grad_r * (1 - alpha)

        return None, grad_Wa, None, None, grad_Wo, None, None, None

def non_linear(x, activation_name):
    """Apply activation function."""
    if activation_name == 'tanh':
        return torch.tanh(x)
    elif activation_name == 'relu':
        return torch.relu(x)
    elif activation_name == 'sigmoid':
        return torch.sigmoid(x)
    elif activation_name == 'silu':
        return torch.nn.functional.silu(x)
    elif activation_name == 'gelu':
        return torch.nn.functional.gelu(x)
    elif activation_name == 'linear' or activation_name is None:
        return x
    else:
        raise ValueError(f"Unknown activation: {activation_name}")


def non_linear_derivative(x, activation_name):
    """
    Compute derivative of activation function.

    Args:
        x: pre-activation input
        activation_name: name of activation function

    Returns:
        derivative w.r.t. x
    """
    if activation_name == 'tanh':
        return 1 - torch.tanh(x) ** 2
    elif activation_name == 'relu':
        return (x > 0).float()
    elif activation_name == 'sigmoid':
        sig = torch.sigmoid(x)
        return sig * (1 - sig)
    elif activation_name == 'silu':
        sig = torch.sigmoid(x)
        return sig * (1 + x * (1 - sig))
    elif activation_name == 'gelu':
        phi_x = 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
        pdf_x = torch.exp(-0.5 * x ** 2) / torch.sqrt(2 * torch.pi)
        return phi_x + x * pdf_x
    elif activation_name == 'linear' or activation_name is None:
        return torch.ones_like(x)
    else:
        raise ValueError(f"Unknown activation: {activation_name}")



def calc_bump_impl(action_signal, Wa, J0, J1, Wo, W_delta7, r_init, alpha, activation_name):
    """
    Wrapper function using custom autograd.

    Args:
        action_signal: (batch_size, seq_len, a_dim)
        Wa: (a_dim, N, N) - differentiable
        J0: scalar - not differentiable
        J1: scalar - not differentiable
        Wo: (N, N) - differentiable
        r_init: (batch_size, N) - not differentiable
        alpha: scalar (0 to 1) - not differentiable
        activation_name: str - activation function name

    Returns:
        bump_history: (batch_size, seq_len, N)
    """
    return CalcBumpFunction_permute.apply(
        action_signal, Wa, J0, J1, Wo, r_init, alpha, activation_name
    )

    # bump_history =  CalcBumpFunction_permute.apply(
    #     action_signal, Wa, J0, J1, Wo, r_init, alpha, activation_name
    # )
    #
    # bump_history = bump_history.permute(1, 0, 2)
    # r_delta7 = bump_history @ W_delta7
    # r_max = r_delta7.max(dim=2, keepdim=True)[0]
    # r_history = r_delta7 / r_max
    #
    # return r_history




def param_prep(num_neurons, a_dim, device='cpu'):
    Wa = torch.randn(a_dim, num_neurons, num_neurons, device=device, requires_grad=True)
    J0 = -0.1
    J1 = 0.1
    Wo = torch.randn(num_neurons, num_neurons, device=device, requires_grad=True)

    indices = torch.arange(num_neurons, dtype=torch.float32)
    i = indices.unsqueeze(1)
    j = indices.unsqueeze(0)
    angle_diff = 2 * torch.pi * (i - j) / num_neurons
    W_delta7 = torch.cos(angle_diff).to(device="cuda").requires_grad_(False)


    return Wa, J0, J1, Wo, W_delta7


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

def calc_bump_permute_ref(action_signal, Wa, J0, J1, Wo, W_delta7, r_init, alpha, activation_name):
    """Original calc_bump for reference."""
    batch_size, seq_len, a_dim = action_signal.shape
    N = Wa.shape[1]
    r = r_init.clone()
    bump_history = []
    bump_history = torch.empty(seq_len, batch_size, N, device="cuda")

    for t in range(seq_len):
        A_t = action_signal[:, t, :]
        Wa_flat = Wa.view(a_dim, N * N)
        Wa_weighted = torch.matmul(A_t, Wa_flat).view(batch_size, N, N)

        W_eff = J0 + J1 * Wo + Wa_weighted
        recurrent_input = (W_eff @ r.unsqueeze(2)).squeeze(2)
        recurrent_input = non_linear(recurrent_input, activation_name)
        r = r * (1 - alpha) + recurrent_input * alpha
        bump_history[t, :, :] = r

    return bump_history

    # bump_history = bump_history.permute(1, 0, 2)
    # r_delta7 = bump_history @ W_delta7
    # r_max = r_delta7.max(dim=2, keepdim=True)[0]
    # r_history = r_delta7 / r_max
    #
    # return r_history

if __name__ == "__main__":

    # Small test case for gradcheck
    # num_neur = 8
    # a_dim = 4
    # seq_len = 10
    # bs = 4
    # act = "tanh"
    # alpha = 0.15
    #
    # set_seed(42)
    # Wa, J0, J1, Wo = param_prep(num_neurons=num_neur, a_dim=a_dim, device='cpu')
    # action_signal, r_init, _ = input_prep(num_neurons=num_neur, batch_size=bs, a_dim=a_dim, seq_len=seq_len, device='cpu')
    #
    # # Convert to double for gradcheck
    # Wa = Wa.double().requires_grad_(True)
    # Wo = Wo.double().requires_grad_(True)
    # action_signal = action_signal.double()
    # r_init = r_init.double()
    #
    # def calc_bump_wrapper(wa, wo):
    #     return calc_bump_impl(action_signal, wa, J0, J1, wo, r_init, alpha, act)

    # test = torch.autograd.gradcheck(calc_bump_wrapper, (Wa, Wo))

    # print(f"torch.autograd.gradcheck passed: {test}")

    num_neur = 512
    a_dim = 16
    act = "silu"
    bs = 64
    seq_len = 4
    alpha = 0.15
    device = "cuda"

    # num_neur = 8
    # a_dim = 4
    # seq_len = 10
    # bs = 4
    # act = "tanh"
    # alpha = 0.15
    # device = "cuda"

    print(f"batch_size: {bs} num_neurons: {num_neur}, action dim: {a_dim}, seq_len {seq_len}, activation: {act}:")

    set_seed(42)
    Wa_impl, J0_impl, J1_impl, Wo_impl, W_delta7_impl = param_prep(num_neurons=num_neur, a_dim=a_dim, device=device)
    action_signal_impl, r_init_impl, _ = input_prep(num_neurons=num_neur, batch_size=bs, a_dim=a_dim, seq_len=seq_len, device=device)

    set_seed(42)
    Wa_ref, J0_ref, J1_ref, Wo_ref, W_delta7_ref = param_prep(num_neurons=num_neur, a_dim=a_dim, device=device)
    action_signal_ref, r_init_ref, _ = input_prep(num_neurons=num_neur, batch_size=bs, a_dim=a_dim, seq_len=seq_len, device=device)


    result_impl = calc_bump_impl(action_signal_impl, Wa_impl, J0_impl, J1_impl, Wo_impl, W_delta7_impl, r_init_impl, alpha, act)
    # result_impl = ring_rnn_cuda_func(action_signal_impl, Wa_impl, J0_impl, J1_impl, Wo_impl, r_init_impl, alpha, act).permute(1, 0, 2)
    result_ref = calc_bump_permute_ref(action_signal_ref, Wa_ref, J0_ref, J1_ref, Wo_ref, W_delta7_ref, r_init_ref, alpha, act)


    gt = torch.randn_like(result_impl, device=device)
    gt_impl = gt.clone().detach()
    check_tensor_match(tsr_ref=result_ref, tsr_impl=result_impl, name="Forward", max_print=10)

    criterion = torch.nn.MSELoss()

    loss_ref = criterion(result_ref, gt)
    loss_ref.backward()
    grad_Wa_ref = Wa_ref.grad
    grad_Wo_ref = Wo_ref.grad

    loss_impl = criterion(result_impl, gt_impl)
    loss_impl.backward()
    grad_Wa_impl = Wa_impl.grad
    grad_Wo_impl = Wo_impl.grad

    check_tensor_match(tsr_ref=grad_Wa_ref, tsr_impl=grad_Wa_impl, name="Backward Wa", max_print=10)
    # check_tensor_match(tsr_ref=grad_Wa_ref, tsr_impl=grad_Wa_impl, name="Backward Wa", atol=1e-2, rtol=1e-3, max_print=10)
    print("a idx = 0")
    print("First 10")
    print("ref : ", grad_Wa_ref[0, 0, :10].cpu().numpy())
    print("impl: ", grad_Wa_impl[0, 0, :10].cpu().numpy())
    print("Last 10")
    print("ref : ", grad_Wa_ref[0, 0, -10:].cpu().numpy())
    print("impl: ", grad_Wa_impl[0, 0, -10:].cpu().numpy())
    print("a idx = last")
    print("First 10")
    print("ref : ", grad_Wa_ref[-1, 0, :10].cpu().numpy())
    print("impl: ", grad_Wa_impl[-1, 0, :10].cpu().numpy())
    print("Last 10")
    print("ref : ", grad_Wa_ref[-1, 0, -10:].cpu().numpy())
    print("impl: ", grad_Wa_impl[-1, 0, -10:].cpu().numpy())

    check_tensor_match(tsr_ref=grad_Wo_ref, tsr_impl=grad_Wo_impl, name="Backward Wo", max_print=10)
    # check_tensor_match(tsr_ref=grad_Wo_ref, tsr_impl=grad_Wo_impl, name="Backward Wo", atol=1e-2, rtol=1e-3, max_print=10)
    print("N idx=0")
    print("First 10")
    print("ref : ", grad_Wo_ref[0, :10].cpu().numpy())
    print("impl: ", grad_Wo_impl[0, :10].cpu().numpy())
    print("Last 10")
    print("ref : ", grad_Wo_ref[0, -10:].cpu().numpy())
    print("impl: ", grad_Wo_impl[0, -10:].cpu().numpy())
    print("N idx=last")
    print("First 10")
    print("ref : ", grad_Wo_ref[-1, :10].cpu().numpy())
    print("impl: ", grad_Wo_impl[-1, :10].cpu().numpy())
    print("Last 10")
    print("ref : ", grad_Wo_ref[-1, -10:].cpu().numpy())
    print("impl: ", grad_Wo_impl[-1, -10:].cpu().numpy())

    # # ============ LATENCY BENCHMARKING ============
    # print("\n" + "="*60)
    # print("LATENCY BENCHMARKING")
    # print("="*60)
    #
    # # Forward pass latency
    # print("\n--- Forward Pass ---")
    # latency_ref_fwd = measure_latency_cuda(
    #     calc_bump_ref,
    #     action_signal_ref, Wa_ref, J0_ref, J1_ref,
    #     Wo_ref, r_init_ref, alpha, act,
    #     n_warmup=5, n_iters=50
    # )
    # print(f"Reference:      {latency_ref_fwd}")
    #
    # latency_impl_fwd = measure_latency_cuda(
    #     calc_bump_impl,
    #     action_signal_impl, Wa_impl, J0_impl, J1_impl,
    #     Wo_impl, r_init_impl, alpha, act,
    #     n_warmup=5, n_iters=50
    # )
    # print(f"Implementation: {latency_impl_fwd}")
    #
    # # Forward + Backward pass latency
    # print("\n--- Forward + Backward Pass ---")
    #
    # def forward_backward_ref():
    #     if Wa_ref.grad is not None:
    #         Wa_ref.grad.zero_()
    #     if Wo_ref.grad is not None:
    #         Wo_ref.grad.zero_()
    #
    #     result = calc_bump_ref(action_signal_ref, Wa_ref, J0_ref,
    #                            J1_ref, Wo_ref, r_init_ref, alpha, act)
    #     loss = criterion(result, gt)
    #     loss.backward()
    #
    # def forward_backward_impl():
    #     if Wa_impl.grad is not None:
    #         Wa_impl.grad.zero_()
    #     if Wo_impl.grad is not None:
    #         Wo_impl.grad.zero_()
    #
    #     result = calc_bump_impl(action_signal_impl, Wa_impl, J0_impl,
    #                             J1_impl, Wo_impl, r_init_impl, alpha, act)
    #     loss = criterion(result, gt_impl)
    #     loss.backward()
    #
    # latency_ref_bwd = measure_latency_cuda(forward_backward_ref, n_warmup=5, n_iters=50)
    # print(f"Reference:      {latency_ref_bwd}")
    #
    # latency_impl_bwd = measure_latency_cuda(forward_backward_impl, n_warmup=5, n_iters=50)
    # print(f"Implementation: {latency_impl_bwd}")
