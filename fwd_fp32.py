import torch
import torch.nn as nn
from generate_av_integration_data import AVIntegrationDataset
from model_fp32 import GeneralizedRingAttractorNoGain_ref
import random
import numpy as np

torch.set_printoptions(linewidth=200)
np.set_printoptions(linewidth=200, precision=6, suppress=True)
#################################################################################################
############################################## CUDA #############################################
#################################################################################################

from torch.utils.cpp_extension import load
import pathlib
import os

print("========================================================") 

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

module = load(
    name='torch_sum',
    sources=[f"{dir_path}/cpp/torch_no_loop.cu", f"{dir_path}/cpp/torch_no_loop.cpp"],
    verbose=True,
    build_directory=build_dir 
)



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# set_seed(42)

#################################################################################################
#################################################################################################
#################################################################################################



def non_linear(x, activation_name):
    if activation_name == 'tanh':
        return torch.tanh(x)
    elif activation_name == 'relu':
        return torch.relu(x)
    elif activation_name == 'gelu':
        return torch.nn.functional.gelu(x, approximate='tanh')
    else:
        raise ValueError(f"Activation function {activation_name} not supported")

def create_initial_bump(initial_angles, num_neurons, bump_width_factor=10, device='cuda'):
    initial_angles = initial_angles.to(device).to(torch.float32)
    bump_centers = initial_angles * num_neurons / (2 * torch.pi)
    bump_centers = bump_centers.unsqueeze(1)
    bump_width = num_neurons / bump_width_factor
    indices = torch.arange(num_neurons, device=device, dtype=torch.float32).unsqueeze(0)
    dist = torch.min(torch.abs(indices - bump_centers),
                     num_neurons - torch.abs(indices - bump_centers))
    initial_bump = torch.exp(-(dist**2 / (2 * bump_width**2)))
    return initial_bump / initial_bump.norm(dim=1, keepdim=True)

def cosine_similarity_loss(predicted_cosine_wave, true_angle):
    """
    Calculates the loss as the Mean Squared Error between the (x,y) components
    of the predicted angle and the true angle on a unit circle.
    This version follows the logic of explicitly decoding the angle first.
    """
    num_neurons = predicted_cosine_wave.shape[-1]
    preferred_angles = torch.linspace(0, 2 * torch.pi, num_neurons,
                                      device=predicted_cosine_wave.device, requires_grad=False)
    
    # --- Predicted Angle Vector ---
    # 1. Calculate Fourier components of the network's output
    pred_x = torch.sum(predicted_cosine_wave * torch.cos(preferred_angles), dim=-1)
    pred_y = torch.sum(predicted_cosine_wave * torch.sin(preferred_angles), dim=-1)
    
    # 2. Differentiably extract the predicted angle theta using atan2
    predicted_theta = torch.atan2(pred_y, pred_x)

    # 3. Calculate the vector components from the decoded angle
    pred_x_from_theta = torch.cos(predicted_theta)
    pred_y_from_theta = torch.sin(predicted_theta)

    # --- Target Angle Vector ---
    # 4. Calculate the vector components from the true angle
    true_x = torch.cos(true_angle)
    true_y = torch.sin(true_angle)
    
    # 5. Calculate the Mean Squared Error between the vector components
    loss = torch.mean((pred_x_from_theta - true_x)**2 + (pred_y_from_theta - true_y)**2)

    return loss

def bump_amplitude_loss(bump_activity, target_amplitude=None):
    """
    Loss to ensure that the total amplitude/energy of the bump remains constant.
    
    Args:
        bump_activity: Neural activity over time, shape (batch, time, neurons)
        target_amplitude: Target total amplitude. If None, use the first time step as target.
    
    Returns:
        loss: Scalar loss value
    """
    batch_size, seq_len, num_neurons = bump_activity.shape
    
    # Calculate total activity (L1 norm) at each time step
    total_activity = torch.sum(torch.abs(bump_activity), dim=2)  # Shape: (batch, time)
    
    if target_amplitude is None:
        # Use the first time step as the target amplitude
        target_amplitude = total_activity[:, 0:1]  # Shape: (batch, 1)
    
    # Calculate loss as deviation from target amplitude
    amplitude_loss = torch.mean((total_activity - target_amplitude)**2)
    
    return amplitude_loss

def av_to_action_signal(av_signal):
    """
    Convert angular velocity signal to L/R action signals.
    Always returns 2D action vector [L, R].
    """
    batch_size, seq_len = av_signal.shape

    # Classic L/R setup
    L = torch.relu(-av_signal)  # Left rotation (negative velocities)
    R = torch.relu(av_signal)   # Right rotation (positive velocities)
    action_signal = torch.stack([L, R], dim=2)

    return action_signal

def av_to_action_signal_ND(av_signal, action_dim=2):
    """
    Convert angular velocity signal to n-dimensional action signals.
    Returns action vector of shape [batch_size, seq_len, action_dim].
    """
    scales = torch.linspace(-1, 1, action_dim, device=av_signal.device, dtype=av_signal.dtype)
    action_signal = torch.relu(av_signal.unsqueeze(-1) * scales)
    
    return action_signal


def torch_sum(A_t, Wa, Wa_weighted):

    # Batch size = 1
    A_t_expanded = A_t.unsqueeze(-1).unsqueeze(-1)
    Wa_weighted.copy_(torch.sum(A_t_expanded * Wa.unsqueeze(0), dim=1))

def process_ring_attractor_sequence_cuda3(action_signal, r, J0, J1, Wo, Wa, W_delta7, activation_name, Wa_weighted, recurrent_input, r_delta7):
    batch_size, seq_len, action_dim = action_signal.shape
    bump_history = []
    r_history = []
    A = action_signal  # (batch, seq, action_dim)
    
    # Pre-allocate W_eff tensor
    W_eff = torch.zeros_like(Wa_weighted)
    r_history = torch.zeros(torch.Size([1, 128, 256]), device='cuda', dtype=torch.float32)
    bump_history = torch.zeros(torch.Size([1, 128, 256]), device='cuda', dtype=torch.float32)

    for t in range(seq_len):
        # Get action vector at time t
        A_t = A[:, t, :]  # (batch, action_dim)
        # A_t_expanded = A_t.unsqueeze(-1).unsqueeze(-1)
        # Wa_weighted.copy_(torch.sum(A_t_expanded * Wa.unsqueeze(0), dim=1))

        # torch_sum(A_t, Wa, Wa_weighted)
        # module.torch_sum(A_t, Wa, Wa_weighted)

        module.torch_sum(
            A_t=A_t, 
            Wa=Wa,
            J0=J0,
            J1=J1,
            Wo=Wo,
            r=r,
            Wa_weighted=Wa_weighted,
            re_inp=recurrent_input,
            W_eff=W_eff,
            t=t,
            bump_history=bump_history,
            r_delta7=r_delta7,
            r_history=r_history
        )

        # W_eff = J0 + J1 * Wo

        # W_eff2 = W_eff + Wa_weighted



        # recurrent_input.copy_((W_eff @ r.unsqueeze(2)).squeeze(2))
        # recurrent_input = non_linear(recurrent_input, activation_name)

        # alpha = 0.15
        # r = r * (1 - alpha) + recurrent_input * alpha
        # bump_history[0, t, :] = r.squeeze(0)

        r_delta7 = r @ W_delta7
        r_max = r_delta7.max(dim=1, keepdim=True)[0]
        r_delta7 = r_delta7 / r_max

        # r_history.append(r_delta7)
        r_history[0, t, :] = r_delta7.squeeze(0)

    # return torch.stack(r_history, dim=1), torch.stack(bump_history, dim=1)
    return r_history, bump_history


def process_ring_attractor_sequence_cuda4(action_signal, r, J0, J1, Wo, Wa, W_delta7, activation_name, Wa_weighted, recurrent_input, r_delta7):
    """Process entire sequence of ring attractor dynamics.
    
    Fully allocate bump_history and r_history, remove for loop, persistent kernel where each block will calculate fully for each t-th action 

    N = 256
    seq_len = 128
    batch_size = 64
    a_dim = 32

    Unchanged in for-loop:
    - action_signal: torch.Size([64, 128, 32])  # input sequence
    - J0: torch.Size([256, 256])                # baseline connectivity
    - J1: scalar = 0.1                          # scaling factor
    - Wo: torch.Size([256, 256])                # recurrent weight
    - Wa: torch.Size([32, 256, 256])            # action-modulated weight bank
    - W_delta7: torch.Size([256, 256])          # output projection
    - activation_name: str                      # nonlinearity to use

    Changed in for-loop:
    - r: torch.Size([64, 256])                  # neural state (updated each step)
    - Wa_weighted: torch.Size([64, 256, 256])   # effective weighted action matrix
    - recurrent_input: torch.Size([64, 256])    # input from recurrent dynamics
    - r_delta7: torch.Size([64, 256])           # normalized output

    Intermediate:
    - W_eff: torch.Size([64, 128, 256, 256])

    Final output: 
    - r_history: torch.Size([64, 128, 256])
    - bump_history: torch.Size([64, 128, 256])

    /////////////////// Batch size = 1

    Unchanged in for-loop:
    - action_signal: torch.Size([1, 128, 32])  # input sequence
    - J0: torch.Size([256, 256])                # baseline connectivity
    - J1: scalar = 0.1                          # scaling factor
    - Wo: torch.Size([256, 256])                # recurrent weight
    - Wa: torch.Size([32, 256, 256])            # action-modulated weight bank
    - W_delta7: torch.Size([256, 256])          # output projection
    - activation_name: str                      # nonlinearity to use

    Changed in for-loop:
    - r: torch.Size([1, 256])                  # neural state (updated each step)
    - Wa_weighted: torch.Size([1, 256, 256])   # effective weighted action matrix
    - recurrent_input: torch.Size([1, 256])    # input from recurrent dynamics
    - r_delta7: torch.Size([1, 256])           # normalized output

    Final output: 
    - r_history: torch.Size([1, 128, 256])
    - bump_history: torch.Size([1, 128, 256])
    """

    batch_size, seq_len, action_dim = action_signal.shape
    bump_history = []
    r_history = []
    A = action_signal  # (batch, seq, action_dim)
    N, _ = J0.shape
    
    # Pre-allocate W_eff tensor
    W_eff = torch.zeros_like(Wa_weighted) # 

    r_history = torch.zeros(torch.Size([batch_size, seq_len, N]), device='cuda', dtype=torch.float32)
    bump_history = torch.zeros(torch.Size([batch_size, seq_len, N]), device='cuda', dtype=torch.float32)

    # r_history = torch.zeros(torch.Size([1, 128, 256]), device='cuda', dtype=torch.float32)
    # bump_history = torch.zeros(torch.Size([1, 128, 256]), device='cuda', dtype=torch.float32)

    # A_expanded = action_signal.unsqueeze(-1).unsqueeze(-1)  # (batch, seq, action_dim, 1, 1)
    # Wa_all = torch.sum(A_expanded * Wa.unsqueeze(0).unsqueeze(0), dim=2)  # (batch, seq, N, N)

    module.torch_sum(
        A=A, 
        Wa=Wa,
        J0=J0,
        J1=J1,
        Wo=Wo,        
        r=r,
        W_delta7=W_delta7,  
        W_eff=W_eff,
        bump_history=bump_history,
        r_delta7=r_delta7,
        r_history=r_history,
        re_inp=recurrent_input
    )

    return r_history, bump_history

class GeneralizedRingAttractorNoGain(nn.Module):

    def __init__(self, num_neurons, action_dim=2, tau=10.0, dt=1.0, activation='tanh',
                 initialization='random', device='cuda'):
        super().__init__()
        self.num_neurons = num_neurons
        self.action_dim = action_dim
        self.tau = tau
        self.dt = dt
        self.activation_name = activation
        self.initialization = initialization
        self.device = device
        
        # Convert indices to float32
        indices = torch.arange(self.num_neurons, dtype=torch.float32)
        i = indices.unsqueeze(1)
        j = indices.unsqueeze(0)
        angle_diff = 2 * torch.pi * (i - j) / self.num_neurons
        self.register_buffer('W_delta7', torch.cos(angle_diff))

        # Fixed parameters with float32
        self.J0 = -0.1 * torch.ones(self.num_neurons, self.num_neurons, 
                                   device=self.device, dtype=torch.float32)
        self.J1 = 0.1

        # Learnable parameters with float32
        self.Wo = nn.Parameter(
            torch.randn(self.num_neurons, self.num_neurons, dtype=torch.float32) / self.num_neurons ** 0.5)
        self.Wa = nn.Parameter(
            torch.randn(self.action_dim, self.num_neurons, self.num_neurons, 
                       dtype=torch.float32) / self.num_neurons ** 0.5)


    def forward(self, action_signal, r_init=None, ref=True):
        """
        Pre-allocates memory for tensors to reduce GPU memory fragmentation.
        """
        batch_size, seq_len, action_dim = action_signal.shape
        assert action_dim == self.action_dim, f"Expected action_dim {self.action_dim}, got {action_dim}"

        self.J0 = self.J0.to(self.Wo.device)
        self.W_delta7 = self.W_delta7.to(self.Wo.device)

        N = self.num_neurons

        # Initialize r
        if r_init is None:
            initial_angle = torch.full((batch_size,), torch.pi, device=self.Wo.device)
            r = create_initial_bump(initial_angle, N, device=self.Wo.device)
        else:
            r = r_init


        # Pre-allocate intermediate tensors
        Wa_weighted = torch.zeros(batch_size, N, N, device=self.Wo.device, dtype=self.Wa.dtype)
        recurrent_input = torch.zeros(batch_size, N, device=self.Wo.device, dtype=r.dtype)
        r_delta7 = torch.zeros(batch_size, N, device=self.Wo.device, dtype=r.dtype)

        return process_ring_attractor_sequence_cuda4(action_signal, r, self.J0, self.J1, self.Wo, self.Wa, self.W_delta7, self.activation_name, Wa_weighted, recurrent_input, r_delta7)       
        

def benchmark(num_neurons=512, seq_len=128, action_dim=32, batch_size=8, activation='gelu'):
    assert torch.cuda.is_available(), "CUDA GPU not detected. Exiting."
    device = torch.device("cuda")

    print("========================================================")    

    # Create dataset for training data generation
    dataset = AVIntegrationDataset(
        num_samples=training_steps * batch_size,
        seq_len=seq_len,
        zero_padding_start_ratio=0.1,
        zero_ratios_in_rest=[0.2, 0.5, 0.8],
        device=device,
        fast_mode=True
    )

    set_seed(42)
    

    ring_rnn = GeneralizedRingAttractorNoGain(
        num_neurons=num_neurons,
        action_dim=action_dim,
        tau=10,
        dt=1,
        activation=activation,
        initialization='random',
        device=device
    )

    ring_rnn.to(device)
    ring_rnn.eval()
    for param in ring_rnn.parameters():
        param.requires_grad = False

    set_seed(42)
    

    ring_rnn_ref = GeneralizedRingAttractorNoGain_ref(
        num_neurons=num_neurons,
        action_dim=action_dim,
        tau=10,
        dt=1,
        activation=activation,
        initialization='random',
        device=device
    )

    ring_rnn_ref.to(device)
    ring_rnn_ref.eval()
    for param in ring_rnn_ref.parameters():
        param.requires_grad = False

    # print("--------------- Model params ----------------------")

    # print("J0:", ring_rnn.J0.shape, ring_rnn.J0.dtype, ring_rnn.J0.device) 
    # print("W_delta7:", ring_rnn.W_delta7.shape, ring_rnn.W_delta7.dtype, ring_rnn.W_delta7.device)
    # print("Wo:", ring_rnn.Wo.shape, ring_rnn.Wo.dtype, ring_rnn.Wo.device)
    # print("Wa:", ring_rnn.Wa.shape, ring_rnn.Wa.dtype, ring_rnn.Wa.device)

    # print("--------------- Data params ----------------------")

    av_signal, target_angle = dataset.generate_batch(batch_size)

    av_signal_fp32 = av_signal.to(torch.float32)
    target_angle_fp32 = target_angle.to(torch.float32)

    av_signal_fp32 = av_to_action_signal_ND(av_signal_fp32, action_dim)
    initial_angle_fp32 = target_angle_fp32[:, 0]
    r_init_fp32 = create_initial_bump(initial_angle_fp32, num_neurons, device=device)



    # print("av_signal:", av_signal_fp32.shape, av_signal_fp32.dtype, av_signal_fp32.device)
    # print("target_angle:", target_angle_fp32.shape, target_angle_fp32.dtype, target_angle_fp32.device)
    # print("r_init_fp32:", r_init_fp32.shape, r_init_fp32.dtype, r_init_fp32.device)
    # print("initial_angle:", initial_angle_fp32.shape, initial_angle_fp32.dtype, initial_angle_fp32.device)

    

    # Forward pass
    r_init_impl = r_init_fp32.detach().clone()
    r_init_ref = r_init_fp32.detach().clone()


    predicted_cosine_wave, bump_activity = ring_rnn(av_signal_fp32, r_init=r_init_impl)
    predicted_cosine_wave_ref, bump_activity_ref = ring_rnn_ref(av_signal_fp32, r_init=r_init_ref)
    
    print("--------------- Check correctness ----------------------")
    # def check_tensor_match(tsr_impl, tsr_ref, name, rtol=0.01, atol=0.0001, max_print=10):
    
    def check_tensor_match(tsr_impl, tsr_ref, name, rtol=1e-5, atol=1e-8, max_print=20):
        if not torch.allclose(tsr_impl, tsr_ref, rtol=rtol, atol=atol):
            print(f"\n{name} differences: a_tol = {atol}, r_tol = {rtol}")
            diff = (tsr_impl - tsr_ref).abs()
            rdiff = diff / (torch.abs(tsr_ref) + atol)
            mismatch = ~torch.isclose(tsr_impl, tsr_ref, rtol=rtol, atol=atol)
            num_mismatched = mismatch.sum().item()
            total_elements = tsr_impl.numel()
            percentage = (num_mismatched / total_elements) * 100
            all_indices = torch.nonzero(mismatch)
            first_indices = all_indices[:max_print]
            last_indices = all_indices[-max_print:] if len(all_indices) > max_print else torch.empty(0, 3, dtype=torch.long)
            
            print("Mismatch at                 ref         impl        diff        rdiff")
            print("---------------------------------------------------------------------")
            print("First mismatches:")
            for idx in first_indices:
                b, t, n = idx
                print(f"[batch={b:2d},t={t:3d},n={n:3d}]: {tsr_ref[b,t,n]:10.6f} {tsr_impl[b,t,n]:10.6f} {diff[b,t,n]:10.6f} {rdiff[b,t,n]:10.6f}")
            
            if len(all_indices) > max_print:
                print("...")
                print("Last mismatches:")
                for idx in last_indices:
                    b, t, n = idx
                    print(f"[batch={b:2d},t={t:3d},n={n:3d}]: {tsr_ref[b,t,n]:10.6f} {tsr_impl[b,t,n]:10.6f} {diff[b,t,n]:10.6f} {rdiff[b,t,n]:10.6f}")
            
            print(f"Total mismatched elements: {num_mismatched} out of {total_elements} ({percentage:.1f}%)")
            print(f"diff: {diff.mean():.6f} ± {diff.std():.6f}")
            print(f"rdiff: {rdiff.mean():.6f} ± {rdiff.std():.6f}")
            return False
        else:
            print(f"{name} match!")
            return True

    # Check both tensors
    # check_tensor_match(tsr_impl=predicted_cosine_wave, tsr_ref=predicted_cosine_wave_ref, name="Cosine waves", max_print=20)
    
    check_tensor_match(tsr_impl=bump_activity, tsr_ref=bump_activity_ref, name="bump_history")

    check_tensor_match(predicted_cosine_wave, predicted_cosine_wave_ref, "r_history", rtol=1e-4, atol=1e-6)
    # check_tensor_match(predicted_cosine_wave, predicted_cosine_wave_ref, "r_history")

    # print("---------------------------------------------------------------------")

    print("bump_history: ")
    print("ref : ", bump_activity_ref[0, 0, :10].cpu().numpy())
    print("impl: ", bump_activity[0, 0, :10].cpu().numpy())

    print("r_history: ")
    print("ref : ", predicted_cosine_wave_ref[0, 0, :10].cpu().numpy())
    print("impl: ", predicted_cosine_wave[0, 0, :10].cpu().numpy())

    def measure_latency_cuda(fn, *args, n_warmup=2, n_iters=20, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(n_warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_iters):
            start_event.record()
            fn(*args, **kwargs)
            end_event.record()

            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))  # ms

        times = torch.tensor(times, device="cpu")
        mean = times.mean().item()
        std = times.std(unbiased=False).item()
        return f"{mean:.3f} ± {std:.3f} ms"
    
    print("---------------------------------------------------------------------")

    # with torch.no_grad():

    #     lat_ring_rnn = measure_latency_cuda(ring_rnn, av_signal_fp32, r_init=r_init_impl)
    #     lat_ring_rnn_ref = measure_latency_cuda(ring_rnn_ref, av_signal_fp32, r_init=r_init_ref)

    #     print("ring_rnn latency:", lat_ring_rnn)
    #     print("ring_rnn_ref latency:", lat_ring_rnn_ref)


if __name__ == "__main__":

    # --- Training Parameters ---
    num_neurons = 512
    seq_len = 128
    action_dim = 32
    activation = 'relu'

    training_steps = 10
    learning_rate = 1e-3
    batch_size = 1

    seq_len_list = [4, 8, 16, 32, 128, 256, 512, 1024, 2048]
    # seq_len_list = [4, 8, 16, 32, 128, 256]
    batch_size_list = [32, 128, 256, 512, 1024, 2048]
    action_dim_list = [2, 4, 8, 32, 128, 256, 512, 1024]

    # for seq_len in seq_len_list:
    # for batch_size in batch_size_list:
    # for action_dim in action_dim_list:
    #     print(f"batch_size: {batch_size} num_neurons: {num_neurons}, action dim: {action_dim}, seq_len {seq_len}: ")
    #     benchmark(num_neurons=num_neurons, seq_len=seq_len, action_dim=action_dim, batch_size=batch_size, activation=activation) 

    benchmark(num_neurons=num_neurons, seq_len=seq_len, action_dim=action_dim, batch_size=batch_size, activation=activation) 


