import torch
import torch.nn as nn
from generate_av_integration_data import AVIntegrationDataset
from model_bf16 import GeneralizedRingAttractorNoGain_ref
torch.manual_seed(42)
torch.set_printoptions(linewidth=200)

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

module = load(
    name='torch_sum',
    sources=[f"{dir_path}/cpp/torch_sum_kernel.cu", f"{dir_path}/cpp/torch_sum.cpp"],
    verbose=True,
    build_directory=build_dir 
)

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
    initial_angles = initial_angles.to(device).to(torch.bfloat16)
    bump_centers = initial_angles * num_neurons / (2 * torch.pi)
    bump_centers = bump_centers.unsqueeze(1)
    bump_width = num_neurons / bump_width_factor
    indices = torch.arange(num_neurons, device=device, dtype=torch.bfloat16).unsqueeze(0)
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

def process_ring_attractor_sequence(action_signal, r, J0, J1, Wo, Wa, W_delta7, activation_name, Wa_weighted, recurrent_input, r_delta7):
    """Process entire sequence of ring attractor dynamics.
    
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


    """
    batch_size, seq_len, action_dim = action_signal.shape
    bump_history = []
    r_history = []
    
    A = action_signal  # (batch, seq, action_dim)
    
    for t in range(seq_len):
        # Get action vector at time t
        A_t = A[:, t, :]  # (batch, action_dim)

        # Compute weighted sum of action matrices
        # A_t_expanded = A_t.unsqueeze(-1).unsqueeze(-1)
        # Wa_weighted.copy_(torch.sum(A_t_expanded * Wa.unsqueeze(0), dim=1))

        torch_sum(A_t, Wa, Wa_weighted)

        W_eff = J0 + J1 * Wo + Wa_weighted

        # Recurrent dynamics
        recurrent_input.copy_((W_eff @ r.unsqueeze(2)).squeeze(2))
        recurrent_input = non_linear(recurrent_input, activation_name)

        # Update rule (leaky integration)
        alpha = 0.15
        r = r * (1 - alpha) + recurrent_input * alpha

        bump_history.append(r)

        # Transform to cosine wave for output
        r_delta7.copy_(r @ W_delta7)
        r_max = r_delta7.max(dim=1, keepdim=True)[0]
        r_delta7.div_(r_max)  # normalize

        r_history.append(r_delta7)

    return torch.stack(r_history, dim=1), torch.stack(bump_history, dim=1)

def process_ring_attractor_sequence_cuda(action_signal, r, J0, J1, Wo, Wa, W_delta7, activation_name, Wa_weighted, recurrent_input, r_delta7):
    batch_size, seq_len, action_dim = action_signal.shape
    bump_history = []
    r_history = []
    A = action_signal  # (batch, seq, action_dim)
    for t in range(seq_len):
        # Get action vector at time t
        A_t = A[:, t, :]  # (batch, action_dim)
        # A_t_expanded = A_t.unsqueeze(-1).unsqueeze(-1)
        # Wa_weighted.copy_(torch.sum(A_t_expanded * Wa.unsqueeze(0), dim=1))

        # torch_sum(A_t, Wa, Wa_weighted)
        module.torch_sum(A_t, Wa, Wa_weighted)

        # module.torch_sum(A_t, 
        #                  Wa,
        #                  J0,
        #                  J1,
        #                  r,
        #                  Wo, 
        #                  Wa_weighted)

        W_eff = J0 + J1 * Wo + Wa_weighted

        recurrent_input.copy_((W_eff @ r.unsqueeze(2)).squeeze(2))
        recurrent_input = non_linear(recurrent_input, activation_name)
        alpha = 0.15
        r = r * (1 - alpha) + recurrent_input * alpha
        bump_history.append(r)
        r_delta7.copy_(r @ W_delta7)
        r_max = r_delta7.max(dim=1, keepdim=True)[0]
        r_delta7.div_(r_max)  # normalize
        r_history.append(r_delta7)

    return torch.stack(r_history, dim=1), torch.stack(bump_history, dim=1)


class GeneralizedRingAttractorNoGain(nn.Module):
    """
    Generalized Ring Attractor model with arbitrary action dimensions but WITHOUT gain networks.
    r += (-r + F(J0.dot(r) + J1*Wo.dot(r) + sum_i(A_i * Wa_i.dot(r)))) * dt / tau
    where A is an action vector of dimension k, and Wa is a tensor of shape (k, N, N)

    This version directly uses action signals without any gain modulation.

    Step 1: Take batch 0
    A_t[0,0] = 0.5, A_t[0,1] = 1.0


    Multiply each matrix by the action weight:

    0.5 * Wa[0] =
    [[0.5, 1.0, 1.5],
    [2.0, 2.5, 3.0],
    [3.5, 4.0, 4.5]]

    1.0 * Wa[1] =
    [[10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]]


    Add them:

    Wa_weighted[0] =
    [[10.5, 21.0, 31.5],
    [42.0, 52.5, 63.0],
    [73.5, 84.0, 94.5]]

    Step 2: Take batch 1
    A_t[1,0] = 2.0, A_t[1,1] = 0.1


    Multiply:

    2.0 * Wa[0] =
    [[2, 4, 6],
    [8, 10, 12],
    [14, 16, 18]]

    0.1 * Wa[1] =
    [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]


    Add them:

    Wa_weighted[1] =
    [[3, 6, 9],
    [12, 15, 18],
    [21, 24, 27]]

    ✅ Final Result
    Wa_weighted =
    [
    [[10.5, 21.0, 31.5],
    [42.0, 52.5, 63.0],
    [73.5, 84.0, 94.5]],

    [[3, 6, 9],
    [12, 15, 18],
    [21, 24, 27]]
    ]
    """
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
        
        # Convert indices to bfloat16
        indices = torch.arange(self.num_neurons, dtype=torch.bfloat16)
        i = indices.unsqueeze(1)
        j = indices.unsqueeze(0)
        angle_diff = 2 * torch.pi * (i - j) / self.num_neurons
        self.register_buffer('W_delta7', torch.cos(angle_diff))

        # Fixed parameters with bfloat16
        self.J0 = -0.1 * torch.ones(self.num_neurons, self.num_neurons, 
                                   device=self.device, dtype=torch.bfloat16)
        self.J1 = 0.1

        # Learnable parameters with bfloat16
        self.Wo = nn.Parameter(
            torch.randn(self.num_neurons, self.num_neurons, dtype=torch.bfloat16) / self.num_neurons ** 0.5)
        self.Wa = nn.Parameter(
            torch.randn(self.action_dim, self.num_neurons, self.num_neurons, 
                       dtype=torch.bfloat16) / self.num_neurons ** 0.5)


    def forward(self, action_signal, r_init_bf16=None, ref=True):
        """
        Pre-allocates memory for tensors to reduce GPU memory fragmentation.
        """
        batch_size, seq_len, action_dim = action_signal.shape
        assert action_dim == self.action_dim, f"Expected action_dim {self.action_dim}, got {action_dim}"

        self.J0 = self.J0.to(self.Wo.device)
        self.W_delta7 = self.W_delta7.to(self.Wo.device)

        N = self.num_neurons

        # Initialize r
        if r_init_bf16 is None:
            initial_angle = torch.full((batch_size,), torch.pi, device=self.Wo.device)
            r = create_initial_bump(initial_angle, N, device=self.Wo.device)
        else:
            r = r_init_bf16

        bump_history = []
        r_history = []

        # Pre-allocate memory for history
        # r_history = torch.zeros(batch_size, seq_len, N, device=self.Wo.device, dtype=r.dtype)
        # bump_history = torch.zeros(batch_size, seq_len, N, device=self.Wo.device, dtype=r.dtype)
        

        # Pre-allocate intermediate tensors
        Wa_weighted = torch.zeros(batch_size, N, N, device=self.Wo.device, dtype=self.Wa.dtype)
        recurrent_input = torch.zeros(batch_size, N, device=self.Wo.device, dtype=r.dtype)
        r_delta7 = torch.zeros(batch_size, N, device=self.Wo.device, dtype=r.dtype)

        # bump_history[:, t, :].copy_(r)

        # r_history[:, t, :].copy_(r_delta7)


        return process_ring_attractor_sequence(action_signal, r, self.J0, self.J1, self.Wo, self.Wa, self.W_delta7, self.activation_name, Wa_weighted, recurrent_input, r_delta7)
        # else:
        #     return process_ring_attractor_sequence_cuda(action_signal, r, self.J0, self.J1, self.Wo, self.Wa, self.W_delta7, self.activation_name, Wa_weighted, recurrent_input, r_delta7)            
        

def benchmark(num_neurons=120, seq_len=120, action_dim=2, batch_size=32):
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

    # Model setup with bfloat16
    ring_rnn = GeneralizedRingAttractorNoGain(
        num_neurons=num_neurons,
        action_dim=action_dim,
        tau=10,
        dt=1,
        activation='gelu',
        initialization='random',
        device=device
    )

    ring_rnn.to(device)

    ring_rnn_ref = GeneralizedRingAttractorNoGain_ref(
        num_neurons=num_neurons,
        action_dim=action_dim,
        tau=10,
        dt=1,
        activation='gelu',
        initialization='random',
        device=device
    )

    ring_rnn_ref.to(device)

    print("--------------- Model params ----------------------")

    print("J0:", ring_rnn.J0.shape, ring_rnn.J0.dtype, ring_rnn.J0.device) 
    print("W_delta7:", ring_rnn.W_delta7.shape, ring_rnn.W_delta7.dtype, ring_rnn.W_delta7.device)
    print("Wo:", ring_rnn.Wo.shape, ring_rnn.Wo.dtype, ring_rnn.Wo.device)
    print("Wa:", ring_rnn.Wa.shape, ring_rnn.Wa.dtype, ring_rnn.Wa.device)

    print("--------------- Data params ----------------------")

    av_signal, target_angle = dataset.generate_batch(batch_size)

    av_signal_bf16 = av_signal.to(torch.bfloat16)
    target_angle_bf16 = target_angle.to(torch.bfloat16)

    av_signal_bf16 = av_to_action_signal_ND(av_signal_bf16, action_dim)
    initial_angle_bf16 = target_angle_bf16[:, 0]
    r_init_bf16 = create_initial_bump(initial_angle_bf16, num_neurons, device=device)



    print("av_signal:", av_signal_bf16.shape, av_signal_bf16.dtype, av_signal_bf16.device)
    print("target_angle:", target_angle_bf16.shape, target_angle_bf16.dtype, target_angle_bf16.device)
    print("r_init_bf16:", r_init_bf16.shape, r_init_bf16.dtype, r_init_bf16.device)
    print("initial_angle:", initial_angle_bf16.shape, initial_angle_bf16.dtype, initial_angle_bf16.device)

    print("--------------- Inference ----------------------")

    # Forward pass
    predicted_cosine_wave, bump_activity = ring_rnn(av_signal_bf16, r_init=r_init_bf16)

    # return predicted_cosine_wave, bump_activity

if __name__ == "__main__":

    # --- Training Parameters ---
    num_neurons = 256
    seq_len = 128
    action_dim = 32

    training_steps = 10
    learning_rate = 1e-3
    batch_size = 1

    with torch.no_grad():

        # predicted_cosine_wave, bump_activity = fwd_ref(num_neurons=num_neurons, seq_len=seq_len, action_dim=action_dim, training_steps=training_steps, learning_rate=learning_rate, batch_size=batch_size)

        # print("cosine_wave ref: ")
        # print(predicted_cosine_wave[0][0][:10])

        # print("bump_activity ref: ")
        # print(bump_activity[0][0][:10])

        # predicted_cosine_wave, bump_activity = benchmark(num_neurons=num_neurons, seq_len=seq_len, action_dim=action_dim, batch_size=batch_size)  

        benchmark(num_neurons=num_neurons, seq_len=seq_len, action_dim=action_dim, batch_size=batch_size) 

        # print("cosine_wave mine: ")
        # print(predicted_cosine_wave[0][0][:10])

        # print("bump_activity mine: ")
        # print(bump_activity[0][0][:10])

    # Run benchmarks
    # results = benchmark_fwd(num_trials=10)
    
    # print("\n=== Benchmark Results ===")
    # print(f"Reference Implementation: {results['reference_mean_ms']:.2f} ± {results['reference_std_ms']:.2f} ms")
    # print(f"CUDA Implementation: {results['cuda_mean_ms']:.2f} ± {results['cuda_std_ms']:.2f} ms")
    # print(f"Speedup: {results['speedup']:.2f}x")
    # print(f"Maximum Error: {results['max_difference']:.2e}")
    
    # if results['max_difference'] > 1e-3:
    #     print("\nWARNING: Large difference between implementations detected!")


