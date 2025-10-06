import torch
import torch.nn as nn
from generate_av_integration_data import AVIntegrationDataset
from model_fp32 import GeneralizedRingAttractorNoGain_ref
import random
import numpy as np

from rnn_cuda import *

torch.set_printoptions(linewidth=200)
np.set_printoptions(linewidth=200, precision=6, suppress=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

class GeneralizedRingAttractorNoGain(nn.Module):

    def __init__(self, num_neurons, action_dim, batch_size, seq_len, 
                 tau=10.0, dt=1.0, activation='relu',
                 initialization='random', device='cuda'):
        super().__init__()
        self.num_neurons = num_neurons
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.tau = tau
        self.dt = dt
        self.activation_name = activation
        self.initialization = initialization
        self.device = device
        
        # Convert indices to float32
        indices = torch.arange(num_neurons, dtype=torch.float32)
        i = indices.unsqueeze(1)
        j = indices.unsqueeze(0)
        angle_diff = 2 * torch.pi * (i - j) / num_neurons
        self.register_buffer('W_delta7', torch.cos(angle_diff))
        self.W_delta7 = self.W_delta7.to(self.device)

        # Fixed parameters with float32
        # self.J0 = -0.1 * torch.ones(self.num_neurons, self.num_neurons, device=self.device, dtype=torch.float32)

        self.J0 = -0.1
        self.J1 = 0.1

        # Learnable parameters with float32
        self.Wo = nn.Parameter(
            torch.randn(num_neurons, num_neurons, dtype=torch.float32) / num_neurons ** 0.5)
        self.Wa = nn.Parameter(
            torch.randn(action_dim, num_neurons, num_neurons, 
                       dtype=torch.float32) / num_neurons ** 0.5)

        # Pre-allocate intermediate tensors
        self.r_history = torch.empty(batch_size, seq_len, num_neurons, device='cuda', dtype=torch.float32)
        self.r_history.fill_(0)

        self.bump_history = torch.empty(batch_size, seq_len, num_neurons, device='cuda', dtype=torch.float32)
        self.bump_history.fill_(0)     
        
        torch.cuda.synchronize()   


    def forward(self, action_signal, r_init=None, ref=True):
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

        # Initialize r
        if r_init is None:
            initial_angle = torch.full((self.batch_size,), torch.pi, device=self.Wo.device)
            r = create_initial_bump(initial_angle, self.num_neurons, device=self.Wo.device)
        else:
            r = r_init


        alpha = 0.15

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
        
        return self.r_history.clone(), self.bump_history.clone()
    
def benchmark(num_neurons, seq_len, action_dim, batch_size, activation, check):
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
        batch_size=batch_size,
        seq_len=seq_len,
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
    
    def check_tensor_match(tsr_impl, tsr_ref, name, rtol=1e-5, atol=1e-8, max_print=1):
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
    
    if check:

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

    lat_ring_rnn = measure_latency_cuda(ring_rnn, av_signal_fp32, r_init=r_init_impl)
    lat_ring_rnn_ref = measure_latency_cuda(ring_rnn_ref, av_signal_fp32, r_init=r_init_ref)

    print("ring_rnn latency:", lat_ring_rnn)
    print("ring_rnn_ref latency:", lat_ring_rnn_ref)


if __name__ == "__main__":

    # --- Training Parameters ---
    num_neurons = 128
    seq_len = 20
    action_dim = 3
    activation = 'relu'

    training_steps = 10
    learning_rate = 1e-3
    batch_size = 32

    seq_len_list = [4, 8, 16, 32, 128, 256, 512, 1024, 2048]
    # seq_len_list = [4, 8, 16, 32, 128, 256]
    batch_size_list = [32, 128, 256, 512, 1024, 2048]
    action_dim_list = [2, 3, 4, 8, 32, 128, 256, 512, 1024]
    num_neurons_list = [128, 128*2, 128*3, 128*4, 128*5, 128*6, 128*7, 128*8]

    # assert num_neurons == 128 or num_neurons == 256, f"num_neurons must be 128 or 256, got {num_neurons}"
    # assert num_neurons == 128 and action_dim < 4 or num_neurons == 256, f"Invalid configuration: num_neurons={num_neurons}, action_dim={action_dim}"
    # assert seq_len >= 1, f"seq_len must be >= 1, got {seq_len}"

    # check_correctness = True

    # for num_neurons in num_neurons_list:
    # # for batch_size in batch_size_list:
    # # for action_dim in action_dim_list:
    #     print(f"batch_size: {batch_size} num_neurons: {num_neurons}, action dim: {action_dim}, seq_len {seq_len}: ")
    #     benchmark(num_neurons=num_neurons, seq_len=seq_len, action_dim=action_dim, batch_size=batch_size, activation=activation, check=check_correctness) 


    check_correctness = True
    print(f"batch_size: {batch_size} num_neurons: {num_neurons}, action dim: {action_dim}, seq_len {seq_len}: ")
    benchmark(num_neurons=num_neurons, seq_len=seq_len, action_dim=action_dim, batch_size=batch_size, activation=activation, check=check_correctness) 


