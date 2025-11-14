import torch
import torch.nn as nn
from utils.generate_av_integration_data import AVIntegrationDataset
from model_fp32 import GeneralizedRingAttractorNoGain_ref
import numpy as np

# from rnn_cuda import *
from ring_rnn_cuda import ring_rnn_cuda_func
from utils.benchmark import *

#################################################################################################
#################################################################################################
#################################################################################################


def non_linear(x, activation_name):
    if activation_name == 'tanh':
        return torch.tanh(x)
    elif activation_name == 'relu':
        return torch.relu(x)
    elif activation_name == 'gelu':
        return torch.nn.functional.gelu(x)
        # return torch.nn.functional.gelu(x, approximate='tanh')
    elif activation_name == 'silu':
        return torch.nn.functional.silu(x)
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
        self.seq_len = seq_len
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

    def forward(self, action_signal, r_init=None, ref=True):
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

        Intermediate:
        - W_eff: torch.Size([64, 128, 256, 256])

        Final output: 
        - r_history: torch.Size([64, 128, 256])
        - bump_history: torch.Size([128, 64, 256])
        """
        batch_size = action_signal.shape[0]

        if r_init is None:
            initial_angle = torch.full((batch_size,), torch.pi, device=self.device)
            r = create_initial_bump(initial_angle, self.num_neurons, device=self.device)
        else:
            r = r_init

        alpha = 0.15

        bump_history = ring_rnn_cuda_func(
            action_signal=action_signal,
            Wa=self.Wa,
            J0=self.J0,
            J1=self.J1,
            Wo=self.Wo,
            r_init=r,
            alpha=alpha,
            activation=self.activation_name,
        )

        # Compute r_history directly from bump_history
        bump_history = bump_history.permute(1, 0, 2)
        r_delta7 = bump_history @ self.W_delta7
        r_max = r_delta7.max(dim=2, keepdim=True)[0]
        r_history = r_delta7 / r_max     
        
        return r_history.clone(), bump_history.clone()

def benchmark(num_neurons, seq_len, action_dim, batch_size, activation, check_forward, check_backward, measure_latency):
    assert torch.cuda.is_available(), "CUDA GPU not detected. Exiting."
    device = "cuda"

    print("========================================================")

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
    ring_rnn.train()  # Changed from eval()
    # Remove the requires_grad = False loop

    set_seed(42)

    ring_rnn_ref = GeneralizedRingAttractorNoGain_ref(
        num_neurons=num_neurons,
        action_dim=action_dim,
        tau=10,
        dt=1,
        activation=activation,
        initialization='random',
        device=device,
        use_matmul=True
    )

    ring_rnn_ref.to(device)
    ring_rnn_ref.train()  # Changed from eval()
    # Remove the requires_grad = False loop

    av_signal, target_angle = dataset.generate_batch(batch_size)

    av_signal_fp32 = av_signal.to(torch.float32)
    target_angle_fp32 = target_angle.to(torch.float32)

    av_signal_fp32 = av_to_action_signal_ND(av_signal_fp32, action_dim)
    initial_angle_fp32 = target_angle_fp32[:, 0]
    r_init_fp32 = create_initial_bump(initial_angle_fp32, num_neurons, device=device)

    r_init_impl = r_init_fp32.detach().clone()
    r_init_ref = r_init_fp32.detach().clone()




    if check_forward:
        with torch.no_grad():
            predicted_cosine_wave, bump_activity = ring_rnn(av_signal_fp32, r_init=r_init_impl)
            predicted_cosine_wave_ref, bump_activity_ref = ring_rnn_ref(av_signal_fp32, r_init=r_init_ref)

            print("--------------- Check correctness Forward ----------------------")
            check_tensor_match(tsr_impl=bump_activity, tsr_ref=bump_activity_ref, name="bump_history", rtol=1e-7, atol=1e-5, max_print=2)
            check_tensor_match(predicted_cosine_wave, predicted_cosine_wave_ref, "r_history", rtol=1e-4, atol=1e-5, max_print=2)
            print("bump_history: ")
            print("ref : ", bump_activity_ref[0, 0, :10].cpu().numpy())
            print("impl: ", bump_activity[0, 0, :10].cpu().numpy())
            print("r_history: ")
            print("ref : ", predicted_cosine_wave_ref[0, 0, :10].cpu().numpy())
            print("impl: ", predicted_cosine_wave[0, 0, :10].cpu().numpy())

            if measure_latency:
                print("---------------------------------------------------------------------")
                lat_ring_rnn = measure_latency_cuda(ring_rnn, av_signal_fp32, r_init=r_init_impl)
                lat_ring_rnn_ref = measure_latency_cuda(ring_rnn_ref, av_signal_fp32, r_init=r_init_ref)
                print("ring_rnn latency:", lat_ring_rnn)
                print("ring_rnn_ref latency:", lat_ring_rnn_ref)



    if check_backward:
        predicted_cosine_wave, bump_activity = ring_rnn(av_signal_fp32, r_init=r_init_impl)
        predicted_cosine_wave_ref, bump_activity_ref = ring_rnn_ref(av_signal_fp32, r_init=r_init_ref)

        loss = cosine_similarity_loss(predicted_cosine_wave, target_angle) + 0.2 * bump_amplitude_loss(bump_activity)
        loss_ref = cosine_similarity_loss(predicted_cosine_wave_ref, target_angle) + 0.2 * bump_amplitude_loss(bump_activity_ref)

        loss.backward()
        loss_ref.backward()

        print("--------------- Check correctness Backward ----------------------")
        for (name_impl, param_impl), (name_ref, param_ref) in zip(ring_rnn.named_parameters(), ring_rnn_ref.named_parameters()):
            if param_impl.grad is not None and param_ref.grad is not None:
                check_tensor_match(param_impl.grad, param_ref.grad, f"grad_{name_impl}", rtol=1e-4, atol=1e-6)




if __name__ == "__main__":

    # --- Training Parameters ---
    
    # Base parameters
    num_neurons = 512
    seq_len = 20
    action_dim = 32
    # relu, gelu, tanh, silu
    activation = 'relu'
    batch_size = 256
    training_steps = 10
    learning_rate = 1e-3

    print("BASE PARAMETERS: ")
    check_correctness_forward = True
    check_correctness_backward = False
    measure_latency = True

    print(f"batch_size: {batch_size} num_neurons: {num_neurons}, action dim: {action_dim}, seq_len {seq_len}, activation: {activation}:")
    benchmark(num_neurons=num_neurons, seq_len=seq_len, action_dim=action_dim, batch_size=batch_size, activation=activation,
              check_forward=check_correctness_forward, check_backward=check_correctness_backward, measure_latency=measure_latency)

    # =========================================================
    # Uncomment below for comprehensive debugging on local GPU
    # =========================================================

    # seq_len_list = [4, 8, 16, 32, 128, 256, 512, 1024, 2048]
    # # seq_len_list = [4, 8, 16, 32, 128, 256]
    # batch_size_list = [32, 128, 256, 512, 1024, 2048]
    # action_dim_list = [2, 3, 6, 12]
    # num_neurons_list = [128, 128*2, 128*3, 128*4, 128*8, 128*16]

    # check_correctness = True

    # for num_neurons in num_neurons_list:
    #     print(f"batch_size: {batch_size} num_neurons: {num_neurons}, action dim: {action_dim}, seq_len {seq_len}: ")
    #     benchmark(num_neurons=num_neurons, seq_len=seq_len, action_dim=action_dim, batch_size=batch_size, activation=activation, check_forward=check_correctness_forward, check_backward=check_correctness_backward, measure_latency=measure_latency)

    # for batch_size in batch_size_list:
    #     print(f"batch_size: {batch_size} num_neurons: {num_neurons}, action dim: {action_dim}, seq_len {seq_len}: ")
    #     benchmark(num_neurons=num_neurons, seq_len=seq_len, action_dim=action_dim, batch_size=batch_size, activation=activation, check=check_correctness)

    # for action_dim in action_dim_list:
    #     print(f"batch_size: {batch_size} num_neurons: {num_neurons}, action dim: {action_dim}, seq_len {seq_len}: ")
    #     benchmark(num_neurons=num_neurons, seq_len=seq_len, action_dim=action_dim, batch_size=batch_size, activation=activation, check_forward=check_correctness_forward, check_backward=check_correctness_backward, measure_latency=measure_latency)       



