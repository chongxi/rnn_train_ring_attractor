import torch
import torch.nn as nn

from generate_av_integration_data import AVIntegrationDataset

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed()

torch.set_printoptions(linewidth=200)

def non_linear(x, activation_name):
    if activation_name == 'tanh':
        return torch.tanh(x)
    elif activation_name == 'relu':
        return torch.relu(x)
    elif activation_name == 'gelu':
        # return torch.nn.functional.gelu(x)
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

class GeneralizedRingAttractorNoGain_ref(nn.Module):
    """
    Generalized Ring Attractor model with arbitrary action dimensions but WITHOUT gain networks.
    r += (-r + F(J0.dot(r) + J1*Wo.dot(r) + sum_i(A_i * Wa_i.dot(r)))) * dt / tau
    where A is an action vector of dimension k, and Wa is a tensor of shape (k, N, N)

    This version directly uses action signals without any gain modulation.
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
        indices = torch.arange(self.num_neurons, dtype=torch.float32)
        i = indices.unsqueeze(1)
        j = indices.unsqueeze(0)
        angle_diff = 2 * torch.pi * (i - j) / self.num_neurons
        self.register_buffer('W_delta7', torch.cos(angle_diff))

        # Fixed parameters with bfloat16
        self.J0 = -0.1 * torch.ones(self.num_neurons, self.num_neurons, 
                                   device=self.device, dtype=torch.float32)
        self.J1 = 0.1

        # Learnable parameters with bfloat16
        self.Wo = nn.Parameter(
            torch.randn(self.num_neurons, self.num_neurons, dtype=torch.float32) / self.num_neurons ** 0.5)
        self.Wa = nn.Parameter(
            torch.randn(self.action_dim, self.num_neurons, self.num_neurons, 
                       dtype=torch.float32) / self.num_neurons ** 0.5)

    # @torch.compile
    def forward(self, action_signal, r_init=None):
        """
        Args:
            action_signal: (batch_size, seq_len, action_dim) - generalized action signals
            r_init: Initial neural state
        """
        batch_size, seq_len, action_dim = action_signal.shape
        assert action_dim == self.action_dim, f"Expected action_dim {self.action_dim}, got {action_dim}"

        self.J0 = self.J0.to(self.Wo.device)
        self.W_delta7 = self.W_delta7.to(self.Wo.device)

        

        if r_init is None:
            initial_angle = torch.full((batch_size,), torch.pi, device=self.Wo.device)
            r = create_initial_bump(initial_angle, self.num_neurons, device=self.Wo.device)
        else:
            r = r_init

        r_history = []
        bump_history = []

        # NO GAIN COMPUTATION - directly use action signals
        A = action_signal  # (batch, seq, action_dim)

        for t in range(seq_len):
            # Get action vector at time t
            A_t = A[:, t, :]  # (batch, action_dim)

            # Compute weighted sum of action matrices
            # Wa has shape (action_dim, num_neurons, num_neurons)
            # A_t has shape (batch, action_dim)
            # We need to compute sum_k(A_t[k] * Wa[k]) for each batch

            # Reshape A_t to (batch, action_dim, 1, 1) for broadcasting
            A_t_expanded = A_t.unsqueeze(-1).unsqueeze(-1)

            # Compute weighted sum: (batch, action_dim, 1, 1) * (action_dim, N, N) -> (batch, N, N)
            Wa_weighted = torch.sum(A_t_expanded * self.Wa.unsqueeze(0), dim=1)

            # Effective weight matrix
            W_eff = self.J0 + self.J1 * self.Wo + Wa_weighted

            # Recurrent dynamics
            recurrent_input = (W_eff @ r.unsqueeze(2)).squeeze(2)
            recurrent_input = non_linear(recurrent_input, self.activation_name)

            # Update rule (leaky integration)
            alpha = 0.15
            r = r * (1 - alpha) + recurrent_input * alpha

            bump_history.append(r)

            # Transform to cosine wave for output
            r_delta7 = r @ self.W_delta7
            r_max = r_delta7.max(dim=1, keepdim=True)[0]
            r_delta7 = r_delta7 / r_max

            r_history.append(r_delta7)

        return torch.stack(r_history, dim=1), torch.stack(bump_history, dim=1)