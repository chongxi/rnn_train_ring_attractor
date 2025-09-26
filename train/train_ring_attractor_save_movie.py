import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import subprocess

# Import the cleaned data generation module
from generate_av_integration_data import AVIntegrationDataset


def non_linear(x, activation_name):
    if activation_name == 'tanh':
        return torch.tanh(x)
    elif activation_name == 'relu':
        return torch.relu(x)
    elif activation_name == 'gelu':
        return torch.nn.functional.gelu(x)
    else:
        raise ValueError(f"Activation function {activation_name} not supported")


def create_initial_bump(initial_angles, num_neurons, bump_width_factor=10, device='cpu'):
    """
    Creates an initial bump of activity centered around the given initial angles.
    Handles circular distance correctly for the ring topology.

    Args:
        initial_angles (torch.Tensor): A tensor of initial angles for each trial in the batch. Shape: (batch_size,).
        num_neurons (int): The total number of neurons in the ring.
        bump_width_factor (int): Factor to determine the width of the bump (num_neurons / factor).
        device (torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: The initial neural activity `r` with shape (batch_size, num_neurons).
    """
    # Ensure initial_angles is on the correct device
    initial_angles = initial_angles.to(device)
    
    # Map angles (0 to 2*pi) to neuron indices (0 to num_neurons)
    bump_centers = initial_angles * num_neurons / (2 * np.pi)
    bump_centers = bump_centers.unsqueeze(1) # Shape: (batch_size, 1)

    bump_width = num_neurons / bump_width_factor
    indices = torch.arange(num_neurons, device=device).unsqueeze(0) # Shape: (1, num_neurons)

    # Calculate distance on the ring (circular distance)
    dist = torch.min(torch.abs(indices - bump_centers),
                     num_neurons - torch.abs(indices - bump_centers))

    initial_bump = torch.exp(-(dist**2 / (2 * bump_width**2)))

    # Normalize each bump to have a norm of 1, consistent with the update rule
    return initial_bump / initial_bump.norm(dim=1, keepdim=True)


class LeakyRingAttractor(nn.Module):
    """
    Leaky Ring Attractor model based on the equation:
    r += (-r + F(J0.dot(r) + J1*Wo.dot(r) + L*Wl.dot(r) + R*Wr.dot(r))) * dt / tau
    """
    def __init__(self, num_neurons, tau=10.0, dt=1.0, activation='tanh', initialization='random', hidden_gain_neurons=16):
        super().__init__()
        self.num_neurons = num_neurons
        self.tau = tau
        self.dt = dt
        self.activation_name = activation
        self.initialization = initialization

        # A small MLP to compute a dynamic gain from av_signal magnitude.
        # The final sigmoid layer constrains the gain to be between 0 and 1.
        def create_gain_net():
            return nn.Sequential(
                nn.Linear(1, hidden_gain_neurons),
                nn.GELU(),
                nn.Linear(hidden_gain_neurons, 1),
                nn.Softplus()
            )

        self.gain_r_net = create_gain_net()
        self.gain_l_net = create_gain_net()

        # Define W_delta7 as a NON-LEARNABLE buffer. This is a fixed, perfect filter.
        indices = torch.arange(num_neurons, dtype=torch.float32)
        i = indices.unsqueeze(1)
        j = indices.unsqueeze(0)
        angle_diff = 2 * np.pi * (i - j) / num_neurons
        self.register_buffer('W_delta7', torch.cos(angle_diff))

        # Fixed parameters from the equation
        self.J0 = -0.1 * torch.ones(num_neurons, num_neurons)  # Weaker global inhibition
        self.J1 = 0.1  # for bump maintenance matrix

        # Learnable parameters
        if self.initialization == 'random':
            self.Wo = nn.Parameter(torch.randn(num_neurons, num_neurons) / np.sqrt(num_neurons))
            self.Wl = nn.Parameter(torch.randn(num_neurons, num_neurons) / np.sqrt(num_neurons))
            self.Wr = nn.Parameter(torch.randn(num_neurons, num_neurons) / np.sqrt(num_neurons))
        elif self.initialization == 'perfect':
            indices = torch.arange(num_neurons, dtype=torch.float32)
            i = indices.unsqueeze(1)
            j = indices.unsqueeze(0)
            angle_diff = 2 * np.pi * (i - j) / num_neurons
            wo_perfect = torch.cos(angle_diff)
            wl_perfect = torch.cos(angle_diff + np.pi / 4)
            wr_perfect = torch.cos(angle_diff - np.pi / 4)
            self.Wo = nn.Parameter(wo_perfect)
            self.Wl = nn.Parameter(wl_perfect)
            self.Wr = nn.Parameter(wr_perfect)
        elif self.initialization == 'debug_perfect':
            # This mode uses perfect weights AND bypasses the gain networks
            # for a true sanity check of the dynamics and loss function.
            self.gain_r_net = None # Disable gain network
            self.gain_l_net = None # Disable gain network
            indices = torch.arange(num_neurons, dtype=torch.float32)
            i = indices.unsqueeze(1)
            j = indices.unsqueeze(0)
            angle_diff = 2 * np.pi * (i - j) / num_neurons
            wo_perfect = torch.cos(angle_diff)
            wl_perfect = torch.cos(angle_diff + np.pi / 4)
            wr_perfect = torch.cos(angle_diff - np.pi / 4)
            self.Wo = nn.Parameter(wo_perfect, requires_grad=False)
            self.Wl = nn.Parameter(wl_perfect, requires_grad=False)
            self.Wr = nn.Parameter(wr_perfect, requires_grad=False)
        else:
            raise ValueError(f"Unknown initialization type: {self.initialization}")

    def forward(self, av_signal, r_init=None):
        batch_size, seq_len = av_signal.shape
        self.J0 = self.J0.to(self.Wo.device)
        self.W_delta7 = self.W_delta7.to(self.Wo.device)

        if self.initialization == 'debug_perfect':
            # For debugging, use a fixed gain of 1.0 and bypass the networks
            gain_r = 1.0
            gain_l = 1.0
        else:
            # Reshape for the gain networks (add a feature dimension)
            R_magnitude = torch.relu(av_signal).unsqueeze(2)
            L_magnitude = torch.relu(-av_signal).unsqueeze(2)
            gain_r = self.gain_r_net(R_magnitude)
            gain_l = self.gain_l_net(L_magnitude)

        # The effective rotation signals are the base signals modulated by the dynamic gains
        R_base = torch.relu(av_signal)
        L_base = torch.relu(-av_signal)
        R = gain_r * R_base.unsqueeze(2)
        L = gain_l * L_base.unsqueeze(2)

        if r_init is None:
            # If no initial state is provided, create a default one centered at pi.
            # This is useful for inference or scenarios without a known start angle.
            initial_angle = torch.full((batch_size,), np.pi, device=self.Wo.device)
            r = create_initial_bump(initial_angle, self.num_neurons, device=self.Wo.device)
        else:
            r = r_init

        r_history = []
        bump_history = []
        for t in range(seq_len):
            # Correctly broadcast L and R to scale the weight matrices
            L_t = L[:, t, :].unsqueeze(1) # Shape: (batch_size, 1, 1)
            R_t = R[:, t, :].unsqueeze(1) # Shape: (batch_size, 1, 1)

            # W_eff is now a batch of matrices, shape: (batch_size, num_neurons, num_neurons)
            W_eff = self.J0 + self.J1 * self.Wo + \
                    L_t * self.Wl + \
                    R_t * self.Wr
            
            # The input to the activation function
            # r has shape (batch, neurons), W_eff has shape (batch, neurons, neurons)
            recurrent_input = (W_eff @ r.unsqueeze(2)).squeeze(2)
            recurrent_input = non_linear(recurrent_input, self.activation_name)

            # Update rule
            alpha = 0.15
            r = r * (1 - alpha) + recurrent_input * alpha

            bump_history.append(r)
            
            # Now, perform the transform to get the cosine wave for the output
            # The hardcoded division is removed as the new loss is amplitude-invariant.
            r_delta7 = r @ self.W_delta7
            r_max = r_delta7.max(dim=1, keepdim=True)[0]
            r_delta7 = r_delta7 / r_max

            r_history.append(r_delta7)
            
        return torch.stack(r_history, dim=1), torch.stack(bump_history, dim=1)


def cosine_similarity_loss(predicted_cosine_wave, true_angle):
    """
    Calculates the loss as the Mean Squared Error between the (x,y) components
    of the predicted angle and the true angle on a unit circle.
    This version follows the logic of explicitly decoding the angle first.
    """
    num_neurons = predicted_cosine_wave.shape[-1]
    preferred_angles = torch.linspace(0, 2 * np.pi, num_neurons,
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


def plot_ring_matrices(ring_rnn, output_dir="", step=None):
    """Modified to save to specific directory with step number."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if step is not None:
        fig.suptitle(f"Weight Matrices - Step {step}")
    else:
        fig.suptitle("Weight Matrices")
    
    weights = [ring_rnn.Wo, ring_rnn.Wl, ring_rnn.Wr]
    titles = ['Wo', 'Wl', 'Wr']
    
    for ax, w, title in zip(axes, weights, titles):
        im = ax.imshow(w.detach().cpu().numpy())
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if step is not None:
        output_filename = os.path.join(output_dir, f"weights_step_{step:06d}.png")
    else:
        output_filename = os.path.join(output_dir, "weights_final.png")
    
    plt.savefig(output_filename, dpi=100)
    plt.close(fig)


def create_weight_video(image_folder, output_video, fps=10):
    """Create a video from the saved weight matrix images using ffmpeg."""
    # Get list of all weight images
    pattern = os.path.join(image_folder, "weights_step_*.png")
    images = sorted(glob.glob(pattern))
    
    if not images:
        print(f"No images found in {image_folder}")
        return
    
    print(f"Creating video from {len(images)} images...")
    
    # Use ffmpeg to create the video
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', f'{image_folder}/weights_step_*.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file if it exists
        output_video
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video saved to {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print("Make sure ffmpeg is installed: sudo apt-get install ffmpeg")
    except FileNotFoundError:
        print("ffmpeg not found. Please install it: sudo apt-get install ffmpeg")


def decode_angle_from_argmax(activity):
    """Decode angle by finding the neuron with the maximum activity."""
    num_neurons = activity.shape[-1]
    preferred_angles = torch.linspace(0, 2 * np.pi, num_neurons, device=activity.device, requires_grad=False)
    
    # Find the index of the max activity neuron for each time step
    # activity shape: (batch, time, neurons)
    max_indices = torch.argmax(activity, dim=2) # Shape: (batch, time)
    
    # Map indices to angles
    decoded_angle = preferred_angles[max_indices]
    
    return decoded_angle


def decode_angle_from_population_vector(activity):
    """Decode angle using population vector average."""
    num_neurons = activity.shape[-1]
    preferred_angles = torch.linspace(0, 2 * np.pi, num_neurons, device=activity.device, requires_grad=False)
    # Reshape for broadcasting: (1, 1, num_neurons)
    preferred_angles = preferred_angles.view(1, 1, -1)
    
    # Activity has shape (batch, time, neurons)
    x_component = torch.cos(preferred_angles) * activity
    y_component = torch.sin(preferred_angles) * activity
    
    pop_vec_x = torch.sum(x_component, dim=2)
    pop_vec_y = torch.sum(y_component, dim=2)
    
    decoded_angle = torch.atan2(pop_vec_y, pop_vec_x)
    return (decoded_angle + 2 * np.pi) % (2 * np.pi)


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


def run_training_and_evaluation():
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Training Parameters ---
    num_neurons = 120
    training_steps = 1000
    learning_rate = 1e-3
    batch_size = 128
    seq_len = 120
    
    # --- Create output directory for weight images ---
    weight_dir = "ring_attractor_weights"
    os.makedirs(weight_dir, exist_ok=True)
    print(f"Weight images will be saved to: {weight_dir}")
    
    # === Create dataset for on-demand batch generation ===
    print(f"\nCreating AVIntegrationDataset for on-demand batch generation...")
    
    # Create dataset for training data generation
    dataset = AVIntegrationDataset(
        num_samples=training_steps * batch_size,  # Total samples we might need
        seq_len=seq_len,
        zero_padding_start_ratio=0.1,
        zero_ratios_in_rest=[0.2, 0.5, 0.8],
        device=device,  # Generate directly on GPU
        fast_mode=True   # Use fast vectorized generation
    )
    print("Dataset created - ready for on-demand generation.")
    
    # --- SANITY CHECK ---
    print("--- Sanity Check ---")
    print("Initializing a TRULY perfect model (fixed gain, no training)...")
    perfect_model = LeakyRingAttractor(num_neurons=360, tau=10, dt=1, 
                                       activation='gelu', initialization='debug_perfect')
    perfect_model.to(device)
    perfect_model.eval()
    
    # Generate one batch for sanity check
    av_signal, target_angle = dataset.generate_batch(batch_size)
    initial_angle = target_angle[:, 0]
    r_init_perfect = create_initial_bump(initial_angle, 360, device=device)

    predicted_wave, _ = perfect_model(av_signal, r_init=r_init_perfect)
    perfect_loss = cosine_similarity_loss(predicted_wave, target_angle)
    print(f"Loss for TRULY perfect model: {perfect_loss.item():.6f}")
    
    if perfect_loss.item() < 0.5:
        print("Sanity check passed: Loss for the fixed-gain model is reasonably low.")
    else:
        print("Warning: Loss for the fixed-gain model is higher than expected.")
    print("--------------------")

    # --- MODEL SETUP ---
    initial_weights = 'random'
    ring_rnn = LeakyRingAttractor(num_neurons=num_neurons, tau=10, dt=1, 
                                  activation='gelu', initialization=initial_weights,
                                  hidden_gain_neurons=3)
    ring_rnn.to(device)
    print(f"\nInitializing a model with {initial_weights} weights")
    print("--------------------")

    # Save initial weights
    plot_ring_matrices(ring_rnn, output_dir=weight_dir, step=0)
    optimizer = torch.optim.Adam(ring_rnn.parameters(), lr=learning_rate)

    # --- TRAINING LOOP ---
    print("\nStarting training with on-demand data generation...")
    loss_history = []
    
    for step in range(training_steps):
        # Generate batch on-demand
        av_signal, target_angle = dataset.generate_batch(batch_size)
        
        # Create initial bump based on true initial angle
        initial_angle = target_angle[:, 0]
        r_init = create_initial_bump(initial_angle, num_neurons, device=device)
        
        # Forward pass
        predicted_cosine_wave, bump_activity = ring_rnn(av_signal, r_init=r_init)
        
        # Calculate losses
        loss = cosine_similarity_loss(predicted_cosine_wave, target_angle)
        bump_loss = bump_amplitude_loss(bump_activity)

        total_loss = loss + 0.2 * bump_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(ring_rnn.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if step % 10 == 0:
            print(f"Step {step}/{training_steps}, Total Loss: {total_loss.item():.4f}, "
                  f"Main loss: {loss.item():.4f}, "
                  f"Bump loss: {bump_loss.item():.4f}")
            
            # Save weight matrices every 10 steps
            plot_ring_matrices(ring_rnn, output_dir=weight_dir, step=step)
    
    print("Training finished.")
    
    # Save final weights
    plot_ring_matrices(ring_rnn, output_dir=weight_dir, step=training_steps)
    
    # Create video from saved weight images
    video_path = "ring_attractor_weights_evolution.mp4"
    create_weight_video(weight_dir, video_path, fps=10)
    
    # --- EVALUATION ---
    print("\nEvaluating the TRAINED model...")
    ring_rnn.eval()
    
    # Generate test data using dataset
    # Create a temporary dataset for longer evaluation sequence
    test_dataset = AVIntegrationDataset(
        num_samples=1,
        seq_len=200,  # Longer sequence for evaluation
        zero_padding_start_ratio=0.01,
        zero_ratios_in_rest=[0.3],
        device=device,
        fast_mode=True
    )
    av_signal_test, target_angle_test = test_dataset.generate_batch(1)
    
    # Align the test data for evaluation
    initial_angle_test = target_angle_test[:, 0].unsqueeze(1)
    offset_test = np.pi - initial_angle_test
    aligned_target_angle_test = (target_angle_test + offset_test) % (2 * np.pi)

    # Get network activity from the trained model
    cos_activity, bump_activity = ring_rnn(av_signal_test)
    
    # Decode angles
    decoded_angle_pv = decode_angle_from_population_vector(cos_activity)
    decoded_angle_argmax = decode_angle_from_argmax(cos_activity)
    
    # Create visualization
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Performance of Trained Model")
    
    im = axes[0].imshow(cos_activity[0].detach().cpu().numpy().T, aspect='auto', interpolation='nearest')
    axes[0].set_title('Network Activity (Cosine Output)')
    axes[0].set_ylabel('Neuron')

    axes[1].plot(bump_activity[0][-1].detach().cpu().numpy().T, label='Last time step')
    axes[1].plot(bump_activity[0][0].detach().cpu().numpy().T, label='First time step')
    middle_step = bump_activity.shape[1] // 2
    axes[1].plot(bump_activity[0][middle_step].detach().cpu().numpy().T, label=f'Middle step ({middle_step})')
    axes[1].legend()
    axes[1].set_title('Bump Activity (EPG)')
    axes[1].set_ylabel('Neuron')

    axes[2].plot(cos_activity[0][-1].detach().cpu().numpy().T, label='Last time step')
    axes[2].plot(cos_activity[0][0].detach().cpu().numpy().T, label='First time step')
    middle_step = cos_activity.shape[1] // 2
    axes[2].plot(cos_activity[0][middle_step].detach().cpu().numpy().T, label=f'Middle step ({middle_step})')
    axes[2].legend()
    axes[2].set_title('Cos Activity (delta_7)')
    axes[2].set_ylabel('Neuron')

    axes[3].plot(av_signal_test[0].cpu().numpy(), label='AV Signal')
    axes[3].set_title('Input Angular Velocity')
    axes[3].set_ylabel('Velocity (rad/step)')
    axes[3].legend()

    axes[4].plot(aligned_target_angle_test[0].cpu().numpy(), label='Ground Truth Angle (Aligned)', linestyle='--')
    axes[4].plot(decoded_angle_pv[0].detach().cpu().numpy(), label='Decoded Angle (Pop. Vector)')
    axes[4].plot(decoded_angle_argmax[0].detach().cpu().numpy(), label='Decoded Angle (Argmax)', linestyle=':')
    axes[4].set_title('Angle Integration')
    axes[4].set_xlabel('Time Step')
    axes[4].set_ylabel('Angle (rad)')
    axes[4].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = "trained_model_integration_results.png"
    plt.savefig(output_filename)
    print(f"Saved evaluation plot to {output_filename}")
    plt.close(fig)


if __name__ == '__main__':
    run_training_and_evaluation()