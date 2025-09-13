import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Import the cleaned data generation module
from generate_av_integration_data import AVIntegrationDataset

# Import helper functions from train_ring_attractor
from train_ring_attractor import (
    non_linear,
    create_initial_bump,
    cosine_similarity_loss,
    bump_amplitude_loss,
    decode_angle_from_population_vector,
    decode_angle_from_argmax,
    plot_ring_matrices
)


class GeneralizedRingAttractor(nn.Module):
    """
    Generalized Ring Attractor model with arbitrary action dimensions.
    r += (-r + F(J0.dot(r) + J1*Wo.dot(r) + sum_i(A_i * Wa_i.dot(r)))) * dt / tau
    where A is an action vector of dimension k, and Wa is a tensor of shape (k, N, N)
    """
    def __init__(self, num_neurons, action_dim=2, tau=10.0, dt=1.0, activation='tanh',
                 initialization='random', hidden_gain_neurons=16):
        super().__init__()
        self.num_neurons = num_neurons
        self.action_dim = action_dim
        self.tau = tau
        self.dt = dt
        self.activation_name = activation
        self.initialization = initialization

        # Gain networks for each action dimension
        def create_gain_net():
            return nn.Sequential(
                nn.Linear(1, hidden_gain_neurons),
                nn.GELU(),
                nn.Linear(hidden_gain_neurons, 1),
                nn.Softplus()
            )

        self.gain_nets = nn.ModuleList([create_gain_net() for _ in range(action_dim)])

        # Define W_delta7 as a NON-LEARNABLE buffer
        indices = torch.arange(num_neurons, dtype=torch.float32)
        i = indices.unsqueeze(1)
        j = indices.unsqueeze(0)
        angle_diff = 2 * np.pi * (i - j) / num_neurons
        self.register_buffer('W_delta7', torch.cos(angle_diff))

        # Fixed parameters
        self.J0 = -0.1 * torch.ones(num_neurons, num_neurons)
        self.J1 = 0.1

        # Learnable parameters
        self.Wo = nn.Parameter(torch.randn(num_neurons, num_neurons) / np.sqrt(num_neurons))

        # Action weight tensor: (action_dim, num_neurons, num_neurons)
        if initialization == 'random':
            self.Wa = nn.Parameter(torch.randn(action_dim, num_neurons, num_neurons) / np.sqrt(num_neurons))
        elif initialization == 'perfect':
            # For backwards compatibility with L/R case
            indices = torch.arange(num_neurons, dtype=torch.float32)
            i = indices.unsqueeze(1)
            j = indices.unsqueeze(0)
            angle_diff = 2 * np.pi * (i - j) / num_neurons

            if action_dim == 2:
                # Classic L/R setup
                wl = torch.cos(angle_diff + np.pi / 4)
                wr = torch.cos(angle_diff - np.pi / 4)
                self.Wa = nn.Parameter(torch.stack([wl, wr], dim=0))
            else:
                # For general case, initialize with rotations at different phases
                wa_list = []
                for k in range(action_dim):
                    phase = k * 2 * np.pi / action_dim
                    wa_list.append(torch.cos(angle_diff + phase))
                self.Wa = nn.Parameter(torch.stack(wa_list, dim=0))
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

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

        # Compute gains for each action dimension
        gains = []
        for k in range(self.action_dim):
            # Extract magnitude for this action dimension
            action_mag = torch.abs(action_signal[:, :, k:k+1])  # (batch, seq, 1)
            gain = self.gain_nets[k](action_mag)  # (batch, seq, 1)
            gains.append(gain.squeeze(-1))  # Remove last dimension
        gains = torch.stack(gains, dim=2)  # (batch, seq, action_dim)

        # Apply gains to action signals
        A = action_signal * gains  # (batch, seq, action_dim)

        if r_init is None:
            initial_angle = torch.full((batch_size,), np.pi, device=self.Wo.device)
            r = create_initial_bump(initial_angle, self.num_neurons, device=self.Wo.device)
        else:
            r = r_init

        r_history = []
        bump_history = []

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


def plot_general_ring_matrices(ring_rnn, title_prefix=""):
    """Plot weight matrices for generalized ring attractor."""
    action_dim = ring_rnn.action_dim

    # Plot Wo and first few Wa matrices
    num_plots = min(4, 1 + action_dim)  # Wo + up to 3 action matrices
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    fig.suptitle(f"{title_prefix} Weight Matrices (Action Dim={action_dim})")

    # Plot Wo
    im = axes[0].imshow(ring_rnn.Wo.detach().cpu().numpy())
    axes[0].set_title('Wo (Maintenance)')
    fig.colorbar(im, ax=axes[0])

    # Plot Wa matrices
    for i in range(min(3, action_dim)):
        im = axes[i+1].imshow(ring_rnn.Wa[i].detach().cpu().numpy())
        axes[i+1].set_title(f'Wa[{i}] (Action {i})')
        fig.colorbar(im, ax=axes[i+1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f"{title_prefix.lower().replace(' ', '_')}general_ring_matrices.png"
    plt.savefig(output_filename)
    print(f"Saved weight matrices plot to {output_filename}")
    plt.close(fig)


def run_training_and_evaluation(action_dim=2):
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training GeneralizedRingAttractor with action_dim={action_dim}")
    print("Note: For angular velocity task, we always use 2D action signals [L, R]")

    # --- Training Parameters ---
    num_neurons = 120
    training_steps = 1000
    learning_rate = 1e-3
    batch_size = 128
    seq_len = 120

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

    # --- MODEL SETUP ---
    initial_weights = 'random'
    ring_rnn = GeneralizedRingAttractor(
        num_neurons=num_neurons,
        action_dim=action_dim,
        tau=10,
        dt=1,
        activation='gelu',
        initialization=initial_weights,
        hidden_gain_neurons=3
    )
    ring_rnn.to(device)
    print(f"\nInitializing a generalized model with {initial_weights} weights and action_dim={action_dim}")
    print("--------------------")

    plot_general_ring_matrices(ring_rnn, title_prefix=f"Initial {initial_weights}")
    optimizer = torch.optim.Adam(ring_rnn.parameters(), lr=learning_rate)

    # --- TRAINING LOOP ---
    print("\nStarting training with on-demand data generation...")
    loss_history = []
    bump_loss_history = []

    for step in range(training_steps):
        # Generate batch on-demand
        av_signal, target_angle = dataset.generate_batch(batch_size)

        # Convert angular velocity to action signals [L, R]
        action_signal = av_to_action_signal(av_signal)  # Always 2D for this task

        # Create initial bump based on true initial angle
        initial_angle = target_angle[:, 0]
        r_init = create_initial_bump(initial_angle, num_neurons, device=device)

        # Forward pass
        predicted_cosine_wave, bump_activity = ring_rnn(action_signal, r_init=r_init)

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
        bump_loss_history.append(bump_loss.item())

        if step % 10 == 0:
            print(f"Step {step}/{training_steps}, Total Loss: {total_loss.item():.4f}, "
                  f"Main loss: {loss.item():.4f}, "
                  f"Bump loss: {bump_loss.item():.4f}")

    print("Training finished.")

    # --- EVALUATION ---
    print("\nEvaluating the TRAINED generalized model...")
    ring_rnn.eval()

    plot_general_ring_matrices(ring_rnn, title_prefix="Trained")

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
    action_signal_test = av_to_action_signal(av_signal_test)  # Always 2D [L, R]

    # Align the test data for evaluation
    initial_angle_test = target_angle_test[:, 0].unsqueeze(1)
    offset_test = np.pi - initial_angle_test
    aligned_target_angle_test = (target_angle_test + offset_test) % (2 * np.pi)

    # Get network activity from the trained model
    cos_activity, bump_activity = ring_rnn(action_signal_test)

    # Decode angles
    decoded_angle_pv = decode_angle_from_population_vector(cos_activity)
    decoded_angle_argmax = decode_angle_from_argmax(cos_activity)

    # Create visualization
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"Performance of Trained Generalized Model (Action Dim={action_dim})")

    im = axes[0].imshow(cos_activity[0].detach().cpu().numpy().T, aspect='auto', interpolation='nearest')
    axes[0].set_title('Network Activity (Cosine Output)')
    axes[0].set_ylabel('Neuron')

    axes[1].plot(bump_activity[0][-1].detach().cpu().numpy().T, label='Last time step')
    axes[1].plot(bump_activity[0][0].detach().cpu().numpy().T, label='First time step')
    middle_step = bump_activity.shape[1] // 2
    axes[1].plot(bump_activity[0][middle_step].detach().cpu().numpy().T, label=f'Middle step ({middle_step})')
    axes[1].legend()
    axes[1].set_title('Bump Activity (EPG)')
    axes[1].set_ylabel('Activity')

    axes[2].plot(cos_activity[0][-1].detach().cpu().numpy().T, label='Last time step')
    axes[2].plot(cos_activity[0][0].detach().cpu().numpy().T, label='First time step')
    middle_step = cos_activity.shape[1] // 2
    axes[2].plot(cos_activity[0][middle_step].detach().cpu().numpy().T, label=f'Middle step ({middle_step})')
    axes[2].legend()
    axes[2].set_title('Cos Activity (delta_7)')
    axes[2].set_ylabel('Activity')

    axes[3].plot(av_signal_test[0].cpu().numpy(), label='AV Signal', color='black')
    axes[3].set_title('Input Angular Velocity')
    axes[3].set_ylabel('Velocity (rad/step)')
    axes[3].legend()

    # Plot action signals (always L and R for this task)
    axes[4].plot(action_signal_test[0, :, 0].cpu().numpy(),
                label='L (Left)', alpha=0.7, color='blue')
    axes[4].plot(action_signal_test[0, :, 1].cpu().numpy(),
                label='R (Right)', alpha=0.7, color='red')
    axes[4].set_title(f'Action Signals [L, R] (Model capacity: {action_dim}D)')
    axes[4].set_ylabel('Action Value')
    axes[4].legend()

    axes[5].plot(aligned_target_angle_test[0].cpu().numpy(), label='Ground Truth Angle (Aligned)', linestyle='--')
    axes[5].plot(decoded_angle_pv[0].detach().cpu().numpy(), label='Decoded Angle (Pop. Vector)')
    axes[5].plot(decoded_angle_argmax[0].detach().cpu().numpy(), label='Decoded Angle (Argmax)', linestyle=':')
    axes[5].set_title('Angle Integration')
    axes[5].set_xlabel('Time Step')
    axes[5].set_ylabel('Angle (rad)')
    axes[5].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f"trained_general_model_integration_results_dim{action_dim}.png"
    plt.savefig(output_filename)
    print(f"Saved evaluation plot to {output_filename}")
    plt.close(fig)

    # Plot loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Training Curves (Action Dim={action_dim})")

    ax1.plot(loss_history)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss Over Training')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2.plot(bump_loss_history)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Bump Loss')
    ax2.set_title('Bump Loss Over Training')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'training_curves_general_dim{action_dim}.png')
    print(f"Saved training curves to training_curves_general_dim{action_dim}.png")
    plt.close(fig)

    return ring_rnn, loss_history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Generalized Ring Attractor')
    parser.add_argument('--action_dim', type=int, default=2,
                       help='Number of action dimensions (default: 2 for L/R)')
    args = parser.parse_args()

    run_training_and_evaluation(action_dim=args.action_dim)