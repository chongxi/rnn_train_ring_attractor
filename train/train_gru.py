import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import the data generation module
from utils.generate_av_integration_data import AVIntegrationDataset


class GRU_Integrator(nn.Module):
    """
    GRU model with:
    - Initial angle encoding
    - Residual connections
    - Better initialization
    - Layer normalization
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initial angle encoder
        self.initial_encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # GRU with dropout
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output layers with residual connection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
                        
    def forward(self, av_signal, initial_angle=None):
        """
        Args:
            av_signal: Angular velocity signal, shape (batch_size, seq_len)
            initial_angle: Initial angle in radians, shape (batch_size,)
        
        Returns:
            angle_predictions: Predicted sin and cos values, shape (batch_size, seq_len, 2)
        """
        batch_size, seq_len = av_signal.shape
        device = av_signal.device
        
        # Encode initial angle
        if initial_angle is None:
            initial_angle = torch.zeros(batch_size, device=device)
        
        initial_sincos = torch.stack([torch.sin(initial_angle), torch.cos(initial_angle)], dim=1)
        h0 = self.initial_encoder(initial_sincos)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # Project input
        av_signal = av_signal.unsqueeze(2)  # (batch, seq, 1)
        x = self.input_proj(av_signal)  # (batch, seq, hidden)
        
        # Pass through GRU
        gru_out, _ = self.gru(x, h0)
        
        # Apply layer norm
        gru_out = self.layer_norm(gru_out)
        
        # Output projection
        output = self.output_proj(gru_out)
        
        # Add initial angle as bias (helps with integration)
        output = output + initial_sincos.unsqueeze(1)
        
        return output


def integration_aware_loss(predictions, true_angles, av_signal, lambda_smooth=0.1):
    """
    Enhanced loss function that:
    1. Compares predicted vs true angles
    2. Penalizes inconsistent integration (smoothness)
    3. Encourages proper velocity-angle relationship
    """
    batch_size, seq_len, _ = predictions.shape
    
    # Basic circular loss
    pred_sin = predictions[:, :, 0]
    pred_cos = predictions[:, :, 1]
    
    # Normalize to unit circle
    norm = torch.sqrt(pred_sin**2 + pred_cos**2 + 1e-8)
    pred_sin_norm = pred_sin / norm
    pred_cos_norm = pred_cos / norm
    
    true_sin = torch.sin(true_angles)
    true_cos = torch.cos(true_angles)
    
    # Main loss
    main_loss = torch.mean((pred_sin_norm - true_sin)**2 + (pred_cos_norm - true_cos)**2)
    
    # Smoothness loss - penalize large jumps
    if seq_len > 1:
        pred_angles = torch.atan2(pred_sin_norm, pred_cos_norm)
        angle_diff = pred_angles[:, 1:] - pred_angles[:, :-1]
        # Wrap angle differences to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # Expected angle difference from velocity
        expected_diff = av_signal[:, 1:] * 1.0  # Assuming dt=1
        
        smoothness_loss = torch.mean((angle_diff - expected_diff)**2)
    else:
        smoothness_loss = 0
    
    total_loss = main_loss + lambda_smooth * smoothness_loss
    
    return total_loss, main_loss, smoothness_loss


def train_gru():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training parameters
    hidden_size = 128
    num_layers = 2
    dropout = 0.1
    training_steps = 2000  # More training steps
    initial_lr = 3e-4
    batch_size = 128
    seq_len = 120
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = AVIntegrationDataset(
        num_samples=training_steps * batch_size,
        seq_len=seq_len,
        zero_padding_start_ratio=0.1,
        zero_ratios_in_rest=[0.2, 0.5, 0.8],
        device=device,
        fast_mode=True
    )
    print("Dataset created.")
    
    # Create GRU model
    print(f"\nInitializing GRU model")
    print(f"Hidden size: {hidden_size}, Layers: {num_layers}, Dropout: {dropout}")
    model = GRU_Integrator(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=2,
        dropout=dropout
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
    
    # Training loop
    print("\nStarting training...")
    loss_history = []
    main_loss_history = []
    smooth_loss_history = []
    
    for step in range(training_steps):
        # Generate batch
        av_signal, target_angle = dataset.generate_batch(batch_size)
        initial_angle = target_angle[:, 0]
        
        # Forward pass
        predictions = model(av_signal, initial_angle)
        
        # Calculate loss
        total_loss, main_loss, smooth_loss = integration_aware_loss(
            predictions, target_angle, av_signal, lambda_smooth=0.1
        )
        
        # Backward pass
        optimizer.zero_grad()
        # total_loss.backward()
        main_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update scheduler
        # scheduler.step(total_loss)
        
        # Record losses
        loss_history.append(total_loss.item())
        main_loss_history.append(main_loss.item())
        smooth_loss_history.append(smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else smooth_loss)
        
        if step % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}/{training_steps}, Total Loss: {total_loss.item():.4f}, "
                  f"Main: {main_loss.item():.4f}, Smooth: {smooth_loss:.4f}, LR: {current_lr:.2e}")
    
    print("Training finished.")
    
    # Evaluation
    print("\nEvaluating GRU model...")
    model.eval()
    
    # Create test dataset
    test_dataset = AVIntegrationDataset(
        num_samples=1,
        seq_len=200,
        zero_padding_start_ratio=0.01,
        zero_ratios_in_rest=[0.3],
        device=device,
        fast_mode=True
    )
    
    with torch.no_grad():
        av_signal_test, target_angle_test = test_dataset.generate_batch(1)
        initial_angle_test = target_angle_test[:, 0]
        predictions_test = model(av_signal_test, initial_angle_test)
        
        # Decode angles
        pred_sin = predictions_test[:, :, 0]
        pred_cos = predictions_test[:, :, 1]
        norm = torch.sqrt(pred_sin**2 + pred_cos**2 + 1e-8)
        pred_sin = pred_sin / norm
        pred_cos = pred_cos / norm
        decoded_angles = torch.atan2(pred_sin, pred_cos)
        decoded_angles = (decoded_angles + 2 * np.pi) % (2 * np.pi)
    
    # Align for visualization
    offset_test = np.pi - initial_angle_test.unsqueeze(1)
    aligned_target_angle = (target_angle_test + offset_test) % (2 * np.pi)
    aligned_decoded_angle = (decoded_angles + offset_test) % (2 * np.pi)
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("GRU Model Performance on Angular Velocity Integration")
    
    # Plot 1: Angular velocity
    axes[0].plot(av_signal_test[0].cpu().numpy(), 'b-', label='Angular Velocity')
    axes[0].set_ylabel('Velocity (rad/step)')
    axes[0].set_title('Input Signal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Sin/Cos predictions
    axes[1].plot(pred_sin[0].cpu().numpy(), 'b-', label='Predicted sin(θ)', alpha=0.7)
    axes[1].plot(pred_cos[0].cpu().numpy(), 'r-', label='Predicted cos(θ)', alpha=0.7)
    axes[1].plot(torch.sin(target_angle_test[0]).cpu().numpy(), 'b--', label='True sin(θ)', alpha=0.5)
    axes[1].plot(torch.cos(target_angle_test[0]).cpu().numpy(), 'r--', label='True cos(θ)', alpha=0.5)
    axes[1].set_ylabel('Value')
    axes[1].set_title('Sin/Cos Predictions')
    axes[1].legend(ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1.5, 1.5)
    
    # Plot 3: Integrated angle
    axes[2].plot(aligned_target_angle[0].cpu().numpy(), 'k--', label='Ground Truth', linewidth=2)
    axes[2].plot(aligned_decoded_angle[0].cpu().numpy(), 'b-', label='GRU Prediction', alpha=0.8)
    axes[2].set_ylabel('Angle (rad)')
    axes[2].set_title('Integrated Angle')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Error
    error = torch.abs(aligned_decoded_angle[0] - aligned_target_angle[0])
    error = torch.min(error, 2*np.pi - error)  # Circular distance
    axes[3].plot(error.cpu().numpy(), 'r-', label='Angular Error')
    axes[3].set_xlabel('Time Step')
    axes[3].set_ylabel('Error (rad)')
    axes[3].set_title('Prediction Error')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gru_integration_results.png', dpi=150)
    print("Saved evaluation plot to gru_integration_results.png")
    plt.close()
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(loss_history, 'b-', alpha=0.7, label='Total Loss')
    ax1.plot(main_loss_history, 'g-', alpha=0.7, label='Main Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(smooth_loss_history, 'r-', alpha=0.7, label='Smoothness Loss')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Smoothness Loss')
    ax2.set_title('Smoothness Loss History')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gru_loss_history.png', dpi=150)
    print("Saved loss history to gru_loss_history.png")
    plt.close()
    
    # Calculate performance metrics
    with torch.no_grad():
        test_dataset_large = AVIntegrationDataset(
            num_samples=100,
            seq_len=seq_len,
            zero_padding_start_ratio=0.1,
            zero_ratios_in_rest=[0.2, 0.5, 0.8],
            device=device,
            fast_mode=True
        )
        av_test, angle_test = test_dataset_large.generate_batch(100)
        initial_angles = angle_test[:, 0]
        pred_test = model(av_test, initial_angles)
        
        # Calculate losses
        total_loss, main_loss, _ = integration_aware_loss(pred_test, angle_test, av_test)
        
        # Decode angles and calculate error
        pred_sin = pred_test[:, :, 0]
        pred_cos = pred_test[:, :, 1]
        norm = torch.sqrt(pred_sin**2 + pred_cos**2 + 1e-8)
        decoded_test = torch.atan2(pred_sin/norm, pred_cos/norm)
        decoded_test = (decoded_test + 2 * np.pi) % (2 * np.pi)
        
        angular_error = torch.abs(decoded_test - angle_test)
        angular_error = torch.min(angular_error, 2*np.pi - angular_error)
        mean_angular_error = torch.mean(angular_error).item()
        
    print(f"\nFinal Performance Metrics:")
    print(f"Test Loss: {total_loss.item():.4f}")
    print(f"Mean Angular Error: {mean_angular_error:.4f} rad ({np.degrees(mean_angular_error):.2f} degrees)")
    
    # Compare with ring attractor theoretical performance
    print(f"\nFor reference, the ring attractor model achieves ~0.5 loss after training")
    print(f"This GRU model achieved {total_loss.item():.4f} loss")
    
    return model, loss_history


if __name__ == '__main__':
    model, loss_history = train_gru()