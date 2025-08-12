import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import the data generation module
from generate_av_integration_data import AVIntegrationDataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer (like GPT) for angular velocity integration.
    Uses causal self-attention which is more natural for autoregressive integration.
    """
    def __init__(self, d_model=128, nhead=8, num_layers=8, 
                 dim_feedforward=512, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Initial angle encoder - stronger encoding
        self.initial_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Decoder layers (using TransformerDecoderLayer for proper causal attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False  # Post-LN works better for this task
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection heads
        self.sin_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.cos_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize weights with smaller values
        # self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1)  # Standard initialization
                
    def generate_causal_mask(self, sz):
        """Generate causal mask for autoregressive attention."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
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
        memory = self.initial_encoder(initial_sincos).unsqueeze(1)  # (batch, 1, d_model)
        
        # Project input signal
        av_signal = av_signal.unsqueeze(2)  # (batch, seq, 1)
        tgt = self.input_proj(av_signal)  # (batch, seq, d_model)
        
        # Add positional encoding
        tgt = self.pos_encoder(tgt)
        
        # Generate causal mask
        tgt_mask = self.generate_causal_mask(seq_len).to(device)
        
        # Apply transformer decoder
        # Memory is the initial state, tgt is the sequence
        # We expand memory to attend to all positions
        memory = memory.expand(-1, seq_len, -1)
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None
        )
        
        # Generate sin and cos predictions separately with tanh to constrain to [-1, 1]
        sin_pred = torch.tanh(self.sin_head(output).squeeze(-1))
        cos_pred = torch.tanh(self.cos_head(output).squeeze(-1))
        
        # Stack predictions
        output = torch.stack([sin_pred, cos_pred], dim=-1)
        
        return output


class CausalTransformer(nn.Module):
    """
    Causal Transformer for angular velocity integration.
    Uses transformer encoder layers with causal masking for autoregressive processing.
    Concatenates initial state with the sequence.
    """
    def __init__(self, d_model=128, nhead=8, num_layers=8, 
                 dim_feedforward=512, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Token embeddings with layer norm
        self.velocity_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model)
        )
        self.initial_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Standard transformer decoder blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=False
            ) for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        
        # Dropout before output for regularization
        # self.output_dropout = nn.Dropout(dropout)
        
        # Output heads
        self.output_proj = nn.Linear(d_model, 2)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1)
                
    def forward(self, av_signal, initial_angle=None):
        batch_size, seq_len = av_signal.shape
        device = av_signal.device
        
        if initial_angle is None:
            initial_angle = torch.zeros(batch_size, device=device)
        
        # Embed initial condition as first token
        initial_sincos = torch.stack([torch.sin(initial_angle), torch.cos(initial_angle)], dim=1)
        initial_token = self.initial_embed(initial_sincos).unsqueeze(1)  # (batch, 1, d_model)
        
        # Embed velocity sequence
        velocity_tokens = self.velocity_embed(av_signal.unsqueeze(-1))  # (batch, seq, d_model)
        
        # Concatenate: [initial_token, velocity_tokens]
        tokens = torch.cat([initial_token, velocity_tokens], dim=1)  # (batch, seq+1, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(tokens)
        
        # Create causal mask for the full sequence
        sz = seq_len + 1
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=mask)
        
        x = self.ln_final(x)
        x = self.output_proj(x[:, 1:, :])  # (batch, seq+1, 2)
        
        # Apply dropout for regularization during training
        # x = self.output_dropout(x)
        
        # Project to sin/cos (skip the initial token in output) with tanh
        # output = self.output_proj(x[:, 1:, :])  # (batch, seq+1, 2)
        
        # Add small Gaussian noise during training to prevent collapse
        # if self.training:
        #     noise = torch.randn_like(output) * 0.1  # Small noise for regularization
        #     output = output + noise
        
        return x


def transformer_loss(predictions, true_angles, av_signal, 
                            lambda_smooth=0.05, lambda_norm=0.1, lambda_cumsum=0.3):
    """
    Enhanced loss function for decoder transformer:
    1. Main loss on sin/cos predictions
    2. Smoothness penalty
    3. Norm regularization
    4. Cumulative sum consistency
    """
    batch_size, seq_len, _ = predictions.shape
    
    # Extract predictions
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
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        expected_diff = av_signal[:, 1:]
        smoothness_loss = torch.mean((angle_diff - expected_diff)**2)
    else:
        smoothness_loss = torch.tensor(0.0)
    
    # Norm regularization
    norm_loss = torch.mean((norm - 1)**2)
    
    # Cumulative sum consistency loss
    # The predicted angle should be close to initial + cumsum(velocities)
    if seq_len > 1:
        pred_angles = torch.atan2(pred_sin_norm, pred_cos_norm)
        cumsum_velocities = torch.cumsum(av_signal, dim=1)
        initial_angle = true_angles[:, 0:1]
        expected_angles = initial_angle + cumsum_velocities
        
        # Circular distance
        angle_error = pred_angles - expected_angles
        angle_error = torch.atan2(torch.sin(angle_error), torch.cos(angle_error))
        cumsum_loss = torch.mean(angle_error**2)
    else:
        cumsum_loss = torch.tensor(0.0)
    
    total_loss = (main_loss + 
                 lambda_smooth * smoothness_loss + 
                 lambda_norm * norm_loss +
                 lambda_cumsum * cumsum_loss)
    
    return total_loss, main_loss, smoothness_loss, norm_loss, cumsum_loss


def evaluate_autoregressive(model, av_signal, initial_angle, chunk_size=120):
    """
    Evaluate model autoregressively on long sequences by processing in chunks.
    This matches how the model was trained.
    
    Args:
        model: The trained transformer model
        av_signal: Angular velocity signal (batch_size, seq_len)
        initial_angle: Initial angles (batch_size,)
        chunk_size: Size of chunks to process (should match training length)
    
    Returns:
        Predictions for the entire sequence (batch_size, seq_len, 2)
    """
    device = av_signal.device
    batch_size, total_seq_len = av_signal.shape
    
    # Initialize predictions list
    all_predictions = []
    
    # Process in chunks
    for start_idx in range(0, total_seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, total_seq_len)
        
        # Get current chunk of angular velocities
        av_chunk = av_signal[:, start_idx:end_idx]
        
        # Use initial angle for first chunk, otherwise use last predicted angle
        if start_idx == 0:
            chunk_initial = initial_angle
        else:
            # Get the last predicted angle from previous chunk
            last_pred = all_predictions[-1][:, -1, :]  # (batch, 2)
            # Convert sin/cos back to angle
            chunk_initial = torch.atan2(last_pred[:, 0], last_pred[:, 1])
        
        # Get predictions for this chunk
        with torch.no_grad():
            chunk_predictions = model(av_chunk, chunk_initial)
        
        all_predictions.append(chunk_predictions)
    
    # Concatenate all predictions
    return torch.cat(all_predictions, dim=1)


def train_causal_transformer():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training parameters
    model_version = "v2"  # "v1" for cross-attention, "v2" for pure decoder - v2 is better for sequential integration
    d_model = 64
    nhead = 8  # Single head for simplicity
    num_layers = 8  # More layers for decoder-only
    dim_feedforward = 128
    dropout = 0.01
    training_steps = 20000
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
    
    # Create model
    if model_version == "v1":
        print(f"\nInitializing Decoder-Only Transformer (with cross-attention)")
        model = DecoderOnlyTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=1201  # Support longer sequences for testing generalization (+1 for initial token in V2)
        )
    else:
        print(f"\nInitializing Causal Transformer")
        model = CausalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=1201  # Support longer sequences for testing generalization (+1 for initial token in V2)
        )
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: d_model={d_model}, heads={nhead}, layers={num_layers}")
    print(f"Total parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    # scheduler = CosineAnnealingLR(optimizer, T_max=training_steps, eta_min=1e-5)
    
    # Training loop
    print("\nStarting training...")
    loss_history = []
    main_loss_history = []
    
    for step in range(training_steps):
        # Generate batch
        av_signal, target_angle = dataset.generate_batch(batch_size)
        initial_angle = target_angle[:, 0]
        
        # Forward pass
        predictions = model(av_signal, initial_angle)
        
        # Calculate loss
        total_loss, main_loss, smooth_loss, norm_loss, cumsum_loss = transformer_loss(
            predictions, target_angle, av_signal,
            lambda_smooth=0.05, lambda_norm=0.1, lambda_cumsum=0.3
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # scheduler.step()
        
        # Record losses
        loss_history.append(total_loss.item())
        main_loss_history.append(main_loss.item())
        
        if step % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}/{training_steps}, Total: {total_loss.item():.4f}, "
                  f"Main: {main_loss.item():.4f}, Smooth: {smooth_loss:.4f}, "
                  f"Cumsum: {cumsum_loss:.4f}, LR: {current_lr:.2e}")
            
            # Debug: Check raw predictions
            # with torch.no_grad():
            #     pred_sample = predictions[0, :5]  # First 5 timesteps of first batch
            #     print(f"  Raw predictions (first 5): sin={pred_sample[:, 0].tolist()}, cos={pred_sample[:, 1].tolist()}")
            #     print(f"  Prediction stats: sin mean={predictions[:,:,0].mean():.4f}, std={predictions[:,:,0].std():.4f}")
            #     print(f"  Norm loss: {norm_loss.item():.4f}")
    
    print("Training finished.")
    
    # Evaluation
    print("\nEvaluating Causal Transformer...")
    model.eval()
    
    # Test sequence length (can be any length now!)
    test_seq_len = 1200  # 10x training length
    
    # Create test dataset
    test_dataset = AVIntegrationDataset(
        num_samples=100,
        seq_len=test_seq_len,
        zero_padding_start_ratio=0.01,  # Intentional distribution shift
        zero_ratios_in_rest=[0.2, 0.5, 0.8],
        device=device,
        fast_mode=True
    )
    
    with torch.no_grad():
        av_test, angle_test = test_dataset.generate_batch(100)
        initial_angles = angle_test[:, 0]
        
        # Method 1: Single forward pass (will fail for sequences > training length)
        print(f"\n1. Single forward pass ({test_seq_len} steps at once):")
        # Can do single pass if test length <= training length
        pred_test_single = model(av_test, initial_angles)
        
        # Decode angles
        pred_sin = pred_test_single[:, :, 0]
        pred_cos = pred_test_single[:, :, 1]
        norm = torch.sqrt(pred_sin**2 + pred_cos**2 + 1e-8)
        decoded_test_single = torch.atan2(pred_sin/norm, pred_cos/norm)
        decoded_test_single = (decoded_test_single + 2 * np.pi) % (2 * np.pi)
        
        # Calculate error
        angular_error_single = torch.abs(decoded_test_single - angle_test)
        angular_error_single = torch.min(angular_error_single, 2*np.pi - angular_error_single)
        mean_error_single = angular_error_single.mean().item()
        final_error_single = angular_error_single[:, -1].mean().item()
        
        print(f"   Mean Error: {mean_error_single:.4f} rad ({np.degrees(mean_error_single):.2f}°)")
        print(f"   Final Error: {final_error_single:.4f} rad ({np.degrees(final_error_single):.2f}°)")

        
        # Method 2: Autoregressive chunked generation (works for any length)
        print(f"\n2. Autoregressive generation (chunks of {seq_len} steps):")
        pred_test = evaluate_autoregressive(model, av_test, initial_angles, chunk_size=seq_len)
        
        # Decode angles
        pred_sin = pred_test[:, :, 0]
        pred_cos = pred_test[:, :, 1]
        norm = torch.sqrt(pred_sin**2 + pred_cos**2 + 1e-8)
        decoded_test = torch.atan2(pred_sin/norm, pred_cos/norm)
        decoded_test = (decoded_test + 2 * np.pi) % (2 * np.pi)
        
        # Calculate error
        angular_error = torch.abs(decoded_test - angle_test)
        angular_error = torch.min(angular_error, 2*np.pi - angular_error)
        mean_error = angular_error.mean().item()
        std_error = angular_error.std().item()
        final_error = angular_error[:, -1].mean().item()
        
        print(f"   Mean Error: {mean_error:.4f} rad ({np.degrees(mean_error):.2f}°)")
        print(f"   Final Error: {final_error:.4f} rad ({np.degrees(final_error):.2f}°)")
        
    print(f"\nFinal Performance Metrics:")
    print(f"Mean Angular Error: {mean_error:.4f} rad ({np.degrees(mean_error):.2f}°)")
    print(f"Std Error: {std_error:.4f} rad ({np.degrees(std_error):.2f}°)")
    print(f"Final Step Error: {final_error:.4f} rad ({np.degrees(final_error):.2f}°)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Causal Transformer Training Results", fontsize=14)
    
    # Loss curves
    axes[0, 0].plot(loss_history, 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(main_loss_history, 'g-', alpha=0.7)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Main Loss')
    axes[0, 1].set_title('Main Loss (Sin/Cos MSE)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample trajectory
    sample_idx = 0
    axes[1, 0].plot(angle_test[sample_idx].cpu().numpy()[:200], 'k--', label='Ground Truth', alpha=0.7)
    axes[1, 0].plot(decoded_test[sample_idx].cpu().numpy()[:200], 'r-', label='Prediction', alpha=0.7)
    axes[1, 0].axvline(x=seq_len, color='gray', linestyle=':', label='Training length')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Angle (rad)')
    axes[1, 0].set_title('Sample Trajectory (first 200 steps)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error over time
    mean_error_over_time = angular_error.mean(dim=0).cpu().numpy()
    axes[1, 1].plot(mean_error_over_time, 'r-', alpha=0.7)
    axes[1, 1].axvline(x=seq_len, color='gray', linestyle=':', label='Training length')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Mean Angular Error (rad)')
    axes[1, 1].set_title('Error Growth Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decoder_transformer_training.png', dpi=150)
    print("Saved training plot to decoder_transformer_training.png")
    
    return model, loss_history


if __name__ == '__main__':
    model, loss_history = train_causal_transformer()