import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from generate_av_integration_data import AVIntegrationDataset
from train_ring_attractor import LeakyRingAttractor, create_initial_bump, decode_angle_from_population_vector, cosine_similarity_loss
from train_gru import GRU_Integrator, integration_aware_loss



training_steps = 1000
batch_size = 128
seq_len = 60

def calculate_angular_error(pred_angles, true_angles):
    """Calculate circular distance error between predicted and true angles."""
    error = torch.abs(pred_angles - true_angles)
    # Handle circular distance
    error = torch.min(error, 2*np.pi - error)
    return error

def evaluate_trained_models():
    """Train both models and compare them using the same angular error metric."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create training dataset
    print("\nCreating training dataset...")
    train_dataset = AVIntegrationDataset(
        num_samples=training_steps * batch_size,
        seq_len=seq_len,
        zero_padding_start_ratio=0.1,
        zero_ratios_in_rest=[0.2, 0.5, 0.8],
        device=device,
        fast_mode=True
    )
    
    # Create test dataset with more samples for robust evaluation
    print("\nCreating test dataset...")
    test_dataset = AVIntegrationDataset(
        num_samples=200,  # 500 test sequences
        seq_len=500,      # Longer sequences to test integration
        zero_padding_start_ratio=0.01,
        zero_ratios_in_rest=[0.2, 0.5, 0.8],
        device=device,
        fast_mode=True
    )
    
    # Initialize models
    print("\n" + "="*60)
    print("TRAINING RING ATTRACTOR MODEL")
    print("="*60)
    
    ring_model = LeakyRingAttractor(
        num_neurons=120,
        tau=10,
        dt=1,
        activation='gelu',
        initialization='random',
        hidden_gain_neurons=3
    )
    ring_model.to(device)
    
    ring_optimizer = torch.optim.Adam(ring_model.parameters(), lr=1e-3)
    ring_losses = []
    
    # Train Ring Attractor
    for step in range(training_steps):
        av_signal, target_angle = train_dataset.generate_batch(batch_size)
        initial_angle = target_angle[:, 0]
        r_init = create_initial_bump(initial_angle, ring_model.num_neurons, device=device)
        
        predicted_cosine_wave, bump_activity = ring_model(av_signal, r_init=r_init)
        loss = cosine_similarity_loss(predicted_cosine_wave, target_angle)
        
        ring_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ring_model.parameters(), 1.0)
        ring_optimizer.step()
        
        ring_losses.append(loss.item())
        if step % 100 == 0:
            print(f"Step {step}/{training_steps}, Loss: {loss.item():.4f}")
    
    print("\n" + "="*60)
    print("TRAINING GRU MODEL")
    print("="*60)
    
    gru_model = GRU_Integrator(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        output_size=2,
        dropout=0.1
    )
    gru_model.to(device)
    
    gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=3e-4)
    gru_losses = []
    
    # Train GRU
    for step in range(training_steps):
        av_signal, target_angle = train_dataset.generate_batch(batch_size)
        initial_angle = target_angle[:, 0]
        
        predictions = gru_model(av_signal, initial_angle)
        total_loss, main_loss, smooth_loss = integration_aware_loss(
            predictions, target_angle, av_signal, lambda_smooth=0.1
        )
        
        gru_optimizer.zero_grad()
        main_loss.backward()
        torch.nn.utils.clip_grad_norm_(gru_model.parameters(), 1.0)
        gru_optimizer.step()
        
        gru_losses.append(total_loss.item())
        if step % 100 == 0:
            print(f"Step {step}/{training_steps}, Loss: {total_loss.item():.4f}")
    
    # Evaluate both models on the same test set
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    ring_model.eval()
    gru_model.eval()
    
    ring_errors_all = []
    gru_errors_all = []
    
    # Process test data in batches
    test_batch_size = 50
    num_test_batches = 200 // test_batch_size
    
    with torch.no_grad():
        for i in range(num_test_batches):
            # Get test batch
            av_signal_test, target_angle_test = test_dataset.generate_batch(test_batch_size)
            initial_angles_test = target_angle_test[:, 0]
            
            # Ring Attractor predictions
            r_init_test = create_initial_bump(initial_angles_test, ring_model.num_neurons, device=device)
            ring_cos_output, _ = ring_model(av_signal_test, r_init=r_init_test)
            ring_decoded = decode_angle_from_population_vector(ring_cos_output)
            
            # GRU predictions
            gru_output = gru_model(av_signal_test, initial_angles_test)
            gru_sin = gru_output[:, :, 0]
            gru_cos = gru_output[:, :, 1]
            norm = torch.sqrt(gru_sin**2 + gru_cos**2 + 1e-8)
            gru_decoded = torch.atan2(gru_sin/norm, gru_cos/norm)
            gru_decoded = (gru_decoded + 2 * np.pi) % (2 * np.pi)
            
            # Calculate errors
            ring_errors = calculate_angular_error(ring_decoded, target_angle_test)
            gru_errors = calculate_angular_error(gru_decoded, target_angle_test)
            
            ring_errors_all.append(ring_errors)
            gru_errors_all.append(gru_errors)
    
    # Concatenate all errors
    ring_errors_all = torch.cat(ring_errors_all, dim=0)  # Shape: (200, 500)
    gru_errors_all = torch.cat(gru_errors_all, dim=0)    # Shape: (200, 500)
    
    # Calculate statistics
    ring_mean_error = ring_errors_all.mean().item()
    ring_std_error = ring_errors_all.std().item()
    ring_max_error = ring_errors_all.max().item()
    ring_final_errors = ring_errors_all[:, -1].mean().item()  # Error at final time step
    
    gru_mean_error = gru_errors_all.mean().item()
    gru_std_error = gru_errors_all.std().item()
    gru_max_error = gru_errors_all.max().item()
    gru_final_errors = gru_errors_all[:, -1].mean().item()
    
    # Print results
    print("\n" + "="*60)
    print("ANGULAR ERROR COMPARISON (200 test sequences, 500 time steps each)")
    print("="*60)
    
    print(f"\nRing Attractor Model:")
    print(f"  Mean error: {ring_mean_error:.4f} rad ({np.degrees(ring_mean_error):.2f}°)")
    print(f"  Std error:  {ring_std_error:.4f} rad ({np.degrees(ring_std_error):.2f}°)")
    print(f"  Max error:  {ring_max_error:.4f} rad ({np.degrees(ring_max_error):.2f}°)")
    print(f"  Mean final error: {ring_final_errors:.4f} rad ({np.degrees(ring_final_errors):.2f}°)")
    
    print(f"\nGRU Model:")
    print(f"  Mean error: {gru_mean_error:.4f} rad ({np.degrees(gru_mean_error):.2f}°)")
    print(f"  Std error:  {gru_std_error:.4f} rad ({np.degrees(gru_std_error):.2f}°)")
    print(f"  Max error:  {gru_max_error:.4f} rad ({np.degrees(gru_max_error):.2f}°)")
    print(f"  Mean final error: {gru_final_errors:.4f} rad ({np.degrees(gru_final_errors):.2f}°)")
    
    print(f"\nRelative Performance:")
    print(f"  GRU error is {ring_mean_error/gru_mean_error:.2f}x smaller than Ring Attractor")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison: Angular Integration Error Analysis", fontsize=16)
    
    # Plot 1: Training loss curves
    ax = axes[0, 0]
    ax.plot(ring_losses, 'b-', alpha=0.7, label='Ring Attractor')
    ax.plot(gru_losses, 'r-', alpha=0.7, label='GRU')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (Note: Different loss functions)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error over time (averaged across all test sequences)
    ax = axes[0, 1]
    ring_error_mean_over_time = ring_errors_all.mean(dim=0).cpu().numpy()
    gru_error_mean_over_time = gru_errors_all.mean(dim=0).cpu().numpy()
    time_steps = np.arange(500)
    ax.plot(time_steps, ring_error_mean_over_time, 'b-', label='Ring Attractor', linewidth=2)
    ax.plot(time_steps, gru_error_mean_over_time, 'r-', label='GRU', linewidth=2)
    ax.fill_between(time_steps, 
                    ring_error_mean_over_time - ring_errors_all.std(dim=0).cpu().numpy(),
                    ring_error_mean_over_time + ring_errors_all.std(dim=0).cpu().numpy(),
                    alpha=0.2, color='blue')
    ax.fill_between(time_steps,
                    gru_error_mean_over_time - gru_errors_all.std(dim=0).cpu().numpy(),
                    gru_error_mean_over_time + gru_errors_all.std(dim=0).cpu().numpy(),
                    alpha=0.2, color='red')
    # Add vertical line at training sequence length (120 steps)
    ax.axvline(x=seq_len, color='gray', linestyle=':', alpha=0.7, linewidth=2, 
               label='Training seq length')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Angular Error (rad)')
    ax.set_title('Mean Error Over Time (with std bands)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution histogram
    ax = axes[1, 0]
    ax.hist(ring_errors_all.flatten().cpu().numpy(), bins=50, alpha=0.5, 
            label=f'Ring Attractor (μ={ring_mean_error:.3f})', color='blue', density=True)
    ax.hist(gru_errors_all.flatten().cpu().numpy(), bins=50, alpha=0.5,
            label=f'GRU (μ={gru_mean_error:.3f})', color='red', density=True)
    ax.set_xlabel('Angular Error (rad)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution (All time steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final time step errors
    ax = axes[1, 1]
    final_ring_errors = ring_errors_all[:, -1].cpu().numpy()
    final_gru_errors = gru_errors_all[:, -1].cpu().numpy()
    ax.boxplot([final_ring_errors, final_gru_errors], labels=['Ring Attractor', 'GRU'])
    ax.set_ylabel('Angular Error at Final Step (rad)')
    ax.set_title('Integration Drift (Error at t=500)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fair_model_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot to fair_model_comparison.png")
    plt.close()
    
    # Save a sample trajectory comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Sample Trajectory Comparison", fontsize=16)
    
    # Use first test sample for visualization
    sample_idx = 0
    av_sample = av_signal_test[sample_idx].cpu().numpy()
    true_angle_sample = target_angle_test[sample_idx].cpu().numpy()
    ring_pred_sample = ring_decoded[sample_idx].cpu().numpy()
    gru_pred_sample = gru_decoded[sample_idx].cpu().numpy()
    
    # Plot 1: Input
    axes[0].plot(av_sample, 'k-', linewidth=1.5)
    axes[0].set_ylabel('Angular Velocity (rad/step)')
    axes[0].set_title('Input Signal')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Predictions
    axes[1].plot(true_angle_sample, 'k--', linewidth=2, label='Ground Truth')
    axes[1].plot(ring_pred_sample, 'b-', alpha=0.8, label='Ring Attractor')
    axes[1].plot(gru_pred_sample, 'r-', alpha=0.8, label='GRU')
    axes[1].set_ylabel('Angle (rad)')
    axes[1].set_title('Integrated Angle')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Errors
    ring_error_sample = calculate_angular_error(
        torch.tensor(ring_pred_sample), torch.tensor(true_angle_sample)
    ).numpy()
    gru_error_sample = calculate_angular_error(
        torch.tensor(gru_pred_sample), torch.tensor(true_angle_sample)
    ).numpy()
    axes[2].plot(ring_error_sample, 'b-', alpha=0.8, label=f'Ring (mean: {ring_error_sample.mean():.3f})')
    axes[2].plot(gru_error_sample, 'r-', alpha=0.8, label=f'GRU (mean: {gru_error_sample.mean():.3f})')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Error (rad)')
    axes[2].set_title('Prediction Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_trajectory_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved sample trajectory to sample_trajectory_comparison.png")
    
    return ring_model, gru_model, ring_errors_all, gru_errors_all


if __name__ == '__main__':
    ring_model, gru_model, ring_errors, gru_errors = evaluate_trained_models()