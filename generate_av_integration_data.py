import torch
import numpy as np
from torch.utils.data import Dataset
import time

"""
This module contains the functions to generate (angular velocity) AV integration data 
for training and testing thering attractor model.
"""

def generate_av_integration_data_ultra_fast(num_trials, seq_len, zero_padding_start_ratio=0.1, zero_ratios_in_rest=None, max_av=0.1 * np.pi, device='cpu'):
    """
    Ultra-fast fully vectorized version with NO loops. Maximum parallel generation.
    """
    if zero_ratios_in_rest is None:
        zero_ratios_in_rest = [0.2, 0.5, 0.8]

    # Pre-allocate tensors
    av_signal = torch.zeros(num_trials, seq_len, device=device)
    
    num_initial_zeros = int(seq_len * zero_padding_start_ratio)
    len_rest = seq_len - num_initial_zeros

    # Generate all random data at once
    initial_angles = torch.rand(num_trials, device=device) * 2 * np.pi

    # Generate zero ratios for all trials
    zero_ratio_indices = torch.randint(0, len(zero_ratios_in_rest), (num_trials,), device=device)
    zero_ratios = torch.tensor(zero_ratios_in_rest, device=device)[zero_ratio_indices]
    
    # Calculate how many non-zeros each trial needs
    num_non_zeros_per_trial = len_rest - (len_rest * zero_ratios).long()
    
    # ULTRA-FAST VECTORIZED APPROACH: Generate all permutations at once
    if len_rest > 0:
        # Generate all possible AV values for all trials at once
        all_av_values = (torch.rand(num_trials, len_rest, device=device) - 0.5) * 2 * max_av
        
        # Create masks for which positions should have non-zero values
        # For each trial, randomly select positions for non-zero values
        random_positions = torch.rand(num_trials, len_rest, device=device).argsort(dim=1)
        
        # Create masks based on how many non-zeros each trial should have
        position_masks = random_positions < num_non_zeros_per_trial.unsqueeze(1)
        
        # Apply the masks to set values to zero where they should be zero
        rest_av_values = all_av_values * position_masks.float()
        
        # Place the rest values in the signal
        av_signal[:, num_initial_zeros:] = rest_av_values

    # Vectorized angle integration
    angle = torch.zeros(num_trials, seq_len, device=device)
    angle[:, 0] = initial_angles
    cumulative_av = torch.cumsum(av_signal[:, :-1], dim=1)
    angle[:, 1:] = (initial_angles.unsqueeze(1) + cumulative_av) % (2 * np.pi)

    return av_signal, angle


def generate_av_integration_data_chunked(num_trials, seq_len, zero_padding_start_ratio=0.1, 
                                       zero_ratios_in_rest=None, max_av=0.1 * np.pi, 
                                       device='cpu', chunk_size=10000):
    """
    Generate data in chunks to handle large datasets efficiently without memory issues.
    """
    if num_trials <= chunk_size:
        # If small enough, generate all at once
        return generate_av_integration_data_ultra_fast(
            num_trials, seq_len, zero_padding_start_ratio, 
            zero_ratios_in_rest, max_av, device
        )
    
    # Generate in chunks
    av_chunks = []
    angle_chunks = []
    
    print(f"Generating {num_trials} samples in chunks of {chunk_size}...")
    start_time = time.time()
    
    for i in range(0, num_trials, chunk_size):
        chunk_end = min(i + chunk_size, num_trials)
        current_chunk_size = chunk_end - i
        
        av_chunk, angle_chunk = generate_av_integration_data_ultra_fast(
            current_chunk_size, seq_len, zero_padding_start_ratio,
            zero_ratios_in_rest, max_av, device
        )
        
        av_chunks.append(av_chunk)
        angle_chunks.append(angle_chunk)
        
        if (i // chunk_size + 1) % 10 == 0:  # Progress every 10 chunks
            elapsed = time.time() - start_time
            progress = (i + current_chunk_size) / num_trials
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            print(f"  Progress: {progress*100:.1f}% ({i + current_chunk_size}/{num_trials}) - "
                  f"ETA: {remaining:.1f}s")
    
    # Concatenate all chunks
    av_signal = torch.cat(av_chunks, dim=0)
    angle = torch.cat(angle_chunks, dim=0)
    
    total_time = time.time() - start_time
    print(f"Chunked generation completed in {total_time:.2f} seconds")
    
    return av_signal, angle


class AVIntegrationDataset(Dataset):
    """
    PyTorch Dataset with pre-generated AV integration data for maximum performance.
    Uses chunked generation for efficient handling of large datasets.
    """
    
    def __init__(self, num_samples, seq_len, zero_padding_start_ratio=0.1, 
                 zero_ratios_in_rest=None, max_av=0.1 * np.pi, device='cpu', 
                 fast_mode=True, chunk_size=10000):
        """
        Args:
            num_samples (int): Total number of samples in the dataset
            seq_len (int): Length of each sequence
            zero_padding_start_ratio (float): Ratio of initial zeros in the AV signal
            zero_ratios_in_rest (list): Ratios of zeros in the rest of the sequence
            max_av (float): Maximum angular velocity
            device (str): Device to generate data on ('cpu' or 'cuda')
            fast_mode (bool): If True, use ultra-fast generation
            chunk_size (int): Size of chunks for generation (helps with memory)
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.zero_padding_start_ratio = zero_padding_start_ratio
        self.zero_ratios_in_rest = zero_ratios_in_rest if zero_ratios_in_rest is not None else [0.2, 0.5, 0.8]
        self.max_av = max_av
        self.device = torch.device(device)
        self.fast_mode = fast_mode
        self.chunk_size = chunk_size
        
        # Pre-compute some constants
        self.num_initial_zeros = int(seq_len * zero_padding_start_ratio)
        self.len_rest = seq_len - self.num_initial_zeros
        self.zero_ratios_tensor = torch.tensor(self.zero_ratios_in_rest, device=self.device)
        
        # PRE-GENERATE ALL DATA USING CHUNKED ULTRA-FAST METHOD
        # print(f"Pre-generating {num_samples} samples using ultra-fast chunked generation...")
        start_time = time.time()
        
        self.av_data, self.angle_data = generate_av_integration_data_chunked(
            num_samples, seq_len, zero_padding_start_ratio,
            zero_ratios_in_rest, max_av, device, chunk_size
        )
        
        generation_time = time.time() - start_time
        # print(f"Total data generation completed in {generation_time:.2f} seconds")
        # print(f"Generation speed: {num_samples/generation_time:.0f} samples/second")
        
        # For batch generation, keep track of a shuffled index to avoid always returning the same batches
        self.current_index = 0
        self.shuffled_indices = torch.randperm(num_samples, device=device)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return a single pre-computed sample."""
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        return self.av_data[idx], self.angle_data[idx]
    
    def generate_batch(self, batch_size):
        """
        Generate a batch by returning slices of pre-computed data.
        This is now extremely fast as it only involves tensor slicing.
        """
        if batch_size > self.num_samples:
            raise ValueError(f"Batch size {batch_size} cannot be larger than dataset size {self.num_samples}")
        
        # Get batch indices with wraparound and shuffling when needed
        if self.current_index + batch_size <= self.num_samples:
            # Simple case: batch fits within remaining data
            batch_indices = self.shuffled_indices[self.current_index:self.current_index + batch_size]
            self.current_index += batch_size
        else:
            # Wraparound case: need to combine end and beginning
            remaining = self.num_samples - self.current_index
            batch_indices = torch.cat([
                self.shuffled_indices[self.current_index:],  # Remaining from current shuffle
                self.shuffled_indices[:batch_size - remaining]  # Start from beginning
            ])
            self.current_index = batch_size - remaining
            
            # If we've used up most of the data, reshuffle for next epoch
            if self.current_index > self.num_samples * 0.8:
                self.shuffled_indices = torch.randperm(self.num_samples, device=self.device)
                self.current_index = 0
        
        # Return the batch by indexing pre-computed data
        return self.av_data[batch_indices], self.angle_data[batch_indices]
    
    def reset_batch_iterator(self):
        """Reset the batch iterator and reshuffle data."""
        self.current_index = 0
        self.shuffled_indices = torch.randperm(self.num_samples, device=self.device)
    
    def get_memory_usage_mb(self):
        """Return approximate memory usage in MB."""
        av_memory = self.av_data.numel() * self.av_data.element_size()
        angle_memory = self.angle_data.numel() * self.angle_data.element_size()
        total_bytes = av_memory + angle_memory
        return total_bytes / (1024 * 1024)