# Ring Attractor Neural Network for Angular Velocity Integration

This repository implements a biologically-inspired Ring Attractor Neural Network that performs angular velocity integration, mimicking how certain brain circuits (like head direction cells) track angular position.

## Weight Evolution During Training

Watch how the network learns three essential circuits:
- **W₀ (Self-Maintaining)**: Keeps the activity bump stable when the animal is stationary, maintaining the current heading direction without any sensory or motor input
- **W_L (Left Turn)**: Activated by left turn motor signals, rotates the bump counterclockwise to track leftward movement
- **W_R (Right Turn)**: Activated by right turn motor signals, rotates the bump clockwise to track rightward movement

![Weight Evolution](assets/ring_attractor_weights_evolution.gif)

## Overview

The Ring Attractor Network maintains a localized "bump" of neural activity on a ring of neurons, where the position of this bump represents an angular value (0-2π). By integrating angular velocity inputs over time, the network learns to rotate this bump to track the true angular position.

## Architecture

### Neural Dynamics

The network follows the leaky integrator dynamics:

```
r += (-r + F(J0·r + J1*Wo·r + L*Wl·r + R*Wr·r)) * dt/tau
```

Where:
- `r`: Neural activity vector (ring of neurons)
- `J0`: Global inhibition matrix
- `Wo`: Maintenance weights (preserves bump shape)
- `Wl`, `Wr`: Left/right rotation weight matrices
- `L`, `R`: Rotation signals derived from angular velocity
- `F()`: Activation function (tanh, ReLU, or GELU)

### Key Components

1. **Ring Topology**: Neurons arranged in a circle, each with a preferred direction (0-2π)
2. **Activity Bump**: Localized peak of activity representing current angle
3. **Rotation Mechanism**: Asymmetric connections shift the bump based on velocity
4. **Gain Networks**: Small MLPs that dynamically modulate rotation strength

## Files

- `generate_av_integration_data.py`: Efficient data generation for training/testing
  - Creates synthetic angular velocity signals
  - Integrates velocities to compute ground truth angles
  - Supports batched, vectorized generation on GPU

- `train_ring_attractor.py`: Main model and training script
  - `LeakyRingAttractor`: Core neural network model
  - Training loop with cosine similarity loss
  - Evaluation and visualization functions

## Usage

### Training

```python
python train_ring_attractor.py
```

This will:
1. Generate training data on-the-fly (angular velocity and targeted integrated angle)
2. Train the ring attractor network (take velocity input, and output estimated angle)
3. Visualize weight matrices (begining and end) and performance

or 

```python
python train_ring_attractor_save_movie.py
```

Additionally, this will:
Save the plot of weights of each 10 training steps (epoch) and compile them into a weight evolution movie (requires ffmpeg)

### Data Generation

```python
from generate_av_integration_data import AVIntegrationDataset

# Create dataset
dataset = AVIntegrationDataset(
    num_samples=10000,
    seq_len=120,
    zero_padding_start_ratio=0.1,
    zero_ratios_in_rest=[0.2, 0.5, 0.8],
    max_av=0.1 * np.pi,
    device='cuda'
)

# Generate batch
av_signals, angles = dataset.generate_batch(batch_size=128)
```

## Model Configurations

### Initialization Modes

1. **Random**: Random weight initialization (default for training)
2. **Perfect**: Ideal ring attractor weights (also can learn from there)
3. **Debug Perfect**: Fixed ideal weights with no learning (for testing)

### Parameters

- `num_neurons`: Number of neurons in the ring (default: 120)
- `tau`: Time constant for neural dynamics
- `dt`: Integration time step
- `alpha`: usually tau/dt, but can be set internally by user to decide how smooth neural dynamics are: r = r * (1 - alpha) + recurrent_input * alpha
- `activation`: Activation function ('tanh', 'relu', 'gelu')
- `hidden_gain_neurons`: Hidden units in gain modulation networks

## Training Details

### Loss Functions

1. **Head Direction Loss**: Measures angle prediction loss against ground-truth
2. **Bump Amplitude Loss**: Ensures stable total activity (prevents vanishing/exploding)

### Angle Decoding

Two methods to extract angle from neural activity:
- **Population Vector**: Weighted circular average of neuron activities
- **Argmax**: Location of peak activity

## Outputs

The training script generates:
- Weight matrix visualizations (initial and trained)
- A movie of how three weight matrix evolves (weights for bump maintaining, turn left, turn right)
- Performance plots showing:
  - Network activity heatmap
  - Bump activity profiles
  - Input angular velocity
  - True vs. decoded angles

## Biological Inspiration

This model is inspired by:
- Ring attractor dynamics in fruit fly navigation circuits
- Head direction cells in the mammalian brain


## CUDA kernel:

1. Run:

```bash
pip install ninja
python fwd_fp32.py
```

2. Benchmark results
Ref is Pytorch version, Impl is CUDA implementation:

# Ring RNN Performance Benchmark Results (RTX PRO 6000 Blackwell)

## Varying Number of Neurons

| Batch Size | Num Neurons | Action Dim | Seq Len | Ref Latency (ms) | Impl Latency (ms) | Speedup |
|------------|-------------|------------|---------|------------------|-------------------|---------|
| 256 | 128 | 3 | 20 | 2.674 ± 0.013 | 0.584 ± 0.017 | 4.58× |
| 256 | 256 | 3 | 20 | 9.328 ± 0.018 | 1.086 ± 0.011 | 8.59× |
| 256 | 384 | 3 | 20 | 21.621 ± 0.021 | 1.648 ± 0.012 | 13.12× |
| 256 | 512 | 3 | 20 | 38.163 ± 0.037 | 2.229 ± 0.013 | 17.12× |
| 256 | 640 | 3 | 20 | 58.070 ± 0.017 | 3.024 ± 0.014 | 19.20× |
| 256 | 768 | 3 | 20 | 83.209 ± 0.054 | 4.081 ± 0.014 | 20.39× |
| 256 | 896 | 3 | 20 | 112.040 ± 0.018 | 4.861 ± 0.020 | 23.05× |
| 256 | 1024 | 3 | 20 | 145.380 ± 0.017 | 5.859 ± 0.020 | 24.81× |

## Varying Sequence Length

| Batch Size | Num Neurons | Action Dim | Seq Len | Ref Latency (ms) | Impl Latency (ms) | Speedup |
|------------|-------------|------------|---------|------------------|-------------------|---------|
| 256 | 256 | 3 | 4 | 1.979 ± 0.006 | 0.241 ± 0.019 | 8.21× |
| 256 | 256 | 3 | 8 | 3.830 ± 0.008 | 0.462 ± 0.006 | 8.29× |
| 256 | 256 | 3 | 16 | 7.527 ± 0.010 | 0.900 ± 0.007 | 8.36× |
| 256 | 256 | 3 | 20 | 9.353 ± 0.020 | 1.109 ± 0.011 | 8.43× |
| 256 | 256 | 3 | 32 | 14.787 ± 0.018 | 1.759 ± 0.014 | 8.41× |
| 256 | 256 | 3 | 128 | 58.563 ± 0.062 | 6.702 ± 0.099 | 8.74× |
| 256 | 256 | 3 | 256 | 116.795 ± 0.037 | 13.365 ± 0.179 | 8.74× |
| 256 | 256 | 3 | 512 | 234.564 ± 0.061 | 27.212 ± 0.320 | 8.62× |
| 256 | 256 | 3 | 1024 | 462.317 ± 0.086 | 50.065 ± 0.998 | 9.23× |
| 256 | 256 | 3 | 2048 | 927.352 ± 0.108 | 98.850 ± 0.283 | 9.38× |

## Varying Batch Size

| Batch Size | Num Neurons | Action Dim | Seq Len | Ref Latency (ms) | Impl Latency (ms) | Speedup |
|------------|-------------|------------|---------|------------------|-------------------|---------|
| 32 | 256 | 3 | 20 | 2.711 ± 0.015 | 0.821 ± 0.013 | 3.30× |
| 128 | 256 | 3 | 20 | 3.443 ± 0.014 | 0.898 ± 0.004 | 3.83× |
| 256 | 256 | 3 | 20 | 9.194 ± 0.017 | 1.064 ± 0.009 | 8.64× |
| 512 | 256 | 3 | 20 | 19.330 ± 0.019 | 1.824 ± 0.022 | 10.60× |
| 1024 | 256 | 3 | 20 | 38.333 ± 0.023 | 3.191 ± 0.013 | 12.01× |
| 2048 | 256 | 3 | 20 | 75.043 ± 0.026 | 5.646 ± 0.020 | 13.29× |

## Varying Action Dimension

| Batch Size | Num Neurons | Action Dim | Seq Len | Ref Latency (ms) | Impl Latency (ms) | Speedup |
|------------|-------------|------------|---------|------------------|-------------------|---------|
| 256 | 256 | 2 | 20 | 6.826 ± 0.013 | 1.032 ± 0.006 | 6.61× |
| 256 | 256 | 3 | 20 | 9.185 ± 0.014 | 1.089 ± 0.016 | 8.43× |
| 256 | 256 | 4 | 20 | 11.104 ± 0.013 | 1.104 ± 0.011 | 10.06× |
| 256 | 256 | 8 | 20 | 18.224 ± 0.021 | 1.166 ± 0.016 | 15.63× |
| 256 | 256 | 32 | 20 | 60.898 ± 0.021 | 1.524 ± 0.018 | 39.95× |
| 256 | 256 | 128 | 20 | 232.754 ± 0.022 | 2.975 ± 0.014 | 78.24× |
| 256 | 256 | 256 | 20 | 695.566 ± 0.089 | 4.915 ± 0.029 | 141.54× |
| 256 | 256 | 512 | 20 | 1390.230 ± 0.068 | 8.950 ± 0.056 | 155.36× |
| 256 | 256 | 1024 | 20 | 2775.448 ± 0.084 | 17.396 ± 0.073 | 159.54× |

# Ring RNN Performance Benchmark Results (RTX 5090)

## Varying Number of Neurons

| Batch Size | Num Neurons | Action Dim | Seq Len | Ref Latency (ms) | Impl Latency (ms) | Speedup |
|------------|-------------|------------|---------|------------------|-------------------|---------|
| 256 | 128 | 3 | 20 | 4.433 ± 0.209 | 0.608 ± 0.007 | 7.29× |
| 256 | 256 | 3 | 20 | 7.865 ± 0.014 | 1.271 ± 0.007 | 6.19× |
| 256 | 384 | 3 | 20 | 19.170 ± 0.014 | 1.887 ± 0.005 | 10.16× |
| 256 | 512 | 3 | 20 | 34.560 ± 0.121 | 2.442 ± 0.017 | 14.15× |
| 256 | 640 | 3 | 20 | 53.179 ± 0.097 | 3.618 ± 0.007 | 14.70× |
| 256 | 768 | 3 | 20 | 76.531 ± 0.047 | 4.924 ± 0.012 | 15.54× |
| 256 | 896 | 3 | 20 | 104.275 ± 0.037 | 6.008 ± 0.022 | 17.36× |
| 256 | 1024 | 3 | 20 | 134.096 ± 0.120 | 6.926 ± 0.017 | 19.36× |

## Varying Sequence Length

| Batch Size | Num Neurons | Action Dim | Seq Len | Ref Latency (ms) | Impl Latency (ms) | Speedup |
|------------|-------------|------------|---------|------------------|-------------------|---------|
| 256 | 256 | 3 | 4 | 1.675 ± 0.006 | 0.281 ± 0.021 | 5.96× |
| 256 | 256 | 3 | 8 | 3.203 ± 0.010 | 0.530 ± 0.026 | 6.04× |
| 256 | 256 | 3 | 16 | 6.314 ± 0.026 | 1.011 ± 0.007 | 6.25× |
| 256 | 256 | 3 | 20 | 7.907 ± 0.028 | 1.271 ± 0.007 | 6.22× |
| 256 | 256 | 3 | 32 | 12.521 ± 0.024 | 1.974 ± 0.021 | 6.34× |
| 256 | 256 | 3 | 128 | 50.217 ± 0.101 | 7.964 ± 0.039 | 6.31× |
| 256 | 256 | 3 | 256 | 100.391 ± 0.080 | 16.384 ± 0.036 | 6.13× |
| 256 | 256 | 3 | 512 | 201.993 ± 0.118 | 32.835 ± 0.023 | 6.15× |
| 256 | 256 | 3 | 1024 | 403.139 ± 0.355 | 65.626 ± 0.043 | 6.14× |
| 256 | 256 | 3 | 2048 | 810.211 ± 0.211 | 131.107 ± 0.152 | 6.18× |

## Varying Batch Size

| Batch Size | Num Neurons | Action Dim | Seq Len | Ref Latency (ms) | Impl Latency (ms) | Speedup |
|------------|-------------|------------|---------|------------------|-------------------|---------|
| 32 | 256 | 3 | 20 | 4.643 ± 0.406 | 0.685 ± 0.004 | 6.78× |
| 128 | 256 | 3 | 20 | 5.715 ± 0.061 | 0.777 ± 0.005 | 7.36× |
| 256 | 256 | 3 | 20 | 8.558 ± 0.282 | 1.293 ± 0.009 | 6.62× |
| 512 | 256 | 3 | 20 | 17.588 ± 0.884 | 1.848 ± 0.010 | 9.52× |
| 1024 | 256 | 3 | 20 | 34.463 ± 0.073 | 3.057 ± 0.011 | 11.27× |
| 2048 | 256 | 3 | 20 | 68.181 ± 0.028 | 5.770 ± 0.017 | 11.82× |

## Varying Action Dimension

| Batch Size | Num Neurons | Action Dim | Seq Len | Ref Latency (ms) | Impl Latency (ms) | Speedup |
|------------|-------------|------------|---------|------------------|-------------------|---------|
| 256 | 256 | 2 | 20 | 6.304 ± 0.074 | 1.272 ± 0.005 | 4.96× |
| 256 | 256 | 3 | 20 | 7.932 ± 0.073 | 1.273 ± 0.009 | 6.23× |
| 256 | 256 | 4 | 20 | 10.214 ± 1.401 | 1.299 ± 0.007 | 7.86× |
| 256 | 256 | 8 | 20 | 16.260 ± 0.024 | 1.343 ± 0.007 | 12.11× |
| 256 | 256 | 32 | 20 | 55.584 ± 0.085 | 1.753 ± 0.006 | 31.71× |
| 256 | 256 | 128 | 20 | 269.584 ± 0.166 | 3.322 ± 0.014 | 81.16× |
| 256 | 256 | 256 | 20 | 656.494 ± 0.185 | 5.441 ± 0.014 | 120.64× |
| 256 | 256 | 512 | 20 | Out of Memory | Out of Memory | N/A |
| 256 | 256 | 1024 | 20 | Out of Memory | Out of Memory | N/A |

**Note:** For action dimensions 512 and 1024, the PyTorch reference implementation ran out of memory (attempted to allocate 32.00 GiB on a 31.37 GiB GPU). The CUDA implementation would likely succeed for these configurations, but comparison could not be performed.