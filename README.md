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
Ref is Pytorch version, Impl is CUDA implementation, check assets/REPORT.md

