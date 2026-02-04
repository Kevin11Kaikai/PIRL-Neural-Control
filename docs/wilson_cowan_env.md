# Wilson-Cowan Environment Documentation

## Overview

The `WilsonCowanEnv` class provides a Gymnasium-compatible reinforcement learning environment for the Wilson-Cowan neural mass model. This environment simulates the dynamics of coupled excitatory (E) and inhibitory (I) neural populations.

## Installation

Ensure you have the required dependencies:
```bash
pip install gymnasium torch torchdiffeq numpy matplotlib
```

## Quick Start

```python
from envs import WilsonCowanEnv

# Create environment
env = WilsonCowanEnv(
    dt=0.001,                    # Time step (seconds)
    max_steps=1000,              # Max steps per episode
    device='cuda',               # 'cpu' or 'cuda'
    action_limit=2.0,            # Action bounds
    target_state=[0.5, 0.3],     # Target [E, I]
    reward_type='quadratic'      # Reward function type
)

# Reset environment
obs, info = env.reset(seed=42)

# Run episode
for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Environment Specifications

### Observation Space
- **Type**: `Box(0.0, 1.0, shape=(2,))`
- **Description**: Neural population activities [E, I]
  - `E`: Excitatory population activity (0 to 1)
  - `I`: Inhibitory population activity (0 to 1)

### Action Space
- **Type**: `Box(-action_limit, action_limit, shape=(1,))`
- **Description**: External control input to the excitatory population
- **Default bounds**: [-2.0, 2.0]

### Reward Function

Three reward types are available:

1. **Quadratic** (default):
   ```
   reward = -(||state - target||² + 0.01 * action²)
   ```
   Encourages reaching the target state with minimal control effort.

2. **Sparse**:
   ```
   reward = 1.0 if ||state - target|| < 0.1 else 0.0
   ```
   Binary reward when close to target.

3. **None**:
   ```
   reward = 0.0
   ```
   No reward signal (useful for exploration).

### Episode Termination

- **Terminated**: Not currently used (always `False`)
- **Truncated**: Episode ends after `max_steps` steps

## Wilson-Cowan Dynamics

The environment simulates the following differential equations:

```
τₑ dE/dt = -E + S(wₑₑE - wₑᵢI + P + u)
τᵢ dI/dt = -I + S(wᵢₑE - wᵢᵢI + Q)
```

where:
- `S(x) = 1/(1 + exp(-a(x - θ)))` is the sigmoid activation
- `u` is the control input (action)

### Default Parameters

The default parameters produce ~10Hz alpha oscillations:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_e` | 0.005 | Excitatory time constant (5ms) |
| `tau_i` | 0.010 | Inhibitory time constant (10ms) |
| `w_ee` | 16.0 | E → E connection weight |
| `w_ei` | 12.0 | I → E connection weight |
| `w_ie` | 15.0 | E → I connection weight |
| `w_ii` | 3.0 | I → I connection weight |
| `a` | 1.3 | Sigmoid slope |
| `theta` | 4.0 | Sigmoid threshold |
| `P` | 1.25 | External input to E |
| `Q` | 0.0 | External input to I |

## Customization

### Custom Initial State

```python
obs, info = env.reset(
    seed=42,
    options={'initial_state': [0.2, 0.3]}
)
```

### Custom Wilson-Cowan Parameters

```python
env = WilsonCowanEnv(
    dt=0.001,
    max_steps=1000,
    tau_e=0.008,      # Custom time constant
    w_ee=18.0,        # Custom connection weight
    target_state=[0.6, 0.4]
)
```

## Examples

### Example 1: Random Policy

```python
env = WilsonCowanEnv()
obs, _ = env.reset()

total_reward = 0
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Total reward: {total_reward:.4f}")
print(f"Final state: E={obs[0]:.4f}, I={obs[1]:.4f}")
```

### Example 2: Zero Action (Uncontrolled)

```python
env = WilsonCowanEnv()
obs, _ = env.reset()

for _ in range(1000):
    action = np.array([0.0])  # No control
    obs, reward, terminated, truncated, info = env.step(action)
```

### Example 3: Custom Reward Function

Modify `_compute_reward()` method for custom reward designs.

## Visualization

See `examples/test_wc_environment.py` for a complete example with visualization including:
- State trajectories over time
- Phase portrait (E vs I)
- Action sequence
- Reward progression

## Testing

Run the environment test:
```bash
python examples/test_wc_environment.py
```

Expected output:
- Episode completes successfully
- Total reward is negative (for random policy with quadratic reward)
- Visualization shows oscillatory dynamics

## GPU Acceleration

The environment supports GPU acceleration via PyTorch:

```python
env = WilsonCowanEnv(device='cuda')  # Use GPU
```

CUDA is automatically used if available. Check with:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Performance

- **Time step**: 0.001s (1ms) recommended
- **Episode length**: 1000 steps = 1 second simulation
- **Speed**: ~1000 steps in <1 second on GPU (RTX 4080)

## References

- Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. Biophysical Journal, 12(1), 1-24.
- Breakspear, M. (2017). Dynamic models of large-scale brain activity. Nature Neuroscience, 20(3), 340-352.

## API Compatibility

Fully compatible with:
- Gymnasium (>=1.0.0)
- Stable-Baselines3
- RLlib
- Any RL framework supporting Gymnasium API
