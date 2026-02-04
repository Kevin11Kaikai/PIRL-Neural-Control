# PIRL: Physics-Informed Reinforcement Learning

[![GitHub](https://img.shields.io/badge/GitHub-PIRL--Neural--Control-blue?logo=github)](https://github.com/Kevin11Kaikai/PIRL-Neural-Control)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Physics-Informed Reinforcement Learning framework for neural dynamics control and optimization.

**🎯 Complete framework with rigorous baseline comparison and comprehensive analysis**

## 📚 Quick Links

- **[Executive Summary](./EXECUTIVE_SUMMARY.md)** - Quick overview and key findings
- **[Final Experiment Analysis](./FINAL_EXPERIMENT_ANALYSIS.md)** - Comprehensive results and failure analysis
- **[Project Completion Report](./PROJECT_COMPLETION_REPORT.md)** - Complete project documentation
- **[Academic Framing Guide](./ACADEMIC_FRAMING_GUIDE.md)** - How to present this work in research

## Project Overview

PIRL combines physics-informed neural networks (PINNs) with reinforcement learning to learn control policies for dynamical systems while respecting underlying physical constraints. This project focuses on neural dynamics systems, starting with the Wilson-Cowan model.

### Key Results

- ✅ **Exceptional World Model**: MSE 0.00002 (500× better than target)
- ✅ **Complete Framework**: ~3,200 lines of tested code
- ✅ **Rigorous Comparison**: 4 baseline controllers with statistical testing
- ⚠️ **RL Agent**: Policy collapse identified (detailed analysis provided)

## Features

- **Physics-Informed RL**: Integrates physical laws and constraints into the RL training process
- **Neural Dynamics Environments**: Simulation environments for various neural models
  - Wilson-Cowan neural population model
- **Custom RL Agents**: Agents designed to learn physics-consistent policies
- **Modular Architecture**: Easy to extend with new models and algorithms

## Directory Structure

```
PIRL_claude/
├── src/
│   ├── envs/          # Simulation environments
│   ├── models/        # Neural network models (PINNs, policy networks)
│   ├── agents/        # RL agents
│   └── utils/         # Utility functions
├── configs/           # Configuration files
├── data/             # Training data and results
├── logs/             # Training logs
├── experiments/      # Experiment scripts
├── tests/            # Unit tests
└── requirements.txt  # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kevin11Kaikai/PIRL-Neural-Control.git
cd PIRL-Neural-Control
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from envs import WilsonCowanEnv

# Create environment
env = WilsonCowanEnv(
    dt=0.001,
    max_steps=1000,
    device='cuda',
    target_state=[0.5, 0.3],
    reward_type='quadratic'
)

# Run episode
obs, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Final state: E={obs[0]:.4f}, I={obs[1]:.4f}")
```

See `examples/test_wc_environment.py` for a complete example with visualization.

## Environments

### Wilson-Cowan Model
The Wilson-Cowan model describes the dynamics of excitatory and inhibitory neural populations:
- State: [E, I] (excitatory and inhibitory activity)
- Dynamics: Coupled differential equations with sigmoid activation
- Control: External inputs to modulate population activity

## Development Roadmap

**Phase 1: Environment Setup** ✓ COMPLETED
- [x] Project initialization
- [x] Conda environment setup (Python 3.10, PyTorch 2.7.1, CUDA 11.8)
- [x] Wilson-Cowan ODE implementation with torchdiffeq
- [x] Gymnasium environment wrapper
- [x] Environment testing and validation (10Hz alpha oscillations confirmed)

**Phase 2: Physics-Informed Learning** ✓ COMPLETED
- [x] PIRL world model with residual learning
- [x] Physics-informed loss functions (prediction + physics)
- [x] Integration with Wilson-Cowan dynamics
- [x] Neural CDE state observer (from partial observations)
- [x] Training and testing pipeline

**Phase 3: RL Agent** ✓ COMPLETED
- [x] PhIHP agent implementation (TD3-based)
- [x] Actor-Critic networks with safety layer
- [x] World model imagination rollouts
- [x] Mixed real+imagine training
- [x] Multi-objective reward design

**Phase 4: Evaluation**
- [ ] Training pipeline
- [ ] Evaluation metrics
- [ ] Visualization tools
- [ ] Benchmarking against baselines

## Current Status

**Environment**: Ready for RL training ✓
- Wilson-Cowan dynamics producing stable 10Hz oscillations
- Full Gymnasium API compatibility
- GPU acceleration on RTX 4080 SUPER
- Comprehensive testing completed (see `TEST_RESULTS.md`)

**PIRL World Model**: Fully implemented ✓
- Residual learning architecture combining physics + data
- Dual loss function (prediction + physics constraint)
- Achieves MSE < 0.01 (target met)
- Small-weight initialization for physics-informed priors
- See `docs/pirl_world_model.md` for details

**Neural CDE Observer**: Implemented ✓
- State reconstruction from partial observations (E only → [E, I])
- Continuous-time modeling with controlled differential equations
- Physics-informed constraints
- Cubic spline interpolation for irregular sampling
- See `docs/neural_cde_observer.md` and `NEURAL_CDE_TEST_RESULTS.md`

**PhIHP Agent**: Fully implemented ✓
- Physics-Informed Hierarchical Planning RL agent
- TD3-based Actor-Critic with twin Q-networks
- World model imagination rollouts (5 steps)
- Safety constraint layer (absolute, rate, state-dependent)
- Multi-objective reward (task + energy + oscillation + safety)
- Mixed real+imagine training
- See `docs/phihp_agent.md` and `PHIHP_AGENT_RESULTS.md`

**Documentation**: Complete ✓
- Environment API: `docs/wilson_cowan_env.md`
- World model: `docs/pirl_world_model.md`
- Neural CDE observer: `docs/neural_cde_observer.md`
- PhIHP agent: `docs/phihp_agent.md`
- Example code: `examples/test_wc_environment.py`, `examples/test_pirl_world_model.py`, `examples/test_neural_cde_observer.py`, `examples/test_phihp_agent.py`
- Test results: `TEST_RESULTS.md`, `PIRL_TEST_RESULTS.md`, `NEURAL_CDE_TEST_RESULTS.md`, `PHIHP_AGENT_RESULTS.md`

## License

[Add license information]

## Citation

[Add citation information if applicable]
