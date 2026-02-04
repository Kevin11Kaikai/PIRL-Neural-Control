"""
Wilson-Cowan Neural Mass Model

Implements the Wilson-Cowan equations for excitatory and inhibitory neural populations.
Parameterized to produce ~10Hz alpha oscillations.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
import os


class WilsonCowanODE(nn.Module):
    """
    Wilson-Cowan neural mass model with two populations (E and I).

    Dynamics:
        tau_e * dE/dt = -E + S(w_ee*E - w_ei*I + P + u)
        tau_i * dI/dt = -I + S(w_ie*E - w_ii*I + Q)

    where S(x) = 1/(1 + exp(-a*(x - theta))) is the sigmoid activation.

    Parameters are set to produce ~10Hz alpha oscillations.
    """

    def __init__(
        self,
        tau_e: float = 0.005,      # Excitatory time constant (5ms)
        tau_i: float = 0.010,      # Inhibitory time constant (10ms)
        w_ee: float = 16.0,        # E -> E connection weight
        w_ei: float = 12.0,        # I -> E connection weight
        w_ie: float = 15.0,        # E -> I connection weight
        w_ii: float = 3.0,         # I -> I connection weight
        a: float = 1.3,            # Sigmoid slope parameter
        theta: float = 4.0,        # Sigmoid threshold
        P: float = 1.25,           # External input to E
        Q: float = 0.0,            # External input to I
        device: str = 'cpu'
    ):
        """
        Initialize Wilson-Cowan model.

        Args:
            tau_e: Time constant for excitatory population
            tau_i: Time constant for inhibitory population
            w_ee: E->E synaptic weight
            w_ei: I->E synaptic weight
            w_ie: E->I synaptic weight
            w_ii: I->I synaptic weight
            a: Slope of sigmoid activation
            theta: Threshold of sigmoid activation
            P: External input to E population
            Q: External input to I population
            device: Device to run on ('cpu' or 'cuda')
        """
        super().__init__()

        self.tau_e = tau_e
        self.tau_i = tau_i
        self.w_ee = w_ee
        self.w_ei = w_ei
        self.w_ie = w_ie
        self.w_ii = w_ii
        self.a = a
        self.theta = theta
        self.P = P
        self.Q = Q
        self.device = device

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sigmoid activation function: S(x) = 1/(1 + exp(-a*(x - theta)))

        Args:
            x: Input tensor

        Returns:
            Activated output
        """
        return 1.0 / (1.0 + torch.exp(-self.a * (x - self.theta)))

    def forward(self, t: float, state: torch.Tensor, u: float = 0.0) -> torch.Tensor:
        """
        Compute derivatives dE/dt and dI/dt.

        Args:
            t: Current time (unused, but required by odeint)
            state: Current state [E, I] with shape [..., 2]
            u: Control input (external stimulus)

        Returns:
            Derivatives [dE/dt, dI/dt] with same shape as state
        """
        # Extract E and I populations
        E = state[..., 0]
        I = state[..., 1]

        # Compute inputs to each population
        input_E = self.w_ee * E - self.w_ei * I + self.P + u
        input_I = self.w_ie * E - self.w_ii * I + self.Q

        # Apply sigmoid activation
        S_E = self.sigmoid(input_E)
        S_I = self.sigmoid(input_I)

        # Compute derivatives
        dE_dt = (-E + S_E) / self.tau_e
        dI_dt = (-I + S_I) / self.tau_i

        # Stack derivatives
        derivatives = torch.stack([dE_dt, dI_dt], dim=-1)

        return derivatives

    def simulate(
        self,
        t_span: torch.Tensor,
        initial_state: torch.Tensor,
        u: float = 0.0,
        method: str = 'rk4'
    ) -> torch.Tensor:
        """
        Simulate the Wilson-Cowan system over a time span.

        Args:
            t_span: Time points to evaluate at, shape [n_steps]
            initial_state: Initial state [E0, I0], shape [..., 2]
            u: Control input (constant for now)
            method: ODE solver method ('euler', 'rk4', 'dopri5', etc.)

        Returns:
            Trajectory of states with shape [n_steps, ..., 2]
        """
        # Store control input
        self.current_u = u

        # Define ODE function with control input
        def ode_func(t, state):
            return self.forward(t, state, u=self.current_u)

        # Solve ODE
        solution = odeint(
            ode_func,
            initial_state,
            t_span,
            method=method
        )

        return solution


class WilsonCowanEnv(gym.Env):
    """
    Gymnasium environment for Wilson-Cowan neural mass model.

    Observation space: [E, I] (2D continuous)
    Action space: External input u (1D continuous, bounded)
    """

    def __init__(
        self,
        dt: float = 0.001,
        max_steps: int = 1000,
        device: str = 'cpu',
        action_limit: float = 2.0,
        target_state: Optional[np.ndarray] = None,
        reward_type: str = 'quadratic',
        **wc_params
    ):
        """
        Initialize environment.

        Args:
            dt: Time step size
            max_steps: Maximum steps per episode
            device: Device to run on ('cpu' or 'cuda')
            action_limit: Maximum absolute value of action
            target_state: Target state [E, I] for reward calculation
            reward_type: Type of reward ('quadratic', 'sparse', or 'none')
            **wc_params: Parameters for WilsonCowanODE
        """
        super().__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.device = device
        self.action_limit = action_limit
        self.reward_type = reward_type

        # Target state for reward calculation
        if target_state is None:
            self.target_state = np.array([0.5, 0.3])
        else:
            self.target_state = np.array(target_state)

        # Create Wilson-Cowan model
        self.model = WilsonCowanODE(device=device, **wc_params)

        # Define action and observation spaces (Gymnasium API)
        self.action_space = spaces.Box(
            low=-action_limit,
            high=action_limit,
            shape=(1,),
            dtype=np.float32
        )

        # Observation: [E, I] both in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # State
        self.current_state = None
        self.current_step = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (can contain 'initial_state')

        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)

        # Get initial state from options or use default
        if options is not None and 'initial_state' in options:
            initial_state = torch.tensor(
                options['initial_state'],
                device=self.device,
                dtype=torch.float32
            )
        else:
            # Default: small random perturbation around equilibrium
            initial_state = torch.tensor([0.1, 0.1], device=self.device, dtype=torch.float32)
            initial_state += 0.01 * torch.randn(2, device=self.device)

        self.current_state = initial_state
        self.current_step = 0

        obs = self.current_state.cpu().numpy()
        info = {
            'step': self.current_step,
            'time': 0.0,
            'state': obs.copy()
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take one step in the environment.

        Args:
            action: Control input (1D array or scalar)

        Returns:
            observation: Next observation
            reward: Reward signal
            terminated: Whether episode naturally ended
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        # Convert action to scalar
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)

        # Clip action to valid range
        action = np.clip(action, -self.action_limit, self.action_limit)

        # Simulate one step
        t_span = torch.tensor([0.0, self.dt], device=self.device)
        trajectory = self.model.simulate(
            t_span,
            self.current_state,
            u=action
        )

        # Update state
        self.current_state = trajectory[-1]
        self.current_step += 1

        # Get observation
        obs = self.current_state.cpu().numpy()

        # Compute reward
        reward = self._compute_reward(obs, action)

        # Check termination conditions
        terminated = False  # Natural termination (not used for now)
        truncated = self.current_step >= self.max_steps

        # Info dict
        info = {
            'step': self.current_step,
            'time': self.current_step * self.dt,
            'state': obs.copy(),
            'action': action
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, state: np.ndarray, action: float) -> float:
        """
        Compute reward based on current state and action.

        Args:
            state: Current state [E, I]
            action: Current action

        Returns:
            reward: Scalar reward
        """
        if self.reward_type == 'none':
            return 0.0

        elif self.reward_type == 'quadratic':
            # Negative quadratic distance to target
            state_error = state - self.target_state
            state_cost = np.sum(state_error ** 2)
            action_cost = 0.01 * (action ** 2)  # Small action penalty
            reward = -(state_cost + action_cost)
            return float(reward)

        elif self.reward_type == 'sparse':
            # Sparse reward when close to target
            state_error = np.linalg.norm(state - self.target_state)
            threshold = 0.1
            reward = 1.0 if state_error < threshold else 0.0
            return float(reward)

        else:
            return 0.0

    def render(self):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Close the environment."""
        pass


def test_wilson_cowan():
    """
    Test the Wilson-Cowan model by simulating 1 second and verifying
    that oscillations are in the 8-15 Hz range.
    """
    print("Testing Wilson-Cowan Model...")
    print("=" * 60)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model with default parameters (should produce ~10Hz oscillations)
    model = WilsonCowanODE(device=device)

    # Print parameters
    print("\nModel Parameters:")
    print(f"  tau_e = {model.tau_e}, tau_i = {model.tau_i}")
    print(f"  w_ee = {model.w_ee}, w_ei = {model.w_ei}")
    print(f"  w_ie = {model.w_ie}, w_ii = {model.w_ii}")
    print(f"  a = {model.a}, theta = {model.theta}")
    print(f"  P = {model.P}, Q = {model.Q}")

    # Simulation parameters
    dt = 0.001  # 1ms time step
    T = 1.0     # 1 second simulation
    t_span = torch.arange(0, T, dt, device=device)

    # Initial state (small perturbation)
    initial_state = torch.tensor([0.1, 0.1], device=device)

    print(f"\nSimulation:")
    print(f"  Duration: {T} seconds")
    print(f"  Time step: {dt} seconds")
    print(f"  Total steps: {len(t_span)}")
    print(f"  Initial state: E={initial_state[0]:.3f}, I={initial_state[1]:.3f}")

    # Simulate
    print("\nRunning simulation...")
    trajectory = model.simulate(t_span, initial_state, u=0.0, method='rk4')

    # Convert to numpy for analysis
    t_np = t_span.cpu().numpy()
    E_np = trajectory[:, 0].cpu().numpy()
    I_np = trajectory[:, 1].cpu().numpy()

    print("Simulation complete!")

    # Analyze oscillations using FFT
    print("\nAnalyzing oscillations...")

    # Remove transient (first 200ms)
    transient_idx = int(0.2 / dt)
    E_steady = E_np[transient_idx:]
    t_steady = t_np[transient_idx:]

    # Compute FFT
    from scipy import signal
    freqs, psd = signal.welch(
        E_steady - E_steady.mean(),
        fs=1/dt,
        nperseg=min(512, len(E_steady)//4)
    )

    # Find peak frequency in 1-50 Hz range
    freq_mask = (freqs >= 1) & (freqs <= 50)
    peak_idx = np.argmax(psd[freq_mask])
    peak_freq = freqs[freq_mask][peak_idx]

    print(f"  Peak frequency: {peak_freq:.2f} Hz")
    print(f"  E range: [{E_np.min():.3f}, {E_np.max():.3f}]")
    print(f"  I range: [{I_np.min():.3f}, {I_np.max():.3f}]")

    # Check if frequency is in alpha range (8-15 Hz)
    is_alpha = 8.0 <= peak_freq <= 15.0
    status = "PASS" if is_alpha else "FAIL"
    print(f"\n[{status}] Frequency in alpha range (8-15 Hz): {is_alpha}")

    # Create visualization
    print("\nGenerating visualization...")

    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Time series
    axes[0].plot(t_np, E_np, 'b-', label='E (Excitatory)', linewidth=1.5)
    axes[0].plot(t_np, I_np, 'r-', label='I (Inhibitory)', linewidth=1.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Activity')
    axes[0].set_title('Wilson-Cowan Neural Dynamics')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Phase portrait
    axes[1].plot(E_np, I_np, 'k-', alpha=0.5, linewidth=0.5)
    axes[1].plot(E_np[0], I_np[0], 'go', markersize=10, label='Start')
    axes[1].plot(E_np[-1], I_np[-1], 'ro', markersize=10, label='End')
    axes[1].set_xlabel('E (Excitatory)')
    axes[1].set_ylabel('I (Inhibitory)')
    axes[1].set_title('Phase Portrait')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Power spectrum
    axes[2].semilogy(freqs, psd, 'b-', linewidth=1.5)
    axes[2].axvline(peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq:.2f} Hz')
    axes[2].axvspan(8, 15, alpha=0.2, color='green', label='Alpha range (8-15 Hz)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Power Spectral Density')
    axes[2].set_title('Frequency Analysis')
    axes[2].set_xlim([0, 30])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = 'figures/wc_test.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Peak oscillation frequency: {peak_freq:.2f} Hz")
    print(f"  Alpha range (8-15 Hz): {'PASS' if is_alpha else 'FAIL'}")
    print(f"  Visualization saved: {save_path}")
    print("=" * 60)

    return peak_freq, is_alpha


if __name__ == "__main__":
    test_wilson_cowan()
