"""
Wilson-Cowan Frequency Control Environment

Frequency-based closed-loop control environment for Wilson-Cowan neural oscillations.

Key differences from WilsonCowanEnv:
- Action: stimulation frequency f_stim ∈ [4, 15] Hz (instead of amplitude)
- Observation: [E(t), f_hat(t)] (includes frequency estimate)
- Reward: frequency tracking error (instead of state error)
- Stimulation: periodic I_stim = A * sin(2π * f_stim * t)
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from src.utils.frequency_estimator import FrequencyEstimator


class WilsonCowanFreqEnv(gym.Env):
    """
    Wilson-Cowan neural dynamics environment with frequency control.

    The agent controls the stimulation frequency to drive the system to
    oscillate at a target frequency.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        f_target: float = 9.0,
        f_natural: float = 10.0,
        stim_amplitude: float = 0.5,
        w_ee_drift: bool = False,
        w_ee_drift_rate: float = 0.0,
        dt: float = 0.001,
        episode_length: float = 5.0,
        device: str = "cpu",
    ):
        """
        Initialize frequency control environment.

        Parameters:
            f_target: Target oscillation frequency (Hz)
            f_natural: Natural frequency of the system (Hz)
            stim_amplitude: Amplitude of periodic stimulation
            w_ee_drift: Whether w_ee drifts over time (for experiment b)
            w_ee_drift_rate: Rate of w_ee drift (per second)
            dt: Integration time step (s)
            episode_length: Episode duration (s)
            device: 'cpu' or 'cuda'
        """
        super().__init__()

        # Action space: stimulation frequency [4, 15] Hz
        self.action_space = spaces.Box(
            low=np.array([4.0]), high=np.array([15.0]), dtype=np.float32
        )

        # Observation space: [E, f_hat]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]), high=np.array([1.0, 20.0]), dtype=np.float32
        )

        self.f_target = f_target
        self.f_natural = f_natural
        self.stim_amplitude = stim_amplitude
        self.w_ee_drift = w_ee_drift
        self.w_ee_drift_rate = w_ee_drift_rate
        self.dt = dt
        self.episode_length = episode_length
        self.device = torch.device(device)

        self.max_steps = int(episode_length / dt)

        # Frequency estimator
        self.freq_estimator = FrequencyEstimator(window_size=500, dt=dt)

        # Wilson-Cowan parameters (adjusted for f_natural)
        self.params = self._compute_params_for_frequency(f_natural)
        self.w_ee_initial = self.params["w_ee"]

        # State variables
        self.state = None
        self.t = 0.0
        self.steps = 0
        self.prev_f_stim = f_target
        self.w_ee_current = self.w_ee_initial

    def _compute_params_for_frequency(self, f_natural: float) -> Dict[str, float]:
        """
        Compute Wilson-Cowan parameters to achieve target natural frequency.

        For Wilson-Cowan model, the oscillation frequency depends primarily on
        w_ee and tau_e. We adjust w_ee to match f_natural.

        Approximate relationship: f ≈ 2.5 * w_ee (empirically derived)
        """
        w_ee = f_natural / 2.5  # Approximate mapping

        return {
            "w_ee": w_ee,
            "w_ei": 6.0,
            "w_ie": 8.0,
            "w_ii": 1.0,
            "tau_e": 0.010,  # 10ms
            "tau_i": 0.010,  # 10ms
            "a_e": 1.5,
            "theta_e": 4.0,
            "a_i": 1.5,
            "theta_i": 3.7,
            "P": 0.5,  # External input to E
            "Q": 0.0,  # External input to I
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Initial state with small perturbation
        if seed is not None:
            np.random.seed(seed)
        self.state = np.array([0.3 + np.random.normal(0, 0.01), 0.2 + np.random.normal(0, 0.01)])

        self.t = 0.0
        self.steps = 0
        self.prev_f_stim = self.f_target
        self.w_ee_current = self.w_ee_initial

        # Reset frequency estimator
        self.freq_estimator.reset()

        # Initial observation
        E = self.state[0]
        f_hat = self.freq_estimator.update(E)

        obs = np.array([E, f_hat], dtype=np.float32)
        info = {"f_hat": f_hat, "f_stim": self.f_target, "state": self.state.copy()}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.

        Parameters:
            action: Stimulation frequency [f_stim]

        Returns:
            observation: [E, f_hat]
            reward: Frequency tracking reward
            terminated: Episode done
            truncated: False
            info: Additional information
        """
        f_stim = np.clip(action[0], 4.0, 15.0)

        # Periodic stimulation: I_stim = A * sin(2π * f_stim * t)
        I_stim = self.stim_amplitude * np.sin(2 * np.pi * f_stim * self.t)

        # W_EE drift (for experiment b)
        if self.w_ee_drift:
            self.w_ee_current += self.w_ee_drift_rate * self.dt
            self.w_ee_current = np.clip(self.w_ee_current, 1.0, 10.0)

        # Integrate Wilson-Cowan dynamics
        self.state = self._integrate_dynamics(self.state, I_stim)

        # Update time and step counter
        self.t += self.dt
        self.steps += 1

        # Update frequency estimate
        E = self.state[0]
        f_hat = self.freq_estimator.update(E)

        # Compute reward
        reward = self._compute_reward(f_hat, f_stim)
        self.prev_f_stim = f_stim

        # Observation
        obs = np.array([E, f_hat], dtype=np.float32)

        # Check termination
        terminated = self.steps >= self.max_steps
        truncated = False

        info = {
            "f_hat": f_hat,
            "f_stim": f_stim,
            "state": self.state.copy(),
            "w_ee": self.w_ee_current,
        }

        return obs, reward, terminated, truncated, info

    def _integrate_dynamics(self, state: np.ndarray, I_stim: float) -> np.ndarray:
        """
        Integrate Wilson-Cowan dynamics using RK4.

        Parameters:
            state: [E, I]
            I_stim: External stimulation

        Returns:
            new_state: [E', I']
        """
        E, I = state

        # Define derivative function
        def derivative(s):
            E_, I_ = s
            # Sigmoid activation
            input_E = self.w_ee_current * E_ - self.params["w_ei"] * I_ + self.params["P"] + I_stim
            input_I = self.params["w_ie"] * E_ - self.params["w_ii"] * I_ + self.params["Q"]

            S_E = 1.0 / (
                1.0 + np.exp(-self.params["a_e"] * (input_E - self.params["theta_e"]))
            )
            S_I = 1.0 / (
                1.0 + np.exp(-self.params["a_i"] * (input_I - self.params["theta_i"]))
            )

            dE_dt = (-E_ + S_E) / self.params["tau_e"]
            dI_dt = (-I_ + S_I) / self.params["tau_i"]

            return np.array([dE_dt, dI_dt])

        # RK4 integration
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * self.dt * k1)
        k3 = derivative(state + 0.5 * self.dt * k2)
        k4 = derivative(state + self.dt * k3)

        new_state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Clip to valid range
        new_state = np.clip(new_state, 0.0, 1.0)

        return new_state

    def _compute_reward(self, f_hat: float, f_stim: float) -> float:
        """
        Compute reward for frequency tracking.

        Components:
        1. Frequency tracking: -(f_hat - f_target)²
        2. Control smoothness: -0.1 * (f_stim - f_prev)²
        3. Locking bonus: +1.0 if |f_hat - f_target| < 0.5 Hz

        Parameters:
            f_hat: Estimated frequency
            f_stim: Stimulation frequency

        Returns:
            reward: Total reward
        """
        # Frequency tracking error
        R_freq = -((f_hat - self.f_target) ** 2)

        # Control smoothness
        R_smooth = -0.1 * ((f_stim - self.prev_f_stim) ** 2)

        # Locking bonus (tight tolerance)
        R_lock = 1.0 if abs(f_hat - self.f_target) < 0.5 else 0.0

        return R_freq + R_smooth + R_lock

    def render(self):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass
