"""
Baseline Controllers for Frequency Control

Implements simple frequency control strategies for comparison with PhIHP.
"""

import numpy as np
from typing import Optional


class PIFrequencyController:
    """
    PI (Proportional-Integral) Frequency Controller

    Uses PI control law to adjust stimulation frequency:
        Δf_stim = α * error + K_i * ∫error dt

    This is the main baseline from the paper.

    Parameters:
        f_target: Target oscillation frequency (Hz)
        alpha: Proportional gain
        Ki: Integral gain
        dt: Time step for integration
    """

    def __init__(
        self,
        f_target: float = 9.0,
        alpha: float = 0.3,
        Ki: float = 0.005,
        dt: float = 0.001,
    ):
        self.f_target = f_target
        self.alpha = alpha
        self.Ki = Ki
        self.dt = dt

        self.integral = 0.0
        self.f_stim = f_target

    def reset(self):
        """Reset controller state for new episode."""
        self.integral = 0.0
        self.f_stim = self.f_target

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute control action.

        Parameters:
            obs: [E, f_hat]

        Returns:
            action: [f_stim]
        """
        f_hat = obs[1]
        error = self.f_target - f_hat

        # Accumulate integral
        self.integral += error * self.dt

        # Anti-windup: limit integral
        self.integral = np.clip(self.integral, -10.0, 10.0)

        # PI control law
        delta = self.alpha * error + self.Ki * self.integral
        self.f_stim = np.clip(self.f_stim + delta, 4.0, 15.0)

        return np.array([self.f_stim], dtype=np.float32)


class OpenLoopFrequencyController:
    """
    Open-Loop Frequency Controller

    Simply outputs fixed f_stim = f_target, ignoring feedback.
    This is the worst-case baseline.

    Parameters:
        f_target: Fixed stimulation frequency
    """

    def __init__(self, f_target: float = 9.0):
        self.f_target = f_target

    def reset(self):
        """Reset controller (no state to reset)."""
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute control action.

        Parameters:
            obs: [E, f_hat] (ignored)

        Returns:
            action: [f_stim]
        """
        return np.array([self.f_target], dtype=np.float32)


class RandomFrequencyController:
    """
    Random Frequency Controller

    Outputs random frequency within valid range.
    Used as sanity check baseline.

    Parameters:
        seed: Random seed for reproducibility
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        """Reset controller."""
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute random control action.

        Parameters:
            obs: [E, f_hat] (ignored)

        Returns:
            action: [f_stim] uniformly sampled from [4, 15] Hz
        """
        f_stim = self.rng.uniform(4.0, 15.0)
        return np.array([f_stim], dtype=np.float32)


class AdaptivePIFrequencyController:
    """
    Adaptive PI Controller with gain scheduling.

    Adjusts gains based on error magnitude for faster response
    when far from target and smoother control when close.

    Parameters:
        f_target: Target frequency
        alpha_base: Base proportional gain
        Ki_base: Base integral gain
        adaptive: Whether to use adaptive gains
    """

    def __init__(
        self,
        f_target: float = 9.0,
        alpha_base: float = 0.3,
        Ki_base: float = 0.005,
        dt: float = 0.001,
        adaptive: bool = True,
    ):
        self.f_target = f_target
        self.alpha_base = alpha_base
        self.Ki_base = Ki_base
        self.dt = dt
        self.adaptive = adaptive

        self.integral = 0.0
        self.f_stim = f_target

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.f_stim = self.f_target

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute adaptive PI control action.

        Parameters:
            obs: [E, f_hat]

        Returns:
            action: [f_stim]
        """
        f_hat = obs[1]
        error = self.f_target - f_hat

        # Adaptive gain scheduling
        if self.adaptive:
            error_abs = abs(error)
            # High gains when error is large, low gains when close
            gain_multiplier = 1.0 + 2.0 * error_abs  # Scale with error magnitude
            alpha = self.alpha_base * gain_multiplier
            Ki = self.Ki_base * gain_multiplier
        else:
            alpha = self.alpha_base
            Ki = self.Ki_base

        # Accumulate integral
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -10.0, 10.0)

        # PI control law
        delta = alpha * error + Ki * self.integral
        self.f_stim = np.clip(self.f_stim + delta, 4.0, 15.0)

        return np.array([self.f_stim], dtype=np.float32)
