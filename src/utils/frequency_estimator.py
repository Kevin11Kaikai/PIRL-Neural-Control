"""
Frequency Estimator for Wilson-Cowan Neural Oscillations

Uses FFT to estimate the dominant frequency from a sliding window of E(t) samples.
"""

import numpy as np
from collections import deque


class FrequencyEstimator:
    """
    Real-time frequency estimation using FFT on a sliding window.

    Parameters:
        window_size (int): Number of samples to use for FFT
        dt (float): Sampling interval (s)
        min_samples (int): Minimum samples before estimating
    """

    def __init__(self, window_size=500, dt=0.001, min_samples=200):
        self.window_size = window_size
        self.dt = dt
        self.min_samples = min_samples
        self.buffer = deque(maxlen=window_size)

    def update(self, E):
        """
        Add new sample and return estimated frequency.

        Parameters:
            E (float): Current excitatory population activity

        Returns:
            f_hat (float): Estimated dominant frequency (Hz)
        """
        self.buffer.append(E)
        return self.estimate()

    def estimate(self):
        """
        Estimate dominant frequency using FFT.

        Returns:
            f_hat (float): Estimated frequency in Hz
        """
        if len(self.buffer) < self.min_samples:
            return 10.0  # Default return 10Hz before enough samples

        # Convert to array and remove DC component
        signal = np.array(self.buffer)
        signal = signal - signal.mean()

        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(signal))
        signal = signal * window

        # Compute FFT
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), self.dt)

        # Only consider 1-20 Hz range (physiologically relevant)
        valid_mask = (freqs >= 1) & (freqs <= 20)
        if not valid_mask.any():
            return 10.0

        valid_fft = np.abs(fft[valid_mask])
        valid_freqs = freqs[valid_mask]

        # Return frequency with maximum power
        peak_idx = np.argmax(valid_fft)
        return valid_freqs[peak_idx]

    def reset(self):
        """Clear buffer for new episode."""
        self.buffer.clear()

    def get_spectrum(self):
        """
        Get full frequency spectrum for visualization.

        Returns:
            freqs (np.ndarray): Frequency bins
            power (np.ndarray): Power spectral density
        """
        if len(self.buffer) < self.min_samples:
            return np.array([]), np.array([])

        signal = np.array(self.buffer)
        signal = signal - signal.mean()

        window = np.hanning(len(signal))
        signal = signal * window

        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), self.dt)
        power = np.abs(fft) ** 2

        return freqs, power
