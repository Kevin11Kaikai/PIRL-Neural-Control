"""
PhIHP Agent for Frequency Control

Adapts the PhIHP agent from state control to frequency control.

Key changes:
- Action space: frequency [4, 15] Hz instead of amplitude [-2, 2]
- Observation space: [E, f_hat] instead of [E, I]
- Reward function: frequency tracking instead of state tracking
- World model: predicts [E', f_hat'] given [E, f_hat, f_stim]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import deque
import random


class ActorNetworkFreq(nn.Module):
    """
    Actor network for frequency control.

    Input: [E, f_hat]
    Output: f_stim ∈ [4, 15] Hz
    """

    def __init__(self, obs_dim: int = 2, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Initialize with small weights for stable initial policy
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
            obs: [batch, 2] - [E, f_hat]

        Returns:
            action: [batch, 1] - f_stim in [4, 15] Hz
        """
        x = self.net(obs)
        # Map to [4, 15] Hz using sigmoid
        f_stim = 4.0 + 11.0 * torch.sigmoid(x)
        return f_stim


class CriticNetworkFreq(nn.Module):
    """
    Twin Q-networks for frequency control.

    Input: [E, f_hat, f_stim]
    Output: Q-value
    """

    def __init__(
        self, obs_dim: int = 2, action_dim: int = 1, hidden_dim: int = 128
    ):
        super().__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks.

        Parameters:
            obs: [batch, 2] - [E, f_hat]
            action: [batch, 1] - f_stim

        Returns:
            q1, q2: [batch, 1] - Q-values
        """
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBufferFreq:
    """Experience replay buffer for frequency control."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Add transition to buffer."""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)

        return (
            np.array(obs),
            np.array(action),
            np.array(reward),
            np.array(next_obs),
            np.array(done),
        )

    def __len__(self):
        return len(self.buffer)


class PhIHPFreqAgent:
    """
    Physics-Informed Hierarchical Planning agent for Frequency Control.

    Simplified version without world model imagination (can be added later).
    """

    def __init__(
        self,
        obs_dim: int = 2,
        action_dim: int = 1,
        hidden_dim: int = 128,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_scale: float = 0.3,
        noise_decay: float = 0.995,
        min_noise: float = 0.05,
        device: str = "cpu",
    ):
        """
        Initialize PhIHP agent for frequency control.

        Parameters:
            obs_dim: Observation dimension (2: E, f_hat)
            action_dim: Action dimension (1: f_stim)
            hidden_dim: Hidden layer size
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            noise_scale: Initial exploration noise
            noise_decay: Noise decay rate per episode
            min_noise: Minimum noise level
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.min_noise = min_noise

        # Networks
        self.actor = ActorNetworkFreq(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = ActorNetworkFreq(obs_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetworkFreq(obs_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.critic_target = CriticNetworkFreq(obs_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        # Replay buffer
        self.replay_buffer = ReplayBufferFreq(capacity=10000)

        # Training state
        self.current_noise = noise_scale

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select action using current policy.

        Parameters:
            obs: [E, f_hat]
            explore: Whether to add exploration noise

        Returns:
            action: [f_stim]
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor(obs_tensor).cpu().numpy()[0]

        if explore:
            # Add Gaussian noise for exploration
            noise = np.random.normal(0, self.current_noise, size=action.shape)
            action = action + noise
            action = np.clip(action, 4.0, 15.0)

        return action

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Update actor and critic networks.

        Parameters:
            batch_size: Batch size for training

        Returns:
            metrics: Training metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample batch
        obs, action, reward, next_obs, done = self.replay_buffer.sample(batch_size)

        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Update critic
        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            # Add target policy smoothing
            noise = torch.randn_like(next_action) * 0.2
            next_action = torch.clamp(next_action + noise, 4.0, 15.0)

            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next)
            q_target = reward + (1 - done) * self.gamma * q_next

        q1, q2 = self.critic(obs, action)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Update actor (delayed)
        actor_action = self.actor(obs)
        actor_loss = -self.critic.q1(torch.cat([obs, actor_action], dim=-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value": q1.mean().item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def decay_noise(self):
        """Decay exploration noise after each episode."""
        self.current_noise = max(
            self.min_noise, self.current_noise * self.noise_decay
        )

    def reset_safety_layer(self):
        """Reset method for compatibility with baseline controllers."""
        pass

    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
