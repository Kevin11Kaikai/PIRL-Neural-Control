"""
Example: Testing Wilson-Cowan Environment

This script demonstrates how to use the WilsonCowanEnv with the Gymnasium API.
It runs an episode with random actions and visualizes the results.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from envs import WilsonCowanEnv


def run_random_episode(render_plot=True, seed=42):
    """
    Run an episode with random actions.

    Args:
        render_plot: Whether to create visualization
        seed: Random seed for reproducibility

    Returns:
        Dictionary with episode statistics
    """
    # Create environment
    env = WilsonCowanEnv(
        dt=0.001,              # 1ms time step
        max_steps=1000,        # 1 second total
        device='cuda',         # Use GPU if available
        action_limit=2.0,      # Action bounds [-2, 2]
        target_state=[0.5, 0.3],  # Target E and I values
        reward_type='quadratic'   # Quadratic reward function
    )

    print("Wilson-Cowan Environment Test")
    print("=" * 60)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Target state: E={env.target_state[0]}, I={env.target_state[1]}")

    # Reset environment
    obs, info = env.reset(seed=seed)
    print(f"Initial state: E={obs[0]:.4f}, I={obs[1]:.4f}")

    # Storage for trajectory
    states = [obs.copy()]
    actions = []
    rewards = []
    times = [0.0]

    # Run episode
    print("\nRunning episode with random actions...")
    total_reward = 0.0

    for step in range(env.max_steps):
        # Sample random action
        action = env.action_space.sample()

        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Store data
        states.append(obs.copy())
        actions.append(action[0])
        rewards.append(reward)
        times.append(info['time'])
        total_reward += reward

        # Check if episode ended
        if terminated or truncated:
            break

    # Convert to arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    times = np.array(times)

    # Print results
    print(f"\nEpisode Results:")
    print(f"  Steps: {len(states)}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Average reward: {total_reward / len(rewards):.4f}")
    print(f"  Final state: E={states[-1, 0]:.4f}, I={states[-1, 1]:.4f}")
    print(f"  Distance to target: {np.linalg.norm(states[-1] - env.target_state):.4f}")
    print("=" * 60)

    # Visualization
    if render_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # State trajectories
        axes[0, 0].plot(times, states[:, 0], 'b-', label='E (Excitatory)', linewidth=1.5)
        axes[0, 0].plot(times, states[:, 1], 'r-', label='I (Inhibitory)', linewidth=1.5)
        axes[0, 0].axhline(env.target_state[0], color='b', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(env.target_state[1], color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Activity')
        axes[0, 0].set_title('State Trajectory (Random Policy)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Phase portrait
        axes[0, 1].plot(states[:, 0], states[:, 1], 'k-', alpha=0.5, linewidth=1)
        axes[0, 1].plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')
        axes[0, 1].plot(states[-1, 0], states[-1, 1], 'ro', markersize=10, label='End')
        axes[0, 1].plot(env.target_state[0], env.target_state[1], 'b*',
                        markersize=15, label='Target')
        axes[0, 1].set_xlabel('E (Excitatory)')
        axes[0, 1].set_ylabel('I (Inhibitory)')
        axes[0, 1].set_title('Phase Portrait')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Actions
        axes[1, 0].plot(times[:-1], actions, 'g-', linewidth=1)
        axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Control Input')
        axes[1, 0].set_title('Random Actions')
        axes[1, 0].grid(True, alpha=0.3)

        # Rewards
        axes[1, 1].plot(times[:-1], rewards, 'm-', linewidth=1, label='Reward')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_title(f'Rewards (Total: {total_reward:.2f})')
        axes[1, 1].grid(True, alpha=0.3)

        # Cumulative reward
        ax2 = axes[1, 1].twinx()
        cumulative_rewards = np.cumsum(rewards)
        ax2.plot(times[:-1], cumulative_rewards, 'c--', alpha=0.5,
                 linewidth=2, label='Cumulative')
        ax2.set_ylabel('Cumulative Reward', color='c')
        ax2.tick_params(axis='y', labelcolor='c')

        plt.tight_layout()
        plt.show()

    env.close()

    return {
        'total_reward': total_reward,
        'episode_length': len(states),
        'final_state': states[-1],
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'times': times
    }


if __name__ == "__main__":
    # Run test
    results = run_random_episode(render_plot=True)
    print("\n[SUCCESS] Environment test completed!")
