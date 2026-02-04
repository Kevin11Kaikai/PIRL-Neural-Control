"""
Quick Demo of Frequency Control

Demonstrates frequency control with minimal training for fast verification.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from src.envs.wilson_cowan_freq import WilsonCowanFreqEnv
from src.agents.baselines_freq import PIFrequencyController, OpenLoopFrequencyController


def demo_frequency_control():
    """Quick demonstration of frequency control."""
    print("\n" + "=" * 80)
    print("QUICK FREQUENCY CONTROL DEMO")
    print("=" * 80)
    print("Task: Lock 12Hz oscillation to 9Hz target\n")

    # Create environment
    env = WilsonCowanFreqEnv(
        f_target=9.0,
        f_natural=12.0,
        dt=0.001,
        episode_length=3.0,  # 3 seconds
    )

    # Create controllers
    controllers = {
        "PI Controller": PIFrequencyController(f_target=9.0, alpha=0.3, Ki=0.005),
        "Open Loop": OpenLoopFrequencyController(f_target=9.0),
    }

    results = {}

    # Run each controller
    for name, controller in controllers.items():
        print(f"\nRunning {name}...")
        obs, _ = env.reset(seed=42)
        controller.reset()

        f_hats = []
        f_stims = []
        E_vals = []
        rewards = []

        done = False
        total_reward = 0

        while not done:
            action = controller(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            f_hats.append(info["f_hat"])
            f_stims.append(info["f_stim"])
            E_vals.append(info["state"][0])
            rewards.append(reward)
            total_reward += reward

            obs = next_obs

        results[name] = {
            "f_hats": f_hats,
            "f_stims": f_stims,
            "E_vals": E_vals,
            "rewards": rewards,
            "total_reward": total_reward,
        }

        mean_freq = np.mean(f_hats[-500:])  # Last 0.5s
        freq_error = abs(mean_freq - 9.0)
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final frequency: {mean_freq:.2f} Hz (error: {freq_error:.2f} Hz)")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    time = np.arange(len(results["PI Controller"]["f_hats"])) * 0.001

    # Plot 1: Frequency tracking
    ax = axes[0]
    for name in controllers.keys():
        ax.plot(time, results[name]["f_hats"], label=name, linewidth=1.5)

    ax.axhline(9.0, color="black", linestyle="--", label="Target (9Hz)", linewidth=2)
    ax.axhline(12.0, color="gray", linestyle=":", label="Natural (12Hz)", linewidth=1)
    ax.set_ylabel("Estimated Frequency (Hz)", fontsize=11)
    ax.set_title("Frequency Locking: 12Hz → 9Hz", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Control actions
    ax = axes[1]
    for name in controllers.keys():
        ax.plot(time, results[name]["f_stims"], label=name, linewidth=1.5)

    ax.set_ylabel("Stimulation Frequency (Hz)", fontsize=11)
    ax.set_title("Control Actions", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: State trajectory
    ax = axes[2]
    for name in controllers.keys():
        ax.plot(time, results[name]["E_vals"], label=name, alpha=0.7)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Excitatory Activity (E)", fontsize=11)
    ax.set_title("Neural Activity", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs("figures/freq_control", exist_ok=True)
    save_path = "figures/freq_control/demo_locking.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n[OK] Figure saved: {save_path}")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for name in controllers.keys():
        mean_freq = np.mean(results[name]["f_hats"][-500:])
        error = abs(mean_freq - 9.0)
        reward = results[name]["total_reward"]
        print(f"\n{name}:")
        print(f"  Final frequency: {mean_freq:.2f} Hz")
        print(f"  Tracking error: {error:.2f} Hz")
        print(f"  Total reward: {reward:.2f}")

    print("\n" + "=" * 80)
    print("[OK] Demo complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    demo_frequency_control()
