"""
Frequency Control Experiments

Runs 4 frequency control experiments comparing PI, OpenLoop, and PhIHP controllers.

Experiments:
    a) Locking: 12Hz → 9Hz (basic frequency locking)
    b) Drift: w_ee drift tracking (adaptive control)
    c) Extended: 14Hz → 9Hz (larger frequency shift)
    d) Wrong: 7Hz → 8Hz (control against natural tendency)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse

from src.envs.wilson_cowan_freq import WilsonCowanFreqEnv
from src.agents.baselines_freq import (
    PIFrequencyController,
    OpenLoopFrequencyController,
    RandomFrequencyController,
)
from src.agents.phihp_freq_agent import PhIHPFreqAgent


# Experiment configurations
FREQ_EXPERIMENTS = {
    "a_locking": {
        "f_target": 9.0,
        "f_natural": 12.0,
        "w_ee_drift": False,
        "w_ee_drift_rate": 0.0,
        "description": "Frequency Locking: 12Hz → 9Hz",
    },
    "b_drift": {
        "f_target": 9.0,
        "f_natural": 10.5,
        "w_ee_drift": True,
        "w_ee_drift_rate": 0.5,
        "description": "Drift Tracking: w_ee changes over time",
    },
    "c_extended": {
        "f_target": 9.0,
        "f_natural": 14.0,
        "w_ee_drift": False,
        "w_ee_drift_rate": 0.0,
        "description": "Extended Range: 14Hz → 9Hz",
    },
    "d_wrong": {
        "f_target": 8.0,
        "f_natural": 7.0,
        "w_ee_drift": False,
        "w_ee_drift_rate": 0.0,
        "description": "Wrong Direction: Target > Natural",
    },
}


def train_phihp(env: WilsonCowanFreqEnv, n_episodes: int = 50, device: str = "cpu") -> PhIHPFreqAgent:
    """
    Train PhIHP agent on frequency control task.

    Parameters:
        env: Frequency control environment
        n_episodes: Number of training episodes
        device: 'cpu' or 'cuda'

    Returns:
        agent: Trained PhIHP agent
    """
    agent = PhIHPFreqAgent(
        obs_dim=2,
        action_dim=1,
        hidden_dim=128,
        actor_lr=5e-5,  # Lower LR for stability
        critic_lr=1e-4,
        gamma=0.99,
        tau=0.005,
        noise_scale=0.3,  # Higher exploration for frequency control
        noise_decay=0.995,
        min_noise=0.05,
        device=device,
    )

    print(f"Training PhIHP for {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # Select action
            action = agent.select_action(obs, explore=True)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.push(obs, action, reward, next_obs, done)

            # Update agent
            if len(agent.replay_buffer) >= 64:
                agent.update(batch_size=64)

            obs = next_obs
            episode_reward += reward

        # Decay noise
        agent.decay_noise()

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}, Reward: {episode_reward:.2f}")

    print("[OK] PhIHP training complete\n")
    return agent


def evaluate_controller(
    controller,
    env: WilsonCowanFreqEnv,
    n_episodes: int = 10,
    controller_name: str = "Controller",
) -> Dict[str, List]:
    """
    Evaluate controller on frequency control task.

    Parameters:
        controller: Controller to evaluate
        env: Environment
        n_episodes: Number of evaluation episodes
        controller_name: Name for printing

    Returns:
        results: Dictionary of metrics
    """
    results = {
        "rewards": [],
        "freq_errors": [],
        "f_hats": [],
        "f_stims": [],
        "trajectories": [],
    }

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        freq_errors = []
        f_hats_ep = []
        f_stims_ep = []
        E_trajectory = []

        done = False
        steps = 0

        while not done:
            # Get action
            if hasattr(controller, "select_action"):
                action = controller.select_action(obs, explore=False)
            else:
                action = controller(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record metrics
            episode_reward += reward
            freq_errors.append(abs(info["f_hat"] - env.f_target))
            f_hats_ep.append(info["f_hat"])
            f_stims_ep.append(info["f_stim"])
            E_trajectory.append(info["state"][0])

            obs = next_obs
            steps += 1

        results["rewards"].append(episode_reward)
        results["freq_errors"].append(np.mean(freq_errors))
        results["f_hats"].append(f_hats_ep)
        results["f_stims"].append(f_stims_ep)
        results["trajectories"].append(E_trajectory)

    mean_reward = np.mean(results["rewards"])
    mean_error = np.mean(results["freq_errors"])

    print(f"  {controller_name}: Reward={mean_reward:.2f}, Freq Error={mean_error:.3f} Hz")

    return results


def run_experiment(
    experiment_name: str, config: Dict, n_train: int = 50, n_eval: int = 10, device: str = "cpu"
) -> Dict:
    """
    Run single frequency control experiment.

    Parameters:
        experiment_name: Name of experiment
        config: Experiment configuration
        n_train: Training episodes for PhIHP
        n_eval: Evaluation episodes
        device: 'cpu' or 'cuda'

    Returns:
        all_results: Results for all controllers
    """
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Description: {config['description']}")
    print(f"f_natural={config['f_natural']:.1f} Hz, f_target={config['f_target']:.1f} Hz")
    print("=" * 80)

    # Create environment
    env = WilsonCowanFreqEnv(
        f_target=config["f_target"],
        f_natural=config["f_natural"],
        stim_amplitude=0.5,
        w_ee_drift=config["w_ee_drift"],
        w_ee_drift_rate=config["w_ee_drift_rate"],
        dt=0.001,
        episode_length=5.0,
        device=device,
    )

    # Train PhIHP
    print("\nPhase 1: Training PhIHP")
    print("-" * 80)
    phihp_agent = train_phihp(env, n_episodes=n_train, device=device)

    # Create baseline controllers
    controllers = {
        "PhIHP": phihp_agent,
        "PI": PIFrequencyController(f_target=config["f_target"], alpha=0.3, Ki=0.005),
        "OpenLoop": OpenLoopFrequencyController(f_target=config["f_target"]),
        "Random": RandomFrequencyController(seed=42),
    }

    # Evaluate all controllers
    print("\nPhase 2: Evaluating Controllers")
    print("-" * 80)

    all_results = {}
    for name, controller in controllers.items():
        if hasattr(controller, "reset"):
            controller.reset()
        results = evaluate_controller(controller, env, n_eval, name)
        all_results[name] = results

    return all_results


def generate_report(experiment_name: str, results: Dict, config: Dict):
    """
    Generate text report for experiment.

    Parameters:
        experiment_name: Name of experiment
        results: Results from all controllers
        config: Experiment configuration
    """
    print("\n" + "=" * 80)
    print(f"Results Summary: {experiment_name}")
    print("=" * 80)

    # Rankings
    print("\nRankings by Mean Reward (higher is better):")
    rewards = [(name, np.mean(res["rewards"])) for name, res in results.items()]
    rewards.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, reward) in enumerate(rewards, 1):
        print(f"{rank}. {name:12s} {reward:8.2f}")

    print("\nRankings by Frequency Error (lower is better):")
    errors = [(name, np.mean(res["freq_errors"])) for name, res in results.items()]
    errors.sort(key=lambda x: x[1])
    for rank, (name, error) in enumerate(errors, 1):
        print(f"{rank}. {name:12s} {error:8.3f} Hz")


def plot_experiment(experiment_name: str, results: Dict, config: Dict, save_path: str = None):
    """
    Generate visualization for experiment.

    Parameters:
        experiment_name: Name of experiment
        results: Results from all controllers
        config: Experiment configuration
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{config['description']}", fontsize=16, fontweight="bold")

    controllers = list(results.keys())
    colors = {"PhIHP": "blue", "PI": "green", "OpenLoop": "orange", "Random": "red"}

    # Plot 1: Frequency tracking (first episode)
    ax = axes[0, 0]
    for name in controllers:
        if name in results:
            f_hats = results[name]["f_hats"][0]  # First episode
            time = np.arange(len(f_hats)) * 0.001
            ax.plot(time, f_hats, label=name, color=colors.get(name, "gray"), alpha=0.7)

    ax.axhline(config["f_target"], color="black", linestyle="--", label="Target", linewidth=2)
    ax.axhline(config["f_natural"], color="gray", linestyle=":", label="Natural", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Estimated Frequency (Hz)")
    ax.set_title("Frequency Tracking")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Control actions
    ax = axes[0, 1]
    for name in controllers:
        if name in results and name != "Random":
            f_stims = results[name]["f_stims"][0]
            time = np.arange(len(f_stims)) * 0.001
            ax.plot(time, f_stims, label=name, color=colors.get(name, "gray"), alpha=0.7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stimulation Frequency (Hz)")
    ax.set_title("Control Actions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Performance comparison (bar chart)
    ax = axes[1, 0]
    names = list(results.keys())
    rewards = [np.mean(results[name]["rewards"]) for name in names]
    bars = ax.bar(names, rewards, color=[colors.get(name, "gray") for name in names], alpha=0.7)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Reward Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    # Plot 4: Frequency error comparison (bar chart)
    ax = axes[1, 1]
    errors = [np.mean(results[name]["freq_errors"]) for name in names]
    bars = ax.bar(names, errors, color=[colors.get(name, "gray") for name in names], alpha=0.7)
    ax.set_ylabel("Mean Frequency Error (Hz)")
    ax.set_title("Tracking Error Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Figure saved: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run frequency control experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=list(FREQ_EXPERIMENTS.keys()) + ["all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument("--n_train", type=int, default=50, help="Training episodes for PhIHP")
    parser.add_argument("--n_eval", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--output_dir", type=str, default="results/freq_control", help="Output directory")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/figures", exist_ok=True)

    # Determine which experiments to run
    if args.experiment == "all":
        experiments_to_run = list(FREQ_EXPERIMENTS.keys())
    else:
        experiments_to_run = [args.experiment]

    print("\n" + "=" * 80)
    print("FREQUENCY CONTROL EXPERIMENTS")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Training: {args.n_train} episodes")
    print(f"Evaluation: {args.n_eval} episodes")
    print(f"Experiments: {', '.join(experiments_to_run)}")
    print("=" * 80 + "\n")

    # Run experiments
    all_experiment_results = {}

    for exp_name in experiments_to_run:
        config = FREQ_EXPERIMENTS[exp_name]

        # Run experiment
        results = run_experiment(
            exp_name, config, n_train=args.n_train, n_eval=args.n_eval, device=args.device
        )

        # Generate report
        generate_report(exp_name, results, config)

        # Generate plot
        plot_path = f"{args.output_dir}/figures/{exp_name}.png"
        plot_experiment(exp_name, results, config, save_path=plot_path)

        all_experiment_results[exp_name] = results

        print("\n")

    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
