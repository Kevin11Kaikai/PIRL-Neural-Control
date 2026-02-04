"""
Quick Controller Comparison (Fast Version)

快速对比实验：
- PhIHP: 10 episodes training
- Evaluation: 5 episodes each
- Total time: ~2-3 minutes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.envs.wilson_cowan import WilsonCowanEnv
from src.models.world_model import PIRLWorldModel
from src.agents.phihp_agent import PhIHPAgent
from src.agents.baselines import (
    PIDController,
    OpenLoopStimulator,
    RandomController,
    BangBangController
)


def quick_evaluate(controller, env, n_episodes=5):
    """快速评估控制器"""
    rewards = []
    final_errors = []

    # 保存一个轨迹
    sample_traj = None

    for ep in range(n_episodes):
        obs, _ = env.reset()

        # Reset controller (different methods for different types)
        if hasattr(controller, 'reset'):
            controller.reset()
        elif hasattr(controller, 'reset_safety_layer'):
            controller.reset_safety_layer()

        states = [obs]
        actions = []
        ep_reward = 0.0

        for step in range(env.max_steps):
            # Get action (different methods for different types)
            if hasattr(controller, 'select_action'):
                action = controller.select_action(obs, explore=False)
            else:
                action = controller(obs)
            next_obs, _, done, trunc, _ = env.step(action)

            # 计算奖励
            E, E_next, u = obs[0], next_obs[0], action[0]
            r = -(( E_next - 0.15)**2) - 0.1*(u**2) - 0.5*((E_next-E)**2)
            if E_next < 0.05 or E_next > 0.95:
                r -= 10.0

            ep_reward += r
            states.append(next_obs)
            actions.append(action[0])
            obs = next_obs

            if done or trunc:
                break

        final_errors.append(abs(states[-1][0] - 0.15))
        rewards.append(ep_reward)

        if ep == 0:
            sample_traj = (np.array(states), np.array(actions))

    return {
        'rewards': rewards,
        'errors': final_errors,
        'traj': sample_traj
    }


def main():
    print("=" * 70)
    print("Quick Controller Comparison")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 环境
    env = WilsonCowanEnv(dt=0.001, max_steps=500, device=device, target_state=[0.15, 0.1])
    print("[OK] Environment created")

    # 世界模型
    world_model = PIRLWorldModel(physics_model=env.model, hidden_dim=64, device=device)

    # 1. 训练 PhIHP (10 episodes)
    print("\nTraining PhIHP (10 episodes)...")
    phihp = PhIHPAgent(state_dim=2, action_dim=1, world_model=world_model, device=device, batch_size=32)

    for ep in range(10):
        state, _ = env.reset()
        phihp.reset_safety_layer()

        for step in range(env.max_steps):
            action = phihp.select_action(state, explore=True, noise_scale=0.1)
            next_state, _, done, trunc, _ = env.step(action)
            reward = phihp.compute_reward(state, action, next_state)

            phihp.replay_buffer.push(state, action, reward, next_state, done or trunc)

            if len(phihp.replay_buffer) > phihp.batch_size:
                phihp.update()

            state = next_state
            if done or trunc:
                break

    print("[OK] PhIHP trained")

    # 2. 创建控制器
    controllers = {
        'PhIHP': phihp,
        'Bang-Bang': BangBangController(target=0.15, threshold=0.05),
        'PID': PIDController(target=0.15, Kp=2.0, Ki=0.5, Kd=0.1, dt=env.dt),
        'Open Loop': OpenLoopStimulator(frequency=10.0, amplitude=0.5, dt=env.dt),
        'Random': RandomController(seed=42)
    }

    # 3. 评估
    print("\nEvaluating controllers (5 episodes each)...")
    results = {}

    for name, ctrl in controllers.items():
        print(f"  {name}...", end=' ')
        res = quick_evaluate(ctrl, env, n_episodes=5)
        results[name] = res
        print(f"Reward: {np.mean(res['rewards']):.1f}, Error: {np.mean(res['errors']):.4f}")

    # 4. 排名
    print("\n" + "=" * 70)
    print("Performance Ranking")
    print("=" * 70)

    print("\nBy Reward (higher is better):")
    sorted_by_reward = sorted(results.items(), key=lambda x: np.mean(x[1]['rewards']), reverse=True)
    for rank, (name, data) in enumerate(sorted_by_reward, 1):
        mean_r = np.mean(data['rewards'])
        std_r = np.std(data['rewards'])
        print(f"{rank}. {name:15s} {mean_r:8.1f} ± {std_r:6.1f}")

    print("\nBy Final Error (lower is better):")
    sorted_by_error = sorted(results.items(), key=lambda x: np.mean(x[1]['errors']))
    for rank, (name, data) in enumerate(sorted_by_error, 1):
        mean_e = np.mean(data['errors'])
        std_e = np.std(data['errors'])
        print(f"{rank}. {name:15s} {mean_e:.6f} ± {std_e:.6f}")

    # 5. 统计检验
    print("\n" + "=" * 70)
    print("Statistical Tests (PhIHP vs Others)")
    print("=" * 70)

    phihp_rewards = results['PhIHP']['rewards']
    phihp_errors = results['PhIHP']['errors']

    for name in ['Bang-Bang', 'PID']:
        print(f"\nPhIHP vs {name}:")

        # 奖励检验
        _, p_r = stats.mannwhitneyu(phihp_rewards, results[name]['rewards'])
        sig_r = "***" if p_r < 0.001 else "**" if p_r < 0.01 else "*" if p_r < 0.05 else "n.s."
        print(f"  Reward: p={p_r:.4f} {sig_r}")

        # 误差检验
        _, p_e = stats.mannwhitneyu(phihp_errors, results[name]['errors'])
        sig_e = "***" if p_e < 0.001 else "**" if p_e < 0.01 else "*" if p_e < 0.05 else "n.s."
        print(f"  Error:  p={p_e:.4f} {sig_e}")

    # 6. 可视化
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    os.makedirs('figures', exist_ok=True)

    # 对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results.keys())
    rewards_mean = [np.mean(results[n]['rewards']) for n in names]
    rewards_std = [np.std(results[n]['rewards']) for n in names]
    errors_mean = [np.mean(results[n]['errors']) for n in names]
    errors_std = [np.std(results[n]['errors']) for n in names]

    colors = ['red' if n == 'PhIHP' else 'skyblue' for n in names]

    # 奖励
    axes[0].bar(range(len(names)), rewards_mean, yerr=rewards_std, capsize=5, color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Episode Reward')
    axes[0].set_title('Reward Comparison')
    axes[0].grid(True, alpha=0.3, axis='y')

    # 误差
    axes[1].bar(range(len(names)), errors_mean, yerr=errors_std, capsize=5, color=colors, edgecolor='black')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel('Mean Final Error')
    axes[1].set_title('Final Error Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figures/quick_comparison.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved: figures/quick_comparison.png")
    plt.close()

    # 轨迹对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(results.items()):
        states, actions = data['traj']
        times = np.arange(len(states)) * env.dt

        ax = axes[idx]
        ax.plot(times, states[:, 0], 'b-', linewidth=2, label='E')
        ax.axhline(y=0.15, color='g', linestyle='--', linewidth=2, label='Target')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('E')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 关闭最后一个subplot
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('figures/quick_trajectories.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved: figures/quick_trajectories.png")
    plt.close()

    print("\n" + "=" * 70)
    print("Quick comparison complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
