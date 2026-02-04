"""
Complete Controller Comparison Experiment

PhIHP Agent vs All Baseline Controllers

完整对比实验：
1. 训练 PhIHP Agent (50 episodes)
2. 评估所有控制器 (20 episodes each)
3. 生成对比报告
4. 统计显著性检验
5. 多维度可视化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple

from src.envs.wilson_cowan import WilsonCowanEnv
from src.models.world_model import PIRLWorldModel
from src.agents.phihp_agent import PhIHPAgent
from src.agents.baselines import (
    PIDController,
    OpenLoopStimulator,
    RandomController,
    BangBangController
)


def compute_episode_metrics(
    states: np.ndarray,
    actions: np.ndarray,
    target_E: float = 0.15,
    dt: float = 0.001
) -> Dict[str, float]:
    """
    计算单个 episode 的所有评估指标

    Args:
        states: 状态轨迹 [T, 2]
        actions: 动作序列 [T-1]
        target_E: 目标 E 值
        dt: 时间步长

    Returns:
        metrics: 指标字典
    """
    E = states[:, 0]
    I = states[:, 1]

    # 1. 最终误差
    final_error = abs(E[-1] - target_E)

    # 2. 平均误差
    mean_error = np.mean(np.abs(E - target_E))

    # 3. 均方根误差 (RMSE)
    rmse = np.sqrt(np.mean((E - target_E) ** 2))

    # 4. 稳定时间（误差首次<5%并保持）
    threshold = 0.05
    stable_mask = np.abs(E - target_E) < threshold
    if np.any(stable_mask):
        # 找到最后一次不稳定的点
        unstable_indices = np.where(~stable_mask)[0]
        if len(unstable_indices) > 0:
            settling_time = (unstable_indices[-1] + 1) * dt
        else:
            settling_time = 0.0
    else:
        settling_time = len(E) * dt  # 从未稳定

    # 5. 控制能量
    control_energy = np.sum(actions ** 2) * dt

    # 6. 平均绝对控制
    mean_abs_control = np.mean(np.abs(actions))

    # 7. 控制平滑度（控制变化率）
    control_changes = np.abs(np.diff(actions))
    control_smoothness = np.mean(control_changes)

    # 8. 振荡指标（状态变化率）
    state_changes = np.abs(np.diff(E))
    oscillation = np.mean(state_changes)

    # 9. 超调量
    overshoot = np.max(E) - target_E if np.max(E) > target_E else 0.0

    # 10. 下冲量
    undershoot = target_E - np.min(E) if np.min(E) < target_E else 0.0

    return {
        'final_error': final_error,
        'mean_error': mean_error,
        'rmse': rmse,
        'settling_time': settling_time,
        'control_energy': control_energy,
        'mean_abs_control': mean_abs_control,
        'control_smoothness': control_smoothness,
        'oscillation': oscillation,
        'overshoot': overshoot,
        'undershoot': undershoot
    }


def evaluate_controller(
    controller,
    env: WilsonCowanEnv,
    n_episodes: int = 20,
    target_E: float = 0.15
) -> Dict[str, List[float]]:
    """
    评估单个控制器

    Returns:
        results: 每个 episode 的指标
    """
    all_metrics = {
        'rewards': [],
        'final_errors': [],
        'mean_errors': [],
        'rmse': [],
        'settling_times': [],
        'control_energies': [],
        'mean_abs_controls': [],
        'control_smoothnesses': [],
        'oscillations': [],
        'overshoots': [],
        'undershoots': []
    }

    # 保存一个完整轨迹用于可视化
    sample_trajectory = None

    for episode in range(n_episodes):
        obs, info = env.reset()

        # Reset controller (different methods for different types)
        if hasattr(controller, 'reset'):
            controller.reset()
        elif hasattr(controller, 'reset_safety_layer'):
            controller.reset_safety_layer()

        states = [obs]
        actions = []
        episode_reward = 0.0

        for step in range(env.max_steps):
            # Get action (different methods for different types)
            if hasattr(controller, 'select_action'):
                action = controller.select_action(obs, explore=False)
            else:
                action = controller(obs)
            actions.append(action[0])

            # 环境步进
            next_obs, _, terminated, truncated, info = env.step(action)

            # 计算奖励
            E, I = obs[0], obs[1]
            E_next, I_next = next_obs[0], next_obs[1]
            u = action[0]

            R_task = -((E_next - target_E) ** 2)
            R_energy = -0.1 * (u ** 2)
            R_oscillation = -0.5 * ((E_next - E) ** 2)

            if E_next < 0.05 or E_next > 0.95:
                R_safety = -10.0
            elif E_next < 0.1 or E_next > 0.9:
                R_safety = -1.0
            else:
                R_safety = 0.0

            reward = R_task + R_energy + R_oscillation + R_safety
            episode_reward += reward

            states.append(next_obs)
            obs = next_obs

            if terminated or truncated:
                break

        # 转换为数组
        states = np.array(states)
        actions = np.array(actions)

        # 计算指标
        metrics = compute_episode_metrics(states, actions, target_E, env.dt)

        all_metrics['rewards'].append(episode_reward)
        all_metrics['final_errors'].append(metrics['final_error'])
        all_metrics['mean_errors'].append(metrics['mean_error'])
        all_metrics['rmse'].append(metrics['rmse'])
        all_metrics['settling_times'].append(metrics['settling_time'])
        all_metrics['control_energies'].append(metrics['control_energy'])
        all_metrics['mean_abs_controls'].append(metrics['mean_abs_control'])
        all_metrics['control_smoothnesses'].append(metrics['control_smoothness'])
        all_metrics['oscillations'].append(metrics['oscillation'])
        all_metrics['overshoots'].append(metrics['overshoot'])
        all_metrics['undershoots'].append(metrics['undershoot'])

        # 保存第一个轨迹
        if episode == 0:
            sample_trajectory = {
                'states': states,
                'actions': actions,
                'times': np.arange(len(states)) * env.dt
            }

    all_metrics['sample_trajectory'] = sample_trajectory

    return all_metrics


def train_phihp_agent(
    env: WilsonCowanEnv,
    world_model,
    n_episodes: int = 50,
    device: str = 'cuda'
) -> PhIHPAgent:
    """
    训练 PhIHP Agent

    Returns:
        trained_agent: 训练好的代理
    """
    print("Training PhIHP Agent...")

    agent = PhIHPAgent(
        state_dim=2,
        action_dim=1,
        world_model=world_model,
        device=device,
        batch_size=64
    )

    for episode in range(n_episodes):
        state, info = env.reset()
        agent.reset_safety_layer()

        for step in range(env.max_steps):
            action = agent.select_action(state, explore=True, noise_scale=0.1)
            next_state, _, terminated, truncated, info = env.step(action)
            reward = agent.compute_reward(state, action, next_state)

            agent.replay_buffer.push(state, action, reward, next_state, terminated or truncated)

            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()

            state = next_state

            if terminated or truncated:
                break

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1}/{n_episodes}")

    print("[OK] Training complete")
    return agent


def statistical_test(
    data1: List[float],
    data2: List[float],
    metric_name: str
) -> Dict:
    """
    统计显著性检验

    使用 Mann-Whitney U 检验（非参数）

    Returns:
        test_results: 检验结果
    """
    # Mann-Whitney U 检验（不假设正态分布）
    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

    # 效应量（Cohen's d）
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1), np.std(data2)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    # 显著性判断
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "n.s."

    return {
        'statistic': statistic,
        'p_value': p_value,
        'significance': significance,
        'cohens_d': cohens_d,
        'mean1': mean1,
        'mean2': mean2,
        'std1': std1,
        'std2': std2
    }


def generate_comparison_report(
    results: Dict,
    save_dir: str = 'results'
) -> str:
    """
    生成详细的对比报告

    Returns:
        report_path: 报告文件路径
    """
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, 'comparison_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Controller Comparison Report\n")
        f.write("=" * 80 + "\n\n")

        # 1. 总体性能排名
        f.write("1. Overall Performance Ranking\n")
        f.write("-" * 80 + "\n\n")

        # 按奖励排序
        f.write("By Mean Reward (higher is better):\n")
        sorted_by_reward = sorted(
            results.items(),
            key=lambda x: np.mean(x[1]['rewards']),
            reverse=True
        )
        for rank, (name, data) in enumerate(sorted_by_reward, 1):
            mean_r = np.mean(data['rewards'])
            std_r = np.std(data['rewards'])
            f.write(f"{rank}. {name:25s} {mean_r:10.2f} ± {std_r:8.2f}\n")

        f.write("\n")

        # 按最终误差排序
        f.write("By Final Error (lower is better):\n")
        sorted_by_error = sorted(
            results.items(),
            key=lambda x: np.mean(x[1]['final_errors'])
        )
        for rank, (name, data) in enumerate(sorted_by_error, 1):
            mean_e = np.mean(data['final_errors'])
            std_e = np.std(data['final_errors'])
            f.write(f"{rank}. {name:25s} {mean_e:10.6f} ± {std_e:8.6f}\n")

        f.write("\n\n")

        # 2. 详细指标对比
        f.write("2. Detailed Metrics Comparison\n")
        f.write("-" * 80 + "\n\n")

        metrics_to_report = [
            ('final_errors', 'Final Error', 'lower'),
            ('mean_errors', 'Mean Error', 'lower'),
            ('rmse', 'RMSE', 'lower'),
            ('settling_times', 'Settling Time (s)', 'lower'),
            ('control_energies', 'Control Energy', 'lower'),
            ('mean_abs_controls', 'Mean |Control|', 'lower'),
            ('control_smoothnesses', 'Control Smoothness', 'lower'),
            ('oscillations', 'Oscillation', 'lower'),
            ('overshoots', 'Overshoot', 'lower'),
            ('undershoots', 'Undershoot', 'lower')
        ]

        for metric_key, metric_name, direction in metrics_to_report:
            f.write(f"{metric_name}:\n")

            sorted_data = sorted(
                results.items(),
                key=lambda x: np.mean(x[1][metric_key]),
                reverse=(direction == 'higher')
            )

            for name, data in sorted_data:
                mean_val = np.mean(data[metric_key])
                std_val = np.std(data[metric_key])
                f.write(f"  {name:25s} {mean_val:10.6f} ± {std_val:8.6f}\n")

            f.write("\n")

        # 3. 统计显著性检验（如果有 PhIHP）
        if 'PhIHP' in results:
            f.write("\n3. Statistical Significance Tests (PhIHP vs Others)\n")
            f.write("-" * 80 + "\n\n")

            phihp_data = results['PhIHP']

            for name, data in results.items():
                if name == 'PhIHP':
                    continue

                f.write(f"PhIHP vs {name}:\n")

                # 对奖励进行检验
                test = statistical_test(
                    phihp_data['rewards'],
                    data['rewards'],
                    'reward'
                )

                f.write(f"  Reward: p={test['p_value']:.4f} {test['significance']}, ")
                f.write(f"Cohen's d={test['cohens_d']:.3f}\n")
                f.write(f"    PhIHP: {test['mean1']:.2f} ± {test['std1']:.2f}\n")
                f.write(f"    {name}: {test['mean2']:.2f} ± {test['std2']:.2f}\n")

                # 对最终误差进行检验
                test = statistical_test(
                    phihp_data['final_errors'],
                    data['final_errors'],
                    'final_error'
                )

                f.write(f"  Final Error: p={test['p_value']:.4f} {test['significance']}, ")
                f.write(f"Cohen's d={test['cohens_d']:.3f}\n")
                f.write(f"    PhIHP: {test['mean1']:.6f} ± {test['std1']:.6f}\n")
                f.write(f"    {name}: {test['mean2']:.6f} ± {test['std2']:.6f}\n\n")

            f.write("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant\n")

    return report_path


def main():
    """主实验函数"""
    print("=" * 80)
    print("Complete Controller Comparison Experiment")
    print("=" * 80)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 创建环境
    env = WilsonCowanEnv(
        dt=0.001,
        max_steps=500,
        device=device,
        target_state=[0.15, 0.1],
        reward_type='none'
    )
    print("[OK] Environment created")

    # 创建世界模型
    world_model = PIRLWorldModel(
        physics_model=env.model,
        hidden_dim=64,
        device=device
    )
    print("[OK] World model created")

    # 1. 训练 PhIHP Agent
    print("\n" + "-" * 80)
    print("Phase 1: Training PhIHP Agent")
    print("-" * 80)

    phihp_agent = train_phihp_agent(env, world_model, n_episodes=50, device=device)

    # 2. 创建所有控制器
    print("\n" + "-" * 80)
    print("Phase 2: Evaluating All Controllers")
    print("-" * 80)

    controllers = {
        'PhIHP': phihp_agent,
        'PID': PIDController(target=0.15, Kp=2.0, Ki=0.5, Kd=0.1, dt=env.dt),
        'Bang-Bang': BangBangController(target=0.15, threshold=0.05),
        'Open Loop': OpenLoopStimulator(frequency=10.0, amplitude=0.5, waveform='sine', dt=env.dt),
        'Random': RandomController(distribution='uniform', seed=42)
    }

    # 评估所有控制器
    n_eval_episodes = 20
    all_results = {}

    for name, controller in controllers.items():
        print(f"\nEvaluating {name}...")
        results = evaluate_controller(controller, env, n_eval_episodes, target_E=0.15)
        all_results[name] = results

        mean_reward = np.mean(results['rewards'])
        mean_error = np.mean(results['final_errors'])
        print(f"  Mean reward: {mean_reward:.2f}")
        print(f"  Final error: {mean_error:.6f}")

    # 3. 生成报告
    print("\n" + "-" * 80)
    print("Phase 3: Generating Report and Visualizations")
    print("-" * 80)

    report_path = generate_comparison_report(all_results)
    print(f"[OK] Report saved: {report_path}")

    # 4. 生成可视化
    print("\nGenerating visualizations...")
    os.makedirs('figures', exist_ok=True)

    # 4.1 对比柱状图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics_to_plot = [
        ('rewards', 'Mean Episode Reward', 'higher'),
        ('final_errors', 'Final Error', 'lower'),
        ('control_energies', 'Control Energy', 'lower'),
        ('settling_times', 'Settling Time (s)', 'lower'),
        ('control_smoothnesses', 'Control Smoothness', 'lower'),
        ('oscillations', 'State Oscillation', 'lower')
    ]

    for idx, (metric_key, title, direction) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]

        names = list(all_results.keys())
        means = [np.mean(all_results[name][metric_key]) for name in names]
        stds = [np.std(all_results[name][metric_key]) for name in names]

        colors = ['red' if name == 'PhIHP' else 'skyblue' for name in names]
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        # 标注最佳值
        if direction == 'lower':
            best_idx = np.argmin(means)
        else:
            best_idx = np.argmax(means)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    save_path = 'figures/comparison_bar.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()

    # 4.2 轨迹对比
    fig, axes = plt.subplots(len(controllers), 2, figsize=(14, 3*len(controllers)))

    for idx, (name, results) in enumerate(all_results.items()):
        traj = results['sample_trajectory']
        states = traj['states']
        actions = traj['actions']
        times = traj['times']

        # 状态轨迹
        ax = axes[idx, 0] if len(controllers) > 1 else axes[0]
        ax.plot(times, states[:, 0], 'b-', label='E', linewidth=2)
        ax.axhline(y=0.15, color='g', linestyle='--', label='Target', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('E (Excitatory)')
        ax.set_title(f'{name}: State Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 动作轨迹
        ax = axes[idx, 1] if len(controllers) > 1 else axes[1]
        ax.plot(times[:-1], actions, 'r-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Control Input u')
        ax.set_title(f'{name}: Control Actions')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'figures/trajectory_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()

    # 4.3 相空间对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (name, results) in enumerate(all_results.items()):
        ax = axes[idx // 3, idx % 3]

        traj = results['sample_trajectory']
        states = traj['states']

        ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')
        ax.plot(states[-1, 0], states[-1, 1], 'ro', markersize=10, label='End')
        ax.plot(0.15, 0.1, 'g*', markersize=15, label='Target')

        ax.set_xlabel('E (Excitatory)')
        ax.set_ylabel('I (Inhibitory)')
        ax.set_title(f'{name}: Phase Portrait')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 如果有空余的subplot，关闭它
    if len(controllers) < 6:
        for idx in range(len(controllers), 6):
            axes[idx // 3, idx % 3].axis('off')

    plt.tight_layout()
    save_path = 'figures/phase_portrait_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()

    # 5. 打印统计显著性
    print("\n" + "=" * 80)
    print("Statistical Significance (PhIHP vs Others)")
    print("=" * 80)

    if 'PhIHP' in all_results:
        phihp_rewards = all_results['PhIHP']['rewards']
        phihp_errors = all_results['PhIHP']['final_errors']

        for name in all_results.keys():
            if name == 'PhIHP':
                continue

            print(f"\nPhIHP vs {name}:")

            # 奖励检验
            test = statistical_test(phihp_rewards, all_results[name]['rewards'], 'reward')
            print(f"  Reward: p={test['p_value']:.4f} {test['significance']}")
            print(f"    PhIHP: {test['mean1']:.2f} ± {test['std1']:.2f}")
            print(f"    {name}: {test['mean2']:.2f} ± {test['std2']:.2f}")
            print(f"    Effect size (Cohen's d): {test['cohens_d']:.3f}")

            # 误差检验
            test = statistical_test(phihp_errors, all_results[name]['final_errors'], 'error')
            print(f"  Final Error: p={test['p_value']:.4f} {test['significance']}")
            print(f"    PhIHP: {test['mean1']:.6f} ± {test['std1']:.6f}")
            print(f"    {name}: {test['mean2']:.6f} ± {test['std2']:.6f}")
            print(f"    Effect size (Cohen's d): {test['cohens_d']:.3f}")

    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {report_path}")
    print("  - figures/comparison_bar.png")
    print("  - figures/trajectory_comparison.png")
    print("  - figures/phase_portrait_comparison.png")


if __name__ == "__main__":
    main()
