"""
Baseline Controllers for Comparison

传统控制器基线，用于与 PhIHP Agent 性能对比：
1. PID Controller - 经典 PID 反馈控制
2. Open Loop Stimulator - 固定频率开环刺激
3. Random Controller - 随机动作基线
"""

import numpy as np
from typing import Optional, Tuple


class PIDController:
    """
    PID (Proportional-Integral-Derivative) 控制器

    经典反馈控制，根据误差、误差积分和误差导数调整控制信号

    u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de/dt

    其中 e(t) = target - current_state
    """

    def __init__(
        self,
        target: float = 0.15,
        Kp: float = 2.0,
        Ki: float = 0.5,
        Kd: float = 0.1,
        dt: float = 0.001,
        action_limit: float = 2.0,
        control_variable: str = 'E'  # 'E' 或 'I'
    ):
        """
        初始化 PID 控制器

        Args:
            target: 目标值（对 E 或 I）
            Kp: 比例增益
            Ki: 积分增益
            Kd: 微分增益
            dt: 时间步长
            action_limit: 动作上下限
            control_variable: 控制变量（'E' 或 'I'）
        """
        self.target = target
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.action_limit = action_limit
        self.control_variable = control_variable

        # 状态变量
        self.integral = 0.0
        self.last_error = 0.0
        self.initialized = False

    def reset(self):
        """重置控制器内部状态"""
        self.integral = 0.0
        self.last_error = 0.0
        self.initialized = False

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        计算控制动作

        Args:
            obs: 观测 [E, I]

        Returns:
            action: 控制信号 [u]
        """
        # 选择控制变量
        if self.control_variable == 'E':
            current_value = obs[0]
        elif self.control_variable == 'I':
            current_value = obs[1]
        else:
            raise ValueError(f"Unknown control variable: {self.control_variable}")

        # 计算误差
        error = self.target - current_value

        # 积分项（梯形法则）
        if self.initialized:
            self.integral += (error + self.last_error) * self.dt / 2.0
        else:
            self.initialized = True

        # 微分项
        if self.last_error is not None:
            derivative = (error - self.last_error) / self.dt
        else:
            derivative = 0.0

        # PID 输出
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # 限幅
        u = np.clip(u, -self.action_limit, self.action_limit)

        # 抗积分饱和（Anti-windup）
        # 如果输出饱和，停止积分累积
        if abs(u) >= self.action_limit and np.sign(u) == np.sign(error):
            self.integral -= (error + self.last_error) * self.dt / 2.0

        # 更新上一次误差
        self.last_error = error

        return np.array([u], dtype=np.float32)


class OpenLoopStimulator:
    """
    开环刺激器

    以固定频率和幅度输出周期性刺激信号，不依赖反馈

    常用模式:
    - Sinusoidal: u(t) = A * sin(2πf*t)
    - Square wave: u(t) = A * sign(sin(2πf*t))
    - Pulse train: u(t) = A if (t mod T) < duty_cycle*T else 0
    """

    def __init__(
        self,
        frequency: float = 10.0,
        amplitude: float = 1.0,
        waveform: str = 'sine',  # 'sine', 'square', 'pulse'
        duty_cycle: float = 0.5,
        phase: float = 0.0,
        dt: float = 0.001,
        action_limit: float = 2.0
    ):
        """
        初始化开环刺激器

        Args:
            frequency: 刺激频率 (Hz)
            amplitude: 刺激幅度
            waveform: 波形类型 ('sine', 'square', 'pulse')
            duty_cycle: 占空比（仅用于 pulse）
            phase: 初始相位 (radians)
            dt: 时间步长
            action_limit: 动作上下限
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.waveform = waveform
        self.duty_cycle = duty_cycle
        self.phase = phase
        self.dt = dt
        self.action_limit = action_limit

        # 内部状态
        self.t = 0.0

    def reset(self):
        """重置时间"""
        self.t = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        生成刺激信号（不使用观测）

        Args:
            obs: 观测（未使用）

        Returns:
            action: 刺激信号 [u]
        """
        # 计算相位
        omega = 2.0 * np.pi * self.frequency
        phase_current = omega * self.t + self.phase

        # 生成波形
        if self.waveform == 'sine':
            u = self.amplitude * np.sin(phase_current)

        elif self.waveform == 'square':
            u = self.amplitude * np.sign(np.sin(phase_current))

        elif self.waveform == 'pulse':
            # 脉冲序列
            period = 1.0 / self.frequency
            t_in_period = self.t % period
            if t_in_period < self.duty_cycle * period:
                u = self.amplitude
            else:
                u = 0.0

        else:
            raise ValueError(f"Unknown waveform: {self.waveform}")

        # 限幅
        u = np.clip(u, -self.action_limit, self.action_limit)

        # 更新时间
        self.t += self.dt

        return np.array([u], dtype=np.float32)


class RandomController:
    """
    随机控制器

    输出随机动作，作为最低性能基线

    支持模式:
    - uniform: 均匀分布
    - gaussian: 高斯分布
    - ornstein_uhlenbeck: OU过程（带时间相关性）
    """

    def __init__(
        self,
        action_limit: float = 2.0,
        distribution: str = 'uniform',  # 'uniform', 'gaussian', 'ou'
        mean: float = 0.0,
        std: float = 0.5,
        ou_theta: float = 0.15,  # OU过程回归速率
        ou_sigma: float = 0.2,   # OU过程噪声强度
        dt: float = 0.001,
        seed: Optional[int] = None
    ):
        """
        初始化随机控制器

        Args:
            action_limit: 动作上下限
            distribution: 分布类型
            mean: 均值（gaussian）
            std: 标准差（gaussian）
            ou_theta: OU过程回归速率
            ou_sigma: OU过程噪声强度
            dt: 时间步长
            seed: 随机种子
        """
        self.action_limit = action_limit
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.dt = dt

        # 随机数生成器
        self.rng = np.random.RandomState(seed)

        # OU过程状态
        self.ou_state = 0.0

    def reset(self):
        """重置内部状态"""
        self.ou_state = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        生成随机动作（不使用观测）

        Args:
            obs: 观测（未使用）

        Returns:
            action: 随机动作 [u]
        """
        if self.distribution == 'uniform':
            # 均匀分布
            u = self.rng.uniform(-self.action_limit, self.action_limit)

        elif self.distribution == 'gaussian':
            # 高斯分布
            u = self.rng.normal(self.mean, self.std)

        elif self.distribution == 'ou':
            # Ornstein-Uhlenbeck 过程
            # dx = theta * (mean - x) * dt + sigma * sqrt(dt) * dW
            dx = (
                self.ou_theta * (self.mean - self.ou_state) * self.dt +
                self.ou_sigma * np.sqrt(self.dt) * self.rng.randn()
            )
            self.ou_state += dx
            u = self.ou_state

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        # 限幅
        u = np.clip(u, -self.action_limit, self.action_limit)

        return np.array([u], dtype=np.float32)


class BangBangController:
    """
    Bang-Bang 控制器（额外基线）

    二值控制：当误差为正时全力刺激，为负时停止

    u(t) = {  +u_max  if e > threshold
           {  -u_max  if e < -threshold
           {   0      otherwise
    """

    def __init__(
        self,
        target: float = 0.15,
        threshold: float = 0.05,
        action_limit: float = 2.0,
        control_variable: str = 'E'
    ):
        """
        初始化 Bang-Bang 控制器

        Args:
            target: 目标值
            threshold: 死区阈值
            action_limit: 动作上下限
            control_variable: 控制变量
        """
        self.target = target
        self.threshold = threshold
        self.action_limit = action_limit
        self.control_variable = control_variable

    def reset(self):
        """无需重置状态"""
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        计算控制动作

        Args:
            obs: 观测 [E, I]

        Returns:
            action: 控制信号 [u]
        """
        # 选择控制变量
        if self.control_variable == 'E':
            current_value = obs[0]
        elif self.control_variable == 'I':
            current_value = obs[1]
        else:
            raise ValueError(f"Unknown control variable: {self.control_variable}")

        # 计算误差
        error = self.target - current_value

        # Bang-Bang 控制律
        if error > self.threshold:
            u = self.action_limit
        elif error < -self.threshold:
            u = -self.action_limit
        else:
            u = 0.0

        return np.array([u], dtype=np.float32)


def test_baseline_controllers():
    """
    测试所有基线控制器

    1. 创建环境
    2. 测试每个控制器 10 episodes
    3. 对比性能
    """
    print("=" * 70)
    print("Testing Baseline Controllers")
    print("=" * 70)

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from src.envs.wilson_cowan import WilsonCowanEnv
    import matplotlib.pyplot as plt

    # 创建环境
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    env = WilsonCowanEnv(
        dt=0.001,
        max_steps=500,
        device=device,
        target_state=[0.15, 0.1],
        reward_type='none'
    )
    print(f"\n[OK] Environment created (device: {device})")
    print(f"Target state: E={env.target_state[0]:.2f}, I={env.target_state[1]:.2f}")

    # 定义控制器
    controllers = {
        'PID': PIDController(
            target=0.15,
            Kp=2.0,
            Ki=0.5,
            Kd=0.1,
            dt=env.dt,
            action_limit=2.0
        ),
        'Open Loop (10Hz Sine)': OpenLoopStimulator(
            frequency=10.0,
            amplitude=0.5,
            waveform='sine',
            dt=env.dt
        ),
        'Random (Uniform)': RandomController(
            action_limit=2.0,
            distribution='uniform',
            seed=42
        ),
        'Bang-Bang': BangBangController(
            target=0.15,
            threshold=0.05,
            action_limit=2.0
        )
    }

    print(f"\n[OK] Created {len(controllers)} baseline controllers")

    # 测试每个控制器
    print("\n" + "-" * 70)
    print("Testing controllers...")

    results = {}
    n_episodes = 10

    for name, controller in controllers.items():
        print(f"\n{name}:")

        episode_rewards = []
        final_errors = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            controller.reset()

            episode_reward = 0.0

            for step in range(env.max_steps):
                # 控制器输出
                action = controller(obs)

                # 环境步进
                next_obs, _, terminated, truncated, info = env.step(action)

                # 计算奖励（与 PhIHP 相同的奖励函数）
                E, I = obs[0], obs[1]
                E_next, I_next = next_obs[0], next_obs[1]
                u = action[0]

                # 任务奖励
                target_E = 0.15
                R_task = -((E_next - target_E) ** 2)

                # 能量惩罚
                R_energy = -0.1 * (u ** 2)

                # 振荡惩罚
                dE = E_next - E
                R_oscillation = -0.5 * (dE ** 2)

                # 安全奖励
                if E_next < 0.05 or E_next > 0.95:
                    R_safety = -10.0
                elif E_next < 0.1 or E_next > 0.9:
                    R_safety = -1.0
                else:
                    R_safety = 0.0

                reward = R_task + R_energy + R_oscillation + R_safety
                episode_reward += reward

                obs = next_obs

                if terminated or truncated:
                    break

            # 记录最终误差
            final_E = obs[0]
            final_error = abs(final_E - target_E)

            episode_rewards.append(episode_reward)
            final_errors.append(final_error)

        # 统计
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_error = np.mean(final_errors)
        std_error = np.std(final_errors)

        results[name] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_error': mean_error,
            'std_error': std_error
        }

        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Final error: {mean_error:.4f} ± {std_error:.4f}")

    # 性能对比
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)

    # 按奖励排序
    sorted_controllers = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)

    print("\nRanking by reward:")
    for rank, (name, stats) in enumerate(sorted_controllers, 1):
        print(f"{rank}. {name:30s} Reward: {stats['mean_reward']:8.2f} ± {stats['std_reward']:6.2f}")

    print("\nRanking by final error:")
    sorted_by_error = sorted(results.items(), key=lambda x: x[1]['mean_error'])
    for rank, (name, stats) in enumerate(sorted_by_error, 1):
        print(f"{rank}. {name:30s} Error:  {stats['mean_error']:.4f} ± {stats['std_error']:.4f}")

    # 可视化一个完整 episode
    print("\n" + "-" * 70)
    print("Generating visualization...")

    os.makedirs('figures', exist_ok=True)

    fig, axes = plt.subplots(len(controllers), 2, figsize=(14, 3*len(controllers)))

    for idx, (name, controller) in enumerate(controllers.items()):
        obs, info = env.reset()
        controller.reset()

        states = [obs]
        actions = []

        for step in range(env.max_steps):
            action = controller(obs)
            next_obs, _, terminated, truncated, info = env.step(action)

            states.append(next_obs)
            actions.append(action[0])

            obs = next_obs

            if terminated or truncated:
                break

        states = np.array(states)
        actions = np.array(actions)
        times = np.arange(len(states)) * env.dt

        # 状态轨迹
        ax = axes[idx, 0] if len(controllers) > 1 else axes[0]
        ax.plot(times, states[:, 0], 'b-', label='E', linewidth=2)
        ax.axhline(y=0.15, color='g', linestyle='--', label='Target E=0.15', linewidth=2)
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
    save_path = 'figures/baseline_controllers.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Visualization saved: {save_path}")
    plt.close()

    # 对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 奖励对比
    names = list(results.keys())
    rewards = [results[name]['mean_reward'] for name in names]
    errors_bar = [results[name]['std_reward'] for name in names]

    axes[0].bar(range(len(names)), rewards, yerr=errors_bar, capsize=5, alpha=0.7)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Episode Reward')
    axes[0].set_title('Controller Comparison: Reward')
    axes[0].grid(True, alpha=0.3, axis='y')

    # 误差对比
    errors = [results[name]['mean_error'] for name in names]
    errors_bar = [results[name]['std_error'] for name in names]

    axes[1].bar(range(len(names)), errors, yerr=errors_bar, capsize=5, alpha=0.7, color='orange')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel('Mean Final Error')
    axes[1].set_title('Controller Comparison: Final Error')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = 'figures/baseline_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Comparison saved: {save_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    test_baseline_controllers()
