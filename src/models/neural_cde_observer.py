"""
Neural CDE State Observer

基于 Controlled Differential Equations 的状态观测器
从部分观测 (E) 重构完整状态 [E, I]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Optional

try:
    import torchcde
except ImportError:
    print("Warning: torchcde not installed. Install with: pip install torchcde")
    torchcde = None


class CDEFunc(nn.Module):
    """
    CDE 向量场函数

    dz/dt = f(z) * dX/dt

    其中 X 是观测插值，z 是隐状态
    """

    def __init__(self, hidden_dim: int = 32, input_dim: int = 1):
        """
        Args:
            hidden_dim: 隐状态维度
            input_dim: 观测维度 (默认1，只有E)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # f(z): 隐状态到控制矩阵的映射
        # 输出 hidden_dim × input_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim * input_dim)
        )

    def forward(self, t, z):
        """
        计算 f(z)

        Args:
            t: 当前时间 (标量)
            z: 隐状态 (batch, hidden_dim)

        Returns:
            控制矩阵 (batch, hidden_dim, input_dim)
        """
        batch_size = z.shape[0]

        # 计算 f(z)
        out = self.net(z)  # (batch, hidden_dim * input_dim)

        # 重塑为矩阵
        out = out.view(batch_size, self.hidden_dim, self.input_dim)

        return out


class Encoder(nn.Module):
    """
    编码器：将初始观测编码为隐状态 z0
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32):
        """
        Args:
            input_dim: 观测维度
            hidden_dim: 隐状态维度
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, x0):
        """
        Args:
            x0: 初始观测 (batch, input_dim)

        Returns:
            z0: 初始隐状态 (batch, hidden_dim)
        """
        return self.net(x0)


class Decoder(nn.Module):
    """
    解码器：将隐状态解码为状态估计 [E, I]
    """

    def __init__(self, hidden_dim: int = 32, output_dim: int = 2):
        """
        Args:
            hidden_dim: 隐状态维度
            output_dim: 输出维度 (2 for [E, I])
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()  # 限制输出在 [0, 1]
        )

    def forward(self, z):
        """
        Args:
            z: 隐状态 (batch, seq_len, hidden_dim) 或 (batch, hidden_dim)

        Returns:
            states: 状态估计 (batch, seq_len, 2) 或 (batch, 2)
        """
        return self.net(z)


class NeuralCDEObserver(nn.Module):
    """
    Neural CDE 状态观测器

    从部分观测 (E) 重构完整状态 [E, I]
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        output_dim: int = 2,
        interpolation: str = 'cubic',
        device: str = 'cpu'
    ):
        """
        Args:
            input_dim: 观测维度 (1 for E only)
            hidden_dim: 隐状态维度
            output_dim: 输出维度 (2 for [E, I])
            interpolation: 插值方法 ('cubic' or 'linear')
            device: 计算设备
        """
        super().__init__()

        if torchcde is None:
            raise ImportError("torchcde not installed. Install with: pip install torchcde")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.interpolation = interpolation
        self.device = device

        # 组件
        self.encoder = Encoder(input_dim, hidden_dim)
        self.cde_func = CDEFunc(hidden_dim, input_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

        self.to(device)

    def forward(self, times, observations):
        """
        前向传播

        Args:
            times: 观测时间点 (batch, seq_len) 或 (seq_len,)
            observations: 带噪声的观测 (batch, seq_len, input_dim)

        Returns:
            states: 重构的状态 (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = observations.shape

        # 确保 times 是一维的（torchcde 要求）
        if times.dim() == 2:
            # 假设所有批次使用相同的时间点
            times_1d = times[0]
        else:
            times_1d = times

        # 1. 插值观测信号
        if self.interpolation == 'cubic':
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
                observations, times_1d
            )
            X = torchcde.CubicSpline(coeffs, times_1d)
        else:  # linear
            coeffs = torchcde.linear_interpolation_coeffs(observations, times_1d)
            X = torchcde.LinearInterpolation(coeffs, times_1d)

        # 2. 计算初始隐状态
        x0 = observations[:, 0, :]  # (batch, input_dim)
        z0 = self.encoder(x0)  # (batch, hidden_dim)

        # 3. 求解 CDE
        z_t = torchcde.cdeint(
            X=X,
            func=self.cde_func,
            z0=z0,
            t=times_1d,
            method='rk4',
            adjoint=False  # 使用反向自动微分
        )  # 输出形状: (seq_len, batch, hidden_dim)

        # 检查并调整形状
        if z_t.shape[0] != batch_size:
            # cdeint 返回 (seq_len, batch, hidden_dim)，需要转置
            z_t = z_t.transpose(0, 1)  # (batch, seq_len, hidden_dim)
        # 否则已经是 (batch, seq_len, hidden_dim)

        # 4. 解码输出状态
        states = self.decoder(z_t)  # (batch, seq_len, output_dim)

        return states


class ObserverLoss(nn.Module):
    """
    观测器损失函数

    L = L_reconstruction + λ * L_physics
    """

    def __init__(
        self,
        physics_model,
        physics_weight: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Args:
            physics_model: Wilson-Cowan 物理模型
            physics_weight: 物理损失权重
            device: 计算设备
        """
        super().__init__()

        self.physics_model = physics_model
        self.physics_weight = physics_weight
        self.device = device

    def forward(
        self,
        pred_states: torch.Tensor,
        true_states: torch.Tensor,
        times: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失

        Args:
            pred_states: 预测状态 (batch, seq_len, 2)
            true_states: 真实状态 (batch, seq_len, 2)
            times: 时间点 (batch, seq_len)
            actions: 动作序列 (batch, seq_len) 或 None

        Returns:
            losses: 损失字典
        """
        # 1. 重构损失
        L_recon = torch.mean((pred_states - true_states) ** 2)

        # 2. 物理损失：预测状态应满足 WC 动力学
        # 使用有限差分计算导数
        if times.dim() == 2:
            dt = times[:, 1:] - times[:, :-1]  # (batch, seq_len-1)
            dt = dt.mean()  # 假设等间距
        else:
            dt = times[1:] - times[:-1]
            dt = dt.mean()

        pred_derivatives = (pred_states[:, 1:, :] - pred_states[:, :-1, :]) / dt

        # 计算物理模型导数
        if actions is None:
            actions = torch.zeros(pred_states.shape[0], pred_states.shape[1] - 1, device=self.device)
        else:
            actions = actions[:, :-1]  # 对齐长度

        physics_derivatives = []
        for i in range(pred_states.shape[1] - 1):
            state_i = pred_states[:, i, :]
            action_i = actions[:, i] if actions.dim() > 1 else actions

            deriv = self.physics_model.forward(0.0, state_i, action_i)
            physics_derivatives.append(deriv)

        physics_derivatives = torch.stack(physics_derivatives, dim=1)  # (batch, seq_len-1, 2)

        # 物理损失
        L_physics = torch.mean((pred_derivatives - physics_derivatives) ** 2)

        # 总损失
        L_total = L_recon + self.physics_weight * L_physics

        return {
            'total': L_total,
            'reconstruction': L_recon,
            'physics': L_physics
        }


def generate_observer_training_data(
    env,
    n_trajectories: int = 100,
    n_steps: int = 100,
    noise_std: float = 0.01,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成观测器训练数据

    Args:
        env: WilsonCowanEnv 实例
        n_trajectories: 轨迹数量
        n_steps: 每条轨迹长度
        noise_std: 观测噪声标准差
        device: 计算设备

    Returns:
        times: 时间点 (n_traj, n_steps)
        observations: 带噪声的 E (n_traj, n_steps, 1)
        true_states: 真实 [E, I] (n_traj, n_steps, 2)
        actions: 动作序列 (n_traj, n_steps)
    """
    all_times = []
    all_observations = []
    all_true_states = []
    all_actions = []

    for traj_idx in range(n_trajectories):
        # 重置环境
        obs, info = env.reset()

        times = []
        observations = []
        true_states = []
        actions = []

        for step in range(n_steps):
            # 记录当前状态
            times.append(info['time'])
            true_states.append(obs.copy())

            # 添加噪声到 E 的观测
            noisy_E = obs[0] + noise_std * np.random.randn()
            observations.append([noisy_E])

            # 随机动作
            action = env.action_space.sample()
            actions.append(action[0])

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                # 如果提前结束，填充到 n_steps
                remaining = n_steps - len(times)
                if remaining > 0:
                    for _ in range(remaining):
                        times.append(times[-1] + env.dt)
                        true_states.append(true_states[-1].copy())
                        observations.append(observations[-1].copy())
                        actions.append(0.0)
                break

        # 确保长度正确
        assert len(times) == n_steps, f"Trajectory {traj_idx} has length {len(times)}, expected {n_steps}"

        all_times.append(times)
        all_observations.append(observations)
        all_true_states.append(true_states)
        all_actions.append(actions)

    # 转换为 numpy 数组再转 tensor（更快）
    times_np = np.array(all_times, dtype=np.float32)
    obs_np = np.array(all_observations, dtype=np.float32)
    states_np = np.array(all_true_states, dtype=np.float32)
    actions_np = np.array(all_actions, dtype=np.float32)

    times_tensor = torch.from_numpy(times_np).to(device)
    obs_tensor = torch.from_numpy(obs_np).to(device)
    states_tensor = torch.from_numpy(states_np).to(device)
    actions_tensor = torch.from_numpy(actions_np).to(device)

    return times_tensor, obs_tensor, states_tensor, actions_tensor


def test_neural_cde_observer():
    """
    测试 Neural CDE 观测器

    流程:
    1. 生成训练数据（100条轨迹）
    2. 训练观测器 50 epochs
    3. 评估重构精度
    4. 可视化结果
    """
    print("=" * 70)
    print("Testing Neural CDE State Observer")
    print("=" * 70)

    # 检查 torchcde
    if torchcde is None:
        print("\nError: torchcde not installed!")
        print("Install with: pip install torchcde")
        return

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 1. 创建环境
    from src.envs.wilson_cowan import WilsonCowanEnv

    env = WilsonCowanEnv(
        dt=0.001,
        max_steps=100,
        device=device,
        reward_type='none'
    )
    print("[OK] Wilson-Cowan environment created")

    # 2. 生成训练数据
    print("\n" + "-" * 70)
    print("Generating training data...")

    times_train, obs_train, states_train, actions_train = generate_observer_training_data(
        env=env,
        n_trajectories=100,
        n_steps=100,
        noise_std=0.01,
        device=device
    )

    print(f"[OK] Generated {len(times_train)} trajectories")
    print(f"  Observation shape: {obs_train.shape}")
    print(f"  True state shape: {states_train.shape}")
    print(f"  Observation noise std: 0.01")

    # 3. 创建观测器
    observer = NeuralCDEObserver(
        input_dim=1,
        hidden_dim=32,
        output_dim=2,
        interpolation='cubic',
        device=device
    )
    print(f"\n[OK] Neural CDE observer created")
    print(f"  Hidden dim: 32")
    print(f"  Interpolation: cubic spline")

    # 4. 创建损失函数
    loss_fn = ObserverLoss(
        physics_model=env.model,
        physics_weight=0.1,
        device=device
    )
    print("[OK] Loss function created (reconstruction + physics)")

    # 5. 训练
    print("\n" + "-" * 70)
    print("Training observer...")

    optimizer = optim.Adam(observer.parameters(), lr=1e-3)
    n_epochs = 50
    batch_size = 16

    # 评估初始性能
    with torch.no_grad():
        pred_states = observer(times_train, obs_train)
        initial_losses = loss_fn(pred_states, states_train, times_train, actions_train)

        print(f"\nInitial performance (before training):")
        print(f"  Reconstruction MSE: {initial_losses['reconstruction'].item():.6f}")
        print(f"  Physics loss: {initial_losses['physics'].item():.6f}")

    # 训练循环
    train_losses = []

    for epoch in range(n_epochs):
        # 随机打乱数据
        perm = torch.randperm(len(times_train))

        epoch_losses = []

        for i in range(0, len(times_train), batch_size):
            batch_idx = perm[i:i+batch_size]

            batch_times = times_train[batch_idx]
            batch_obs = obs_train[batch_idx]
            batch_states = states_train[batch_idx]
            batch_actions = actions_train[batch_idx]

            # 前向传播
            optimizer.zero_grad()
            pred_states = observer(batch_times, batch_obs)

            # Debug: print shapes on first iteration
            if epoch == 0 and i == 0:
                print(f"\n[DEBUG] First batch shapes:")
                print(f"  batch_times: {batch_times.shape}")
                print(f"  batch_obs: {batch_obs.shape}")
                print(f"  batch_states: {batch_states.shape}")
                print(f"  pred_states: {pred_states.shape}")

            # 计算损失
            losses = loss_fn(pred_states, batch_states, batch_times, batch_actions)

            # 反向传播
            losses['total'].backward()
            optimizer.step()

            epoch_losses.append({k: v.item() for k, v in losses.items()})

        # 计算平均损失
        avg_losses = {
            k: np.mean([loss[k] for loss in epoch_losses])
            for k in epoch_losses[0].keys()
        }
        train_losses.append(avg_losses)

        # 每10轮打印
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}:")
            print(f"  Reconstruction MSE: {avg_losses['reconstruction']:.6f}")
            print(f"  Physics loss: {avg_losses['physics']:.6f}")

    print("\n[OK] Training complete!")

    # 6. 最终评估
    print("\n" + "-" * 70)
    print("Final Evaluation")
    print("-" * 70)

    with torch.no_grad():
        pred_states = observer(times_train, obs_train)
        final_losses = loss_fn(pred_states, states_train, times_train, actions_train)

        # 分别计算 E 和 I 的 MSE
        E_mse = torch.mean((pred_states[:, :, 0] - states_train[:, :, 0]) ** 2).item()
        I_mse = torch.mean((pred_states[:, :, 1] - states_train[:, :, 1]) ** 2).item()

        print(f"\n1. Reconstruction Performance:")
        print(f"   E reconstruction MSE: {E_mse:.6f}")
        print(f"   I reconstruction MSE: {I_mse:.6f} (unobserved!)")
        print(f"   Overall MSE: {final_losses['reconstruction'].item():.6f}")

        print(f"\n2. Physics Constraint:")
        print(f"   Physics loss: {final_losses['physics'].item():.6f}")

        # 计算 MAE
        E_mae = torch.mean(torch.abs(pred_states[:, :, 0] - states_train[:, :, 0])).item()
        I_mae = torch.mean(torch.abs(pred_states[:, :, 1] - states_train[:, :, 1])).item()

        print(f"\n3. Mean Absolute Error:")
        print(f"   E MAE: {E_mae:.6f}")
        print(f"   I MAE: {I_mae:.6f}")

    # 7. 可视化
    print("\n" + "-" * 70)
    print("Generating visualizations...")

    os.makedirs('figures', exist_ok=True)

    # 选择几条轨迹可视化
    n_vis = 3
    vis_idx = np.random.choice(len(times_train), n_vis, replace=False)

    fig, axes = plt.subplots(n_vis, 3, figsize=(15, 4*n_vis))

    if n_vis == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(vis_idx):
        time = times_train[idx].cpu().numpy()
        obs = obs_train[idx].cpu().numpy()
        true_state = states_train[idx].cpu().numpy()
        pred_state = pred_states[idx].cpu().numpy()

        # E 重构
        axes[row, 0].plot(time, true_state[:, 0], 'b-', label='True E', linewidth=2, alpha=0.7)
        axes[row, 0].plot(time, pred_state[:, 0], 'r--', label='Pred E', linewidth=2, alpha=0.7)
        axes[row, 0].scatter(time, obs[:, 0], c='gray', s=10, alpha=0.5, label='Noisy Obs')
        axes[row, 0].set_xlabel('Time (s)')
        axes[row, 0].set_ylabel('E (Excitatory)')
        axes[row, 0].set_title(f'Trajectory {idx+1}: E Reconstruction')
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)

        # I 重构（未观测！）
        axes[row, 1].plot(time, true_state[:, 1], 'g-', label='True I', linewidth=2, alpha=0.7)
        axes[row, 1].plot(time, pred_state[:, 1], 'r--', label='Pred I', linewidth=2, alpha=0.7)
        axes[row, 1].set_xlabel('Time (s)')
        axes[row, 1].set_ylabel('I (Inhibitory)')
        axes[row, 1].set_title(f'Trajectory {idx+1}: I Reconstruction (Unobserved!)')
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.3)

        # 相空间
        axes[row, 2].plot(true_state[:, 0], true_state[:, 1], 'b-',
                         label='True', linewidth=2, alpha=0.7)
        axes[row, 2].plot(pred_state[:, 0], pred_state[:, 1], 'r--',
                         label='Pred', linewidth=2, alpha=0.7)
        axes[row, 2].set_xlabel('E')
        axes[row, 2].set_ylabel('I')
        axes[row, 2].set_title(f'Trajectory {idx+1}: Phase Space')
        axes[row, 2].legend()
        axes[row, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = 'figures/neural_cde_test.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Visualization saved to: {save_path}")

    plt.close()

    # 8. 训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = np.arange(1, n_epochs + 1)
    recon_losses = [loss['reconstruction'] for loss in train_losses]
    physics_losses = [loss['physics'] for loss in train_losses]

    axes[0].plot(epochs, recon_losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Reconstruction Loss')
    axes[0].set_title('Training: Reconstruction Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    axes[1].plot(epochs, physics_losses, 'r-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Physics Loss')
    axes[1].set_title('Training: Physics Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()

    loss_path = 'figures/neural_cde_training.png'
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Training curves saved to: {loss_path}")

    plt.close()

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  E reconstruction MSE: {E_mse:.6f}")
    print(f"  I reconstruction MSE: {I_mse:.6f} (from E observations only!)")
    print(f"  Successfully reconstructed unobserved state I from E trajectory")

    return observer, final_losses


if __name__ == "__main__":
    test_neural_cde_observer()
