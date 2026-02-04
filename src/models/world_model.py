"""
Physics-Informed Residual Learning (PIRL) World Model

结合 Wilson-Cowan 物理先验的世界模型，使用残差学习架构。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Optional


class ResidualNetwork(nn.Module):
    """
    残差网络：学习物理模型的修正项

    架构：3层MLP with ReLU activation
    权重初始化为小值（std=0.01），确保初始时主要依赖物理先验
    """

    def __init__(
        self,
        input_dim: int = 3,      # [E, I, u]
        hidden_dim: int = 64,
        output_dim: int = 2,     # [dE/dt, dI/dt]
        init_std: float = 0.01
    ):
        """
        初始化残差网络

        Args:
            input_dim: 输入维度 [E, I, u]
            hidden_dim: 隐藏层维度
            output_dim: 输出维度 [dE/dt修正, dI/dt修正]
            init_std: 权重初始化标准差
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 小权重初始化，确保初始残差接近零
        self._initialize_weights(init_std)

    def _initialize_weights(self, std: float):
        """用小标准差初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态 [batch, 2] 或 [2]
            action: 动作 [batch, 1] 或 [1] 或标量

        Returns:
            residual: 残差 [batch, 2] 或 [2]
        """
        # 处理输入形状
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if isinstance(action, (int, float)):
            action = torch.tensor([[action]], device=state.device, dtype=state.dtype)
        elif action.dim() == 0:
            action = action.unsqueeze(0).unsqueeze(0)
        elif action.dim() == 1:
            action = action.unsqueeze(1) if action.shape[0] == state.shape[0] else action.unsqueeze(0)

        # 确保batch维度匹配
        if action.shape[0] != state.shape[0]:
            action = action.expand(state.shape[0], -1)

        # 拼接输入 [E, I, u]
        x = torch.cat([state, action], dim=-1)

        # 计算残差
        residual = self.network(x)

        return residual.squeeze(0) if residual.shape[0] == 1 else residual


class PIRLWorldModel(nn.Module):
    """
    Physics-Informed Residual Learning 世界模型

    预测 = 物理模型 + 残差网络

    损失函数:
        L_pred = ||s_pred - s_true||^2
        L_physics = ||ds_pred/dt - f_WC(state)||^2
        L_total = L_pred + λ * L_physics
    """

    def __init__(
        self,
        physics_model,              # WilsonCowanODE 实例
        hidden_dim: int = 64,
        physics_weight: float = 0.1,
        device: str = 'cpu'
    ):
        """
        初始化 PIRL 世界模型

        Args:
            physics_model: Wilson-Cowan 物理模型
            hidden_dim: 残差网络隐藏层维度
            physics_weight: 物理损失权重 λ
            device: 计算设备
        """
        super().__init__()

        self.physics_model = physics_model
        self.residual_net = ResidualNetwork(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=2,
            init_std=0.01
        )
        self.physics_weight = physics_weight
        self.device = device

        self.to(device)

    def predict_derivative(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测状态导数 = 物理模型 + 残差

        Args:
            state: 当前状态 [batch, 2] 或 [2]
            action: 控制输入 [batch, 1] 或 [batch] 或 标量

        Returns:
            total_derivative: 总导数 [batch, 2] 或 [2]
            physics_derivative: 物理模型导数
            residual: 残差网络输出
        """
        # 处理 action 维度，确保与 WilsonCowanODE 兼容
        # WC expects action as [batch] or scalar
        action_for_physics = action
        if isinstance(action, torch.Tensor):
            if action.dim() == 2 and action.shape[1] == 1:
                # (batch, 1) -> (batch,)
                action_for_physics = action.squeeze(1)
            elif action.dim() == 1 and len(action) == 1:
                # (1,) -> scalar
                action_for_physics = action.item()

        # 物理模型导数
        physics_derivative = self.physics_model.forward(0.0, state, action_for_physics)

        # 残差网络修正
        residual = self.residual_net(state, action)

        # 总导数 = 物理 + 残差
        total_derivative = physics_derivative + residual

        return total_derivative, physics_derivative, residual

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dt: float = 0.001
    ) -> torch.Tensor:
        """
        使用欧拉法预测下一状态

        Args:
            state: 当前状态 [batch, 2] 或 [2]
            action: 控制输入
            dt: 时间步长

        Returns:
            next_state: 预测的下一状态
        """
        derivative, _, _ = self.predict_derivative(state, action)
        next_state = state + dt * derivative
        return next_state

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        dt: float = 0.001
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失

        Args:
            states: 当前状态 [batch, 2]
            actions: 动作 [batch, 1] 或 [batch]
            next_states: 真实下一状态 [batch, 2]
            dt: 时间步长

        Returns:
            losses: 包含各项损失的字典
        """
        # 预测下一状态
        pred_next_states = self.predict_next_state(states, actions, dt)

        # 预测损失：MSE
        L_pred = torch.mean((pred_next_states - next_states) ** 2)

        # 物理损失：预测导数与物理模型导数的差异
        total_derivative, physics_derivative, residual = self.predict_derivative(states, actions)

        # 从真实转移计算实际导数（有限差分）
        true_derivative = (next_states - states) / dt

        # 物理损失 = ||d_pred/dt - f_WC(state)||^2
        # 这里我们希望预测导数接近真实导数，同时不要偏离物理模型太远
        L_physics = torch.mean((total_derivative - physics_derivative) ** 2)

        # 总损失
        L_total = L_pred + self.physics_weight * L_physics

        return {
            'total': L_total,
            'prediction': L_pred,
            'physics': L_physics,
            'residual_norm': torch.mean(torch.norm(residual, dim=-1))
        }

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        optimizer: optim.Optimizer,
        dt: float = 0.001
    ) -> Dict[str, float]:
        """
        执行一步训练

        Args:
            states: 当前状态批次
            actions: 动作批次
            next_states: 下一状态批次
            optimizer: 优化器
            dt: 时间步长

        Returns:
            losses: 损失值字典
        """
        optimizer.zero_grad()

        # 计算损失
        losses = self.compute_loss(states, actions, next_states, dt)

        # 反向传播
        losses['total'].backward()
        optimizer.step()

        # 返回数值
        return {k: v.item() for k, v in losses.items()}


def generate_wc_trajectory(
    physics_model,
    n_steps: int = 1000,
    dt: float = 0.001,
    initial_state: Optional[torch.Tensor] = None,
    action_sequence: Optional[torch.Tensor] = None,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成 Wilson-Cowan 轨迹数据

    Args:
        physics_model: WilsonCowanODE 实例
        n_steps: 轨迹长度
        dt: 时间步长
        initial_state: 初始状态 [2]
        action_sequence: 动作序列 [n_steps] 或 None（使用随机动作）
        device: 计算设备

    Returns:
        states: 状态序列 [n_steps, 2]
        actions: 动作序列 [n_steps]
        next_states: 下一状态序列 [n_steps, 2]
    """
    if initial_state is None:
        initial_state = torch.tensor([0.1, 0.1], device=device, dtype=torch.float32)
        initial_state += 0.01 * torch.randn(2, device=device)

    if action_sequence is None:
        # 随机动作序列，范围 [-0.5, 0.5]
        action_sequence = torch.randn(n_steps, device=device) * 0.5

    states = []
    next_states = []
    actions = []

    current_state = initial_state

    for step in range(n_steps):
        action = action_sequence[step].item() if action_sequence.dim() > 0 else action_sequence.item()

        # 使用物理模型模拟一步
        t_span = torch.tensor([0.0, dt], device=device)
        trajectory = physics_model.simulate(t_span, current_state, u=action, method='rk4')
        next_state = trajectory[-1]

        states.append(current_state)
        actions.append(action)
        next_states.append(next_state)

        current_state = next_state

    states = torch.stack(states)
    actions = torch.tensor(actions, device=device, dtype=torch.float32)
    next_states = torch.stack(next_states)

    return states, actions, next_states


def test_world_model():
    """
    测试 PIRL 世界模型

    流程:
    1. 创建物理模型和世界模型
    2. 生成训练数据（WC轨迹）
    3. 训练100步
    4. 验证性能
    """
    print("=" * 70)
    print("Testing PIRL World Model")
    print("=" * 70)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 1. 创建物理模型
    from src.envs.wilson_cowan import WilsonCowanODE

    physics_model = WilsonCowanODE(device=device)
    print("\n[OK] Physics model (Wilson-Cowan) created")

    # 2. 创建世界模型
    world_model = PIRLWorldModel(
        physics_model=physics_model,
        hidden_dim=64,
        physics_weight=0.1,
        device=device
    )
    print("[OK] PIRL world model created")

    # 打印残差网络初始权重统计
    residual_params = []
    for param in world_model.residual_net.parameters():
        residual_params.extend(param.data.cpu().numpy().flatten())
    residual_params = np.array(residual_params)
    print(f"\nResidual network initialization:")
    print(f"  Weight mean: {residual_params.mean():.6f}")
    print(f"  Weight std: {residual_params.std():.6f}")
    print(f"  Weight range: [{residual_params.min():.6f}, {residual_params.max():.6f}]")

    # 3. 生成训练数据
    print("\n" + "-" * 70)
    print("Generating training data...")

    n_trajectories = 10
    n_steps_per_traj = 100
    dt = 0.001

    all_states = []
    all_actions = []
    all_next_states = []

    for i in range(n_trajectories):
        states, actions, next_states = generate_wc_trajectory(
            physics_model=physics_model,
            n_steps=n_steps_per_traj,
            dt=dt,
            device=device
        )
        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)

    # 合并所有轨迹
    train_states = torch.cat(all_states, dim=0)
    train_actions = torch.cat(all_actions, dim=0)
    train_next_states = torch.cat(all_next_states, dim=0)

    print(f"[OK] Generated {n_trajectories} trajectories")
    print(f"  Total samples: {len(train_states)}")
    print(f"  State range: E=[{train_states[:, 0].min():.3f}, {train_states[:, 0].max():.3f}], "
          f"I=[{train_states[:, 1].min():.3f}, {train_states[:, 1].max():.3f}]")

    # 4. 训练世界模型
    print("\n" + "-" * 70)
    print("Training world model...")

    optimizer = optim.Adam(world_model.residual_net.parameters(), lr=1e-3)
    n_epochs = 100
    batch_size = 128

    # 训练前的初始性能
    with torch.no_grad():
        initial_losses = world_model.compute_loss(
            train_states, train_actions, train_next_states, dt
        )
        print(f"\nInitial performance (before training):")
        print(f"  Prediction MSE: {initial_losses['prediction'].item():.6f}")
        print(f"  Physics loss: {initial_losses['physics'].item():.6f}")
        print(f"  Residual norm: {initial_losses['residual_norm'].item():.6f}")

    # 训练循环
    for epoch in range(n_epochs):
        # 随机打乱数据
        perm = torch.randperm(len(train_states))

        epoch_losses = []

        # Mini-batch训练
        for i in range(0, len(train_states), batch_size):
            batch_idx = perm[i:i+batch_size]

            batch_states = train_states[batch_idx]
            batch_actions = train_actions[batch_idx]
            batch_next_states = train_next_states[batch_idx]

            losses = world_model.train_step(
                batch_states, batch_actions, batch_next_states,
                optimizer, dt
            )

            epoch_losses.append(losses)

        # 计算平均损失
        avg_losses = {
            k: np.mean([loss[k] for loss in epoch_losses])
            for k in epoch_losses[0].keys()
        }

        # 每20轮打印一次
        if (epoch + 1) % 20 == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}:")
            print(f"  Prediction MSE: {avg_losses['prediction']:.6f}")
            print(f"  Physics loss: {avg_losses['physics']:.6f}")
            print(f"  Residual norm: {avg_losses['residual_norm']:.6f}")

    print("\n[OK] Training complete!")

    # 5. 最终评估
    print("\n" + "-" * 70)
    print("Final Evaluation")
    print("-" * 70)

    with torch.no_grad():
        final_losses = world_model.compute_loss(
            train_states, train_actions, train_next_states, dt
        )

        # 计算预测误差
        pred_next_states = world_model.predict_next_state(train_states, train_actions, dt)
        mse = torch.mean((pred_next_states - train_next_states) ** 2).item()
        mae = torch.mean(torch.abs(pred_next_states - train_next_states)).item()

        print(f"\n1. Prediction Performance:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   Status: {'PASS' if mse < 0.01 else 'FAIL'} (target: MSE < 0.01)")

        print(f"\n2. Residual Network Statistics:")
        _, _, residuals = world_model.predict_derivative(train_states, train_actions)
        residual_norm = torch.mean(torch.norm(residuals, dim=-1)).item()
        print(f"   Average output norm: {residual_norm:.6f}")
        print(f"   Max residual: {torch.max(torch.abs(residuals)).item():.6f}")

        print(f"\n3. Physics Constraint Satisfaction:")
        print(f"   Physics loss: {final_losses['physics'].item():.6f}")

        # 计算物理模型与完整模型的相对偏差
        total_deriv, physics_deriv, _ = world_model.predict_derivative(
            train_states, train_actions
        )
        relative_deviation = torch.mean(
            torch.norm(total_deriv - physics_deriv, dim=-1) /
            (torch.norm(physics_deriv, dim=-1) + 1e-8)
        ).item()
        print(f"   Relative deviation from physics: {relative_deviation:.4f}")
        print(f"   Status: Physics-informed (residual provides corrections)")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

    return world_model, final_losses


if __name__ == "__main__":
    test_world_model()
