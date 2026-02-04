"""
Physics-Informed Hierarchical Planning (PhIHP) Agent

结合物理先验的分层规划强化学习代理
- Actor-Critic 架构
- 世界模型想象展开
- 混合真实+想象数据训练
- 安全约束层
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import os


class ActorNetwork(nn.Module):
    """
    Actor 网络：输出确定性策略

    输入: 状态 [E, I]
    输出: 动作 u ∈ [-action_limit, action_limit]
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        hidden_dim: int = 128,
        action_limit: float = 2.0
    ):
        super().__init__()

        self.action_limit = action_limit

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出 [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: (batch, state_dim) or (state_dim,)

        Returns:
            action: (batch, action_dim) or (action_dim,)
        """
        raw_action = self.net(state)
        action = self.action_limit * raw_action
        return action


class CriticNetwork(nn.Module):
    """
    Critic 网络：估计 Q(s, a)

    输入: 状态 [E, I] + 动作 u
    输出: Q 值
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        hidden_dim: int = 128
    ):
        super().__init__()

        # Q1 网络
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 网络 (Twin Critic for TD3)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            state: (batch, state_dim)
            action: (batch, action_dim)

        Returns:
            q1, q2: (batch, 1)
        """
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """只计算 Q1（用于策略更新）"""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class SafetyLayer(nn.Module):
    """
    安全约束层

    确保控制输出满足：
    1. 绝对值约束: |u| <= u_max
    2. 变化率约束: |du/dt| <= du_max
    3. 状态依赖约束: 高 E 时限制正向刺激
    """

    def __init__(
        self,
        u_max: float = 2.0,
        du_max: float = 5.0,
        dt: float = 0.001,
        E_high_threshold: float = 0.8
    ):
        super().__init__()

        self.u_max = u_max
        self.du_max = du_max
        self.dt = dt
        self.E_high_threshold = E_high_threshold

        self.last_action = None

    def forward(
        self,
        action: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        应用安全约束

        Args:
            action: 原始动作
            state: 当前状态 [E, I]

        Returns:
            safe_action: 安全的动作
        """
        safe_action = action.clone()

        # 1. 绝对值约束
        safe_action = torch.clamp(safe_action, -self.u_max, self.u_max)

        # 2. 变化率约束
        if self.last_action is not None:
            du = safe_action - self.last_action
            du_limit = self.du_max * self.dt
            du = torch.clamp(du, -du_limit, du_limit)
            safe_action = self.last_action + du

        # 3. 状态依赖约束
        if state is not None:
            E = state[..., 0] if state.dim() > 1 else state[0]
            # 如果 E 很高，限制正向刺激
            high_E_mask = E > self.E_high_threshold
            if high_E_mask.any():
                if safe_action.dim() > 1:
                    safe_action[high_E_mask] = torch.clamp(
                        safe_action[high_E_mask],
                        -self.u_max,
                        0.5  # 限制正向刺激
                    )
                else:
                    if high_E_mask:
                        safe_action = torch.clamp(safe_action, -self.u_max, 0.5)

        self.last_action = safe_action.detach()

        return safe_action

    def reset(self):
        """重置内部状态"""
        self.last_action = None


class ReplayBuffer:
    """
    经验回放缓冲区

    存储 (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """采样批次"""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PhIHPAgent:
    """
    Physics-Informed Hierarchical Planning Agent

    结合世界模型的 Actor-Critic 代理
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        hidden_dim: int = 128,
        action_limit: float = 2.0,
        world_model=None,
        device: str = 'cpu',
        # 超参数
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        buffer_size: int = 100000,
        batch_size: int = 128,
        # 想象展开参数
        imagination_steps: int = 5,
        imagination_weight: float = 0.5,
        # 安全参数
        u_max: float = 2.0,
        du_max: float = 5.0,
        dt: float = 0.001
    ):
        """
        初始化 PhIHP 代理

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            action_limit: 动作范围限制
            world_model: 世界模型（PIRLWorldModel）
            device: 计算设备
            gamma: 折扣因子
            tau: 软更新系数
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            buffer_size: 经验池大小
            batch_size: 批次大小
            imagination_steps: 想象展开步数
            imagination_weight: 想象数据权重
            u_max: 动作绝对值限制
            du_max: 动作变化率限制
            dt: 时间步长
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.imagination_steps = imagination_steps
        self.imagination_weight = imagination_weight
        self.dt = dt

        # 网络
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dim, action_limit
        ).to(device)

        self.actor_target = ActorNetwork(
            state_dim, action_dim, hidden_dim, action_limit
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(
            state_dim, action_dim, hidden_dim
        ).to(device)

        self.critic_target = CriticNetwork(
            state_dim, action_dim, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 世界模型
        self.world_model = world_model

        # 安全层
        self.safety_layer = SafetyLayer(u_max, du_max, dt)

        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 训练统计
        self.total_steps = 0
        self.actor_losses = []
        self.critic_losses = []

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        noise_scale: float = 0.1
    ) -> np.ndarray:
        """
        选择动作

        Args:
            state: 当前状态
            explore: 是否探索
            noise_scale: 探索噪声标准差

        Returns:
            action: 动作
        """
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor)

            # 应用安全约束
            action = self.safety_layer(action, state_tensor)

            # 探索噪声
            if explore:
                noise = torch.randn_like(action) * noise_scale * self.action_limit
                action = action + noise
                action = torch.clamp(action, -self.action_limit, self.action_limit)

        return action.cpu().numpy()

    def compute_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """
        计算奖励

        R = R_task + R_energy + R_oscillation + R_safety

        Args:
            state: 当前状态 [E, I]
            action: 动作 u
            next_state: 下一状态 [E, I]

        Returns:
            reward: 总奖励
        """
        E, I = state[0], state[1]
        E_next, I_next = next_state[0], next_state[1]
        u = action[0] if isinstance(action, np.ndarray) else action

        # 1. 任务奖励：鼓励降低到睡眠态 (E ≈ 0.15)
        target_E = 0.15
        R_task = -((E_next - target_E) ** 2)

        # 2. 能量惩罚：减少控制能量
        R_energy = -0.1 * (u ** 2)

        # 3. 振荡惩罚：减少剧烈变化
        dE = E_next - E
        R_oscillation = -0.5 * (dE ** 2)

        # 4. 安全奖励：使用 barrier function
        # 惩罚危险状态（E 过高或过低）
        E_min, E_max = 0.0, 1.0
        if E_next < 0.05 or E_next > 0.95:
            R_safety = -10.0  # 严重惩罚
        elif E_next < 0.1 or E_next > 0.9:
            R_safety = -1.0   # 轻度惩罚
        else:
            R_safety = 0.0

        # 总奖励
        total_reward = R_task + R_energy + R_oscillation + R_safety

        return float(total_reward)

    def imagine_trajectory(
        self,
        state: torch.Tensor,
        n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用世界模型想象展开轨迹

        Args:
            state: 初始状态 (batch, state_dim)
            n_steps: 展开步数

        Returns:
            states: (batch, n_steps, state_dim)
            actions: (batch, n_steps, action_dim)
            rewards: (batch, n_steps)
        """
        if self.world_model is None:
            raise ValueError("World model not provided")

        batch_size = state.shape[0] if state.dim() > 1 else 1
        if state.dim() == 1:
            state = state.unsqueeze(0)

        states = [state]
        actions = []
        rewards = []

        current_state = state

        for step in range(n_steps):
            # Actor 选择动作
            with torch.no_grad():
                action = self.actor(current_state)

            # 世界模型预测下一状态
            next_state = self.world_model.predict_next_state(
                current_state, action, self.dt
            )

            # 计算奖励
            reward_list = []
            for i in range(batch_size):
                r = self.compute_reward(
                    current_state[i].detach().cpu().numpy(),
                    action[i].detach().cpu().numpy(),
                    next_state[i].detach().cpu().numpy()
                )
                reward_list.append(r)
            reward = torch.FloatTensor(reward_list).to(self.device)

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)

            current_state = next_state

        states = torch.stack(states[:-1], dim=1)  # (batch, n_steps, state_dim)
        actions = torch.stack(actions, dim=1)     # (batch, n_steps, action_dim)
        rewards = torch.stack(rewards, dim=1)     # (batch, n_steps)

        return states, actions, rewards

    def update(self) -> Dict[str, float]:
        """
        更新网络

        混合真实和想象数据训练

        Returns:
            losses: 损失字典
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # 采样真实经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # 转换为 tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # ==================== 更新 Critic ====================
        with torch.no_grad():
            # 目标动作
            next_actions = self.actor_target(next_states)

            # 添加噪声（TD3 技巧）
            noise = torch.randn_like(next_actions) * 0.2 * self.action_limit
            noise = torch.clamp(noise, -0.5 * self.action_limit, 0.5 * self.action_limit)
            next_actions = next_actions + noise
            next_actions = torch.clamp(next_actions, -self.action_limit, self.action_limit)

            # 目标 Q 值（取两个 Q 网络的最小值）
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # 计算目标
            y = rewards + (1 - dones) * self.gamma * target_q

        # 当前 Q 值
        current_q1, current_q2 = self.critic(states, actions)

        # Critic 损失
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

        # 如果有世界模型，添加想象数据
        if self.world_model is not None:
            # 想象展开
            imag_states, imag_actions, imag_rewards = self.imagine_trajectory(
                states, self.imagination_steps
            )

            # 计算想象数据的 Q 值
            for t in range(self.imagination_steps):
                state_t = imag_states[:, t, :]
                action_t = imag_actions[:, t, :]
                reward_t = imag_rewards[:, t].unsqueeze(1)

                # 下一状态
                if t < self.imagination_steps - 1:
                    next_state_t = imag_states[:, t + 1, :]
                    with torch.no_grad():
                        next_action_t = self.actor_target(next_state_t)
                        q1_next, q2_next = self.critic_target(next_state_t, next_action_t)
                        q_next = torch.min(q1_next, q2_next)
                        y_imag = reward_t + self.gamma * q_next
                else:
                    y_imag = reward_t

                # 当前 Q
                q1_imag, q2_imag = self.critic(state_t, action_t)

                # 添加想象损失
                critic_loss += self.imagination_weight * (
                    F.mse_loss(q1_imag, y_imag) + F.mse_loss(q2_imag, y_imag)
                )

        # 更新 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ==================== 更新 Actor ====================
        # 延迟策略更新（每2步更新一次）
        actor_loss = torch.tensor(0.0)
        if self.total_steps % 2 == 0:
            # 策略损失
            new_actions = self.actor(states)
            actor_loss = -self.critic.q1_forward(states, new_actions).mean()

            # 更新 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        self.total_steps += 1

        # 记录损失
        self.critic_losses.append(critic_loss.item())
        if actor_loss != 0.0:
            self.actor_losses.append(actor_loss.item())

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss != 0.0 else 0.0
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def reset_safety_layer(self):
        """重置安全层（每个 episode 开始时调用）"""
        self.safety_layer.reset()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


def test_phihp_agent():
    """
    测试 PhIHP Agent

    1. 创建环境和世界模型
    2. 创建 Agent
    3. 训练 200 episodes
    4. 评估性能
    5. 可视化结果
    """
    print("=" * 70)
    print("Testing PhIHP Agent")
    print("=" * 70)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 1. 创建环境
    from src.envs.wilson_cowan import WilsonCowanEnv

    env = WilsonCowanEnv(
        dt=0.001,
        max_steps=500,
        device=device,
        target_state=[0.15, 0.1],  # 睡眠态目标
        reward_type='none'  # 使用自定义奖励
    )
    print("[OK] Environment created (target: sleep state E=0.15)")

    # 2. 创建世界模型
    from src.models.world_model import PIRLWorldModel

    physics_model = env.model
    world_model = PIRLWorldModel(
        physics_model=physics_model,
        hidden_dim=64,
        physics_weight=0.1,
        device=device
    )
    print("[OK] World model created (PIRL)")

    # 3. 创建 Agent
    agent = PhIHPAgent(
        state_dim=2,
        action_dim=1,
        hidden_dim=128,
        action_limit=2.0,
        world_model=world_model,
        device=device,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=3e-4,
        batch_size=128,
        imagination_steps=5,
        imagination_weight=0.5
    )
    print("[OK] PhIHP Agent created")
    print(f"  Hidden dim: 128")
    print(f"  Imagination steps: 5")
    print(f"  Imagination weight: 0.5")

    # 4. 训练
    print("\n" + "-" * 70)
    print("Training...")

    n_episodes = 200
    episode_rewards = []
    episode_lengths = []
    best_reward = -float('inf')

    for episode in range(n_episodes):
        state, info = env.reset()
        agent.reset_safety_layer()

        episode_reward = 0
        episode_length = 0

        for step in range(env.max_steps):
            # 选择动作
            action = agent.select_action(state, explore=True, noise_scale=0.1)

            # 执行动作
            next_state, _, terminated, truncated, info = env.step(action)

            # 计算奖励
            reward = agent.compute_reward(state, action, next_state)

            # 存储经验
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 更新网络
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()

            episode_reward += reward
            episode_length += 1

            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_episode = episode + 1

        # 打印进度
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_length = np.mean(episode_lengths[-20:])
            print(f"\nEpisode {episode+1}/{n_episodes}:")
            print(f"  Avg reward (last 20): {avg_reward:.2f}")
            print(f"  Avg length (last 20): {avg_length:.0f}")
            print(f"  Buffer size: {len(agent.replay_buffer)}")

    print(f"\n[OK] Training complete!")
    print(f"  Best episode: {best_episode} (reward: {best_reward:.2f})")

    # 5. 评估
    print("\n" + "-" * 70)
    print("Evaluation")
    print("-" * 70)

    n_eval_episodes = 10
    eval_rewards = []
    eval_final_states = []

    for _ in range(n_eval_episodes):
        state, info = env.reset()
        agent.reset_safety_layer()

        episode_reward = 0

        for step in range(env.max_steps):
            action = agent.select_action(state, explore=False)
            next_state, _, terminated, truncated, info = env.step(action)
            reward = agent.compute_reward(state, action, next_state)

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        eval_rewards.append(episode_reward)
        eval_final_states.append(state)

    print(f"\nEvaluation Results ({n_eval_episodes} episodes):")
    print(f"  Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Mean final E: {np.mean([s[0] for s in eval_final_states]):.3f}")
    print(f"  Target E: 0.15")
    print(f"  Mean error: {abs(np.mean([s[0] for s in eval_final_states]) - 0.15):.3f}")

    # 6. 可视化
    print("\n" + "-" * 70)
    print("Generating visualizations...")

    os.makedirs('figures', exist_ok=True)

    # 训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 回合奖励
    axes[0, 0].plot(episode_rewards, alpha=0.6)
    axes[0, 0].plot(np.convolve(episode_rewards, np.ones(20)/20, mode='valid'), 'r-', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].set_title('Training: Episode Rewards')
    axes[0, 0].grid(True, alpha=0.3)

    # 回合长度
    axes[0, 1].plot(episode_lengths, alpha=0.6)
    axes[0, 1].plot(np.convolve(episode_lengths, np.ones(20)/20, mode='valid'), 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length')
    axes[0, 1].set_title('Training: Episode Lengths')
    axes[0, 1].grid(True, alpha=0.3)

    # Actor 损失
    if len(agent.actor_losses) > 0:
        axes[1, 0].plot(agent.actor_losses, alpha=0.6)
        if len(agent.actor_losses) > 50:
            axes[1, 0].plot(
                np.convolve(agent.actor_losses, np.ones(50)/50, mode='valid'),
                'r-', linewidth=2
            )
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Actor Loss')
        axes[1, 0].set_title('Training: Actor Loss')
        axes[1, 0].grid(True, alpha=0.3)

    # Critic 损失
    if len(agent.critic_losses) > 0:
        axes[1, 1].plot(agent.critic_losses, alpha=0.6)
        if len(agent.critic_losses) > 100:
            axes[1, 1].plot(
                np.convolve(agent.critic_losses, np.ones(100)/100, mode='valid'),
                'r-', linewidth=2
            )
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Critic Loss')
        axes[1, 1].set_title('Training: Critic Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

    plt.tight_layout()
    save_path = 'figures/phihp_training.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Training curves saved: {save_path}")
    plt.close()

    # 评估轨迹
    print("\nGenerating evaluation trajectory...")
    state, info = env.reset()
    agent.reset_safety_layer()

    states_traj = [state]
    actions_traj = []

    for step in range(env.max_steps):
        action = agent.select_action(state, explore=False)
        next_state, _, terminated, truncated, info = env.step(action)

        states_traj.append(next_state)
        actions_traj.append(action[0])

        state = next_state

        if terminated or truncated:
            break

    states_traj = np.array(states_traj)
    actions_traj = np.array(actions_traj)
    times = np.arange(len(states_traj)) * env.dt

    # 绘制轨迹
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 状态轨迹
    axes[0].plot(times, states_traj[:, 0], 'b-', label='E (Excitatory)', linewidth=2)
    axes[0].plot(times, states_traj[:, 1], 'r-', label='I (Inhibitory)', linewidth=2)
    axes[0].axhline(y=0.15, color='g', linestyle='--', label='Target E=0.15', linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Activity')
    axes[0].set_title('Evaluation: State Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 动作轨迹
    axes[1].plot(times[:-1], actions_traj, 'k-', linewidth=2)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Control Input u')
    axes[1].set_title('Evaluation: Control Actions')
    axes[1].grid(True, alpha=0.3)

    # 相空间
    axes[2].plot(states_traj[:, 0], states_traj[:, 1], 'b-', linewidth=2, alpha=0.7)
    axes[2].plot(states_traj[0, 0], states_traj[0, 1], 'go', markersize=10, label='Start')
    axes[2].plot(states_traj[-1, 0], states_traj[-1, 1], 'ro', markersize=10, label='End')
    axes[2].plot(0.15, 0.1, 'g*', markersize=15, label='Target')
    axes[2].set_xlabel('E (Excitatory)')
    axes[2].set_ylabel('I (Inhibitory)')
    axes[2].set_title('Evaluation: Phase Space')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'figures/phihp_evaluation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Evaluation trajectory saved: {save_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

    return agent


if __name__ == "__main__":
    test_phihp_agent()
