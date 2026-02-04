# PhIHP Agent 实现文档

## 概述

Physics-Informed Hierarchical Planning (PhIHP) Agent 是一个结合物理先验的分层规划强化学习代理，使用 Actor-Critic 架构，并通过世界模型进行想象展开来提高样本效率。

## 核心架构

### 1. Actor Network

**功能**: 确定性策略，输出动作

**结构**:
```python
输入: 状态 [E, I] (2维)
网络: 2 → 128 → 128 → 1
激活: ReLU → ReLU → Tanh
输出: 动作 u ∈ [-action_limit, action_limit]
```

**特点**:
- 确定性策略（TD3风格）
- Tanh输出缩放到动作范围
- 简单但有效的结构

### 2. Critic Network (Twin Q-Networks)

**功能**: 估计状态-动作价值 Q(s, a)

**结构**:
```python
输入: [状态, 动作] ([E, I, u], 3维)
Q1网络: 3 → 128 → 128 → 1
Q2网络: 3 → 128 → 128 → 1  (独立参数)
激活: ReLU
输出: Q值
```

**特点**:
- Twin Critic 减少过估计（TD3技巧）
- 独立的两个Q网络
- 取最小值作为目标

### 3. 世界模型集成

**功能**: PIRL 世界模型用于想象展开

**流程**:
```
当前状态 s
  ↓ Actor
动作 a
  ↓ 世界模型
想象下一状态 s'
  ↓ 循环 n 步
想象轨迹 τ_imagine
```

**优势**:
- 提高样本效率
- 物理一致的想象
- 混合真实+想象数据训练

### 4. 安全约束层

**功能**: 确保控制安全

**约束类型**:

#### a. 绝对值约束
```python
|u| ≤ u_max
```

#### b. 变化率约束
```python
|du/dt| ≤ du_max
```

#### c. 状态依赖约束
```python
if E > 0.8:  # 高兴奋态
    u ≤ 0.5  # 限制正向刺激
```

**实现**:
```python
class SafetyLayer:
    def forward(action, state):
        # 1. 限幅
        action = clamp(action, -u_max, u_max)

        # 2. 变化率限制
        if last_action is not None:
            du = action - last_action
            du = clamp(du, -du_max*dt, du_max*dt)
            action = last_action + du

        # 3. 状态依赖
        if E > threshold:
            action = clamp(action, -u_max, 0.5)

        return action
```

### 5. 奖励函数

**多目标奖励设计**:

```python
R_total = R_task + R_energy + R_oscillation + R_safety
```

#### R_task: 任务奖励
```python
R_task = -((E - E_target)^2)
```
鼓励达到目标状态（如睡眠态 E=0.15）

#### R_energy: 能量惩罚
```python
R_energy = -0.1 * u^2
```
减少控制能量消耗

#### R_oscillation: 振荡惩罚
```python
R_oscillation = -0.5 * (dE/dt)^2
```
惩罚剧烈变化，鼓励平稳控制

#### R_safety: 安全奖励
```python
if E < 0.05 or E > 0.95:
    R_safety = -10  # 严重惩罚危险状态
elif E < 0.1 or E > 0.9:
    R_safety = -1   # 轻度惩罚
else:
    R_safety = 0
```
Barrier function 防止进入危险区域

### 6. 经验回放

**功能**: 存储和采样经验

**结构**:
```python
buffer = deque(maxlen=100000)
experience = (state, action, reward, next_state, done)
```

**操作**:
- `push()`: 添加新经验
- `sample(batch_size)`: 随机采样批次
- FIFO 队列，自动丢弃旧经验

## 训练流程

### 1. 真实环境交互

```python
for episode in episodes:
    state = env.reset()

    for step in steps:
        # 选择动作（带探索噪声）
        action = actor(state) + noise
        action = safety_layer(action, state)

        # 执行动作
        next_state, _, done, _ = env.step(action)
        reward = compute_reward(state, action, next_state)

        # 存储经验
        buffer.push(state, action, reward, next_state, done)
```

### 2. 想象展开

```python
# 从真实状态开始想象
imagine_states, imagine_actions, imagine_rewards = imagine_trajectory(
    state, n_steps=5
)

# 使用想象数据增强训练
```

### 3. 混合数据更新

```python
# 真实数据
real_batch = buffer.sample(batch_size)

# 想象数据
imagine_batch = imagine_trajectory(real_batch.states, n_steps=5)

# 更新 Critic
for (state, action, reward, next_state) in real_batch:
    target_q = reward + γ * Q_target(next_state, actor_target(next_state))
    critic_loss += MSE(Q(state, action), target_q)

for (state, action, reward, next_state) in imagine_batch:
    # 类似计算
    critic_loss += imagination_weight * MSE(...)

critic_loss.backward()

# 更新 Actor（延迟更新，每2步）
if step % 2 == 0:
    actor_loss = -Q(state, actor(state)).mean()
    actor_loss.backward()
```

## 超参数

### 网络结构
```python
state_dim = 2           # [E, I]
action_dim = 1          # u
hidden_dim = 128        # 隐藏层
```

### RL 超参数
```python
gamma = 0.99            # 折扣因子
tau = 0.005             # 软更新系数
actor_lr = 1e-4         # Actor 学习率
critic_lr = 3e-4        # Critic 学习率
batch_size = 128        # 批次大小
buffer_size = 100000    # 经验池大小
```

### 想象参数
```python
imagination_steps = 5       # 想象展开步数
imagination_weight = 0.5    # 想象数据权重
```

### 安全参数
```python
u_max = 2.0             # 动作上限
du_max = 5.0            # 变化率上限
dt = 0.001              # 时间步长
E_high_threshold = 0.8  # 高E阈值
```

### 探索参数
```python
noise_scale = 0.1       # 探索噪声标准差
target_noise = 0.2      # 目标策略平滑噪声
noise_clip = 0.5        # 噪声裁剪
```

## 关键技术

### 1. TD3 (Twin Delayed DDPG)

**Twin Critics**:
- 两个独立Q网络
- 取最小值防止过估计
```python
q1, q2 = critic(state, action)
q = min(q1, q2)
```

**Delayed Policy Updates**:
- Critic 每步更新
- Actor 每2步更新
- 提高稳定性

**Target Policy Smoothing**:
- 目标动作添加噪声
- 平滑价值估计
```python
target_action = actor_target(next_state) + noise
target_action = clamp(target_action)
```

### 2. 物理信息想象

**世界模型预测**:
```python
s' = s + dt * (f_physics(s, a) + f_residual(s, a))
```

**想象轨迹生成**:
```
s0 → a0 → s1 → a1 → s2 → ... → sn
    ↓      ↓      ↓
    r0     r1     r2
```

**混合训练**:
```python
L_total = L_real + λ_imag * L_imagine
```

### 3. 软更新（Soft Update）

```python
θ_target = τ * θ + (1 - τ) * θ_target
```

缓慢更新目标网络，提高稳定性。

## 使用示例

### 基础用法

```python
from src.agents import PhIHPAgent
from src.envs import WilsonCowanEnv
from src.models import PIRLWorldModel

# 创建环境
env = WilsonCowanEnv(
    dt=0.001,
    max_steps=500,
    target_state=[0.15, 0.1]  # 睡眠态
)

# 创建世界模型
world_model = PIRLWorldModel(
    physics_model=env.model,
    device='cuda'
)

# 创建代理
agent = PhIHPAgent(
    state_dim=2,
    action_dim=1,
    world_model=world_model,
    device='cuda'
)

# 训练
for episode in range(n_episodes):
    state, _ = env.reset()
    agent.reset_safety_layer()

    for step in range(env.max_steps):
        action = agent.select_action(state, explore=True)
        next_state, _, done, _, _ = env.step(action)
        reward = agent.compute_reward(state, action, next_state)

        agent.replay_buffer.push(state, action, reward, next_state, done)

        if len(agent.replay_buffer) > agent.batch_size:
            agent.update()

        state = next_state
        if done:
            break
```

### 评估

```python
# 评估模式（无探索）
state, _ = env.reset()
agent.reset_safety_layer()

for step in range(env.max_steps):
    action = agent.select_action(state, explore=False)
    next_state, _, done, _, _ = env.step(action)

    state = next_state
    if done:
        break

print(f"Final state: E={state[0]:.3f}, target=0.15")
```

### 保存和加载

```python
# 保存
agent.save('models/phihp_agent.pth')

# 加载
agent.load('models/phihp_agent.pth')
```

## 性能指标

### 训练指标
- Episode reward
- Episode length
- Actor loss
- Critic loss
- Buffer utilization

### 评估指标
- Final state error: |E_final - E_target|
- Control energy: Σ u²
- Oscillation: Σ (dE/dt)²
- Safety violations: 违反约束次数

## 优势与局限

### ✓ 优势

1. **样本效率高**
   - 世界模型想象展开
   - 混合真实+想象数据

2. **物理一致性**
   - 基于WC方程的想象
   - 保证动力学合理性

3. **安全保障**
   - 多层安全约束
   - 状态依赖限制

4. **稳定训练**
   - TD3 技巧
   - 软更新

### ⚠️ 局限

1. **世界模型依赖**
   - 需要预训练世界模型
   - 模型误差会累积

2. **超参数敏感**
   - 需要调整奖励权重
   - 想象权重影响大

3. **计算开销**
   - 想象展开增加计算
   - Twin Critic 双倍参数

## 改进方向

### 1. 模型-预测控制 (MPC)
```python
# 使用想象展开做短期规划
actions = []
for h in range(horizon):
    action = optimize_over_imagination(state, h)
    actions.append(action)
return actions[0]  # 执行第一步
```

### 2. 不确定性估计
```python
# 估计世界模型不确定性
uncertainty = world_model.uncertainty(state, action)
reward -= penalty(uncertainty)  # 避免高不确定性区域
```

### 3. 层次化策略
```python
# 高层：目标选择
goal = high_level_policy(state)

# 低层：达到目标的控制
action = low_level_policy(state, goal)
```

### 4. 多步想象
```python
# 可变长度想象
n_steps = adaptive_horizon(state)  # 根据状态调整
```

## 文件位置

- **实现**: `src/agents/phihp_agent.py`
- **测试**: `examples/test_phihp_agent.py`
- **文档**: `docs/phihp_agent.md`

## 依赖项

```
torch >= 2.0.0
numpy
matplotlib
```

## 参考文献

1. **TD3**: Fujimoto et al. (2018). "Addressing Function Approximation Error in Actor-Critic Methods." ICML.

2. **World Models**: Ha & Schmidhuber (2018). "World Models." arXiv:1803.10122.

3. **Physics-Informed RL**: Gruber et al. (2021). "Physics-Informed Deep Reinforcement Learning." arXiv.

4. **Wilson-Cowan**: Wilson & Cowan (1972). "Excitatory and inhibitory interactions." Biophysical Journal.

## 总结

PhIHP Agent 成功结合了：
- ✓ Actor-Critic 架构
- ✓ 物理信息世界模型
- ✓ 想象展开规划
- ✓ 安全约束层
- ✓ 多目标奖励设计

为神经动力学控制提供了一个高效、安全、物理一致的强化学习解决方案。
