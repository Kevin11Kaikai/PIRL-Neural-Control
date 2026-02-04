# PIRL World Model 实现文档

## 概述

Physics-Informed Residual Learning (PIRL) 世界模型结合了 Wilson-Cowan 神经场方程的物理先验和数据驱动的残差学习，能够在保持物理一致性的同时学习数据中的偏差和不确定性。

## 核心架构

### 1. 残差网络 (`ResidualNetwork`)

**结构：**
- 输入层：3维 [E, I, u] (状态 + 动作)
- 隐藏层：64维，2层，ReLU激活
- 输出层：2维 [dE/dt修正, dI/dt修正]

**关键设计：**
```python
# 小权重初始化 (std=0.01)
# 确保初始时残差接近零，模型主要依赖物理先验
nn.init.normal_(m.weight, mean=0.0, std=0.01)
```

**作用：**
学习物理模型的修正项，补偿模型误差和未建模动力学。

### 2. PIRL 世界模型 (`PIRLWorldModel`)

**预测方程：**
```
ds/dt = f_physics(s, u) + f_residual(s, u)
       └─ WC模型 ─┘    └─ 残差网络 ─┘
```

**时间积分：**
```python
s_next = s + dt * (f_physics + f_residual)
```

## 损失函数

### 1. 预测损失 (L_pred)

```python
L_pred = ||s_pred - s_true||²
```

衡量模型预测的下一状态与真实状态的差异。

### 2. 物理损失 (L_physics)

```python
L_physics = ||(f_physics + f_residual) - f_physics||²
          = ||f_residual||²
```

约束残差不要偏离物理模型太远，保持物理一致性。

### 3. 总损失

```python
L_total = L_pred + λ * L_physics
```

其中 λ=0.1 是物理损失权重。

## 训练流程

### 数据生成

```python
generate_wc_trajectory(
    physics_model,
    n_steps=100,
    dt=0.001,
    action_sequence=None  # 随机动作
)
```

- 生成10条轨迹，每条100步
- 总计1000个训练样本
- 使用真实的 Wilson-Cowan 模型模拟

### 训练配置

```python
optimizer = Adam(lr=1e-3)
n_epochs = 100
batch_size = 128
```

## 测试结果

### 初始化验证

```
Residual network initialization:
  Weight mean: -0.000072
  Weight std: 0.009866
  Weight range: [-0.031112, 0.033778]
```

✓ 权重标准差约为0.01，符合设计要求

### 训练数据

```
Generated 10 trajectories
  Total samples: 1000
  State range: E=[0.006, 0.968], I=[0.015, 0.941]
```

✓ 覆盖了广泛的状态空间

### 训练过程

| Epoch | Prediction MSE | Physics Loss | Residual Norm |
|-------|---------------|--------------|---------------|
| 初始  | 0.000020      | 0.000000     | 0.000039      |
| 20    | 0.000020      | 0.000000     | 0.000007      |
| 40    | 0.000020      | 0.000000     | 0.000009      |
| 60    | 0.000020      | 0.000000     | 0.000012      |
| 80    | 0.000020      | 0.000000     | 0.000012      |
| 100   | 0.000020      | 0.000000     | 0.000030      |

**观察：**
- 预测损失保持稳定在极低水平 (2e-5)
- 物理损失接近零，表明模型保持了物理一致性
- 残差范数很小，说明物理模型已经足够准确

### 最终性能

#### 1. 预测性能

```
MSE: 0.000020
MAE: 0.002108
Status: PASS (目标: MSE < 0.01)
```

✓ **远超目标性能**（比目标好500倍）

#### 2. 残差网络统计

```
Average output norm: 0.000007
Max residual: 0.000006
```

✓ 残差极小，表明物理模型高度准确

#### 3. 物理约束满足度

```
Physics loss: 0.000000
Relative deviation from physics: 0.0000
```

✓ 完美保持物理一致性

## 实现亮点

### 1. 物理先验集成

```python
# 物理模型导数
physics_derivative = self.physics_model.forward(0.0, state, action)

# 残差修正
residual = self.residual_net(state, action)

# 组合预测
total_derivative = physics_derivative + residual
```

### 2. 残差初始化策略

```python
def _initialize_weights(self, std: float):
    """小权重初始化 → 初始残差≈0 → 依赖物理先验"""
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
```

### 3. 物理一致性约束

```python
L_physics = torch.mean((total_derivative - physics_derivative) ** 2)
```

限制残差网络不会产生物理上不合理的预测。

## 使用示例

### 训练世界模型

```python
from src.models import PIRLWorldModel, generate_wc_trajectory
from src.envs import WilsonCowanODE

# 创建物理模型
physics_model = WilsonCowanODE(device='cuda')

# 创建 PIRL 世界模型
world_model = PIRLWorldModel(
    physics_model=physics_model,
    hidden_dim=64,
    physics_weight=0.1,
    device='cuda'
)

# 生成训练数据
states, actions, next_states = generate_wc_trajectory(
    physics_model=physics_model,
    n_steps=1000,
    dt=0.001,
    device='cuda'
)

# 训练
optimizer = torch.optim.Adam(world_model.residual_net.parameters(), lr=1e-3)
losses = world_model.train_step(states, actions, next_states, optimizer, dt=0.001)
```

### 使用模型预测

```python
# 预测下一状态
state = torch.tensor([0.5, 0.3], device='cuda')
action = torch.tensor([0.1], device='cuda')

next_state = world_model.predict_next_state(state, action, dt=0.001)

# 获取导数分解
total_deriv, physics_deriv, residual = world_model.predict_derivative(state, action)
print(f"Physics contribution: {physics_deriv}")
print(f"Residual correction: {residual}")
```

## 性能分析

### 为什么残差这么小？

训练数据由物理模型生成，因此：
1. 数据完全符合 WC 方程
2. 不存在模型误差或噪声
3. 残差网络学到的是数值误差和舍入误差

### 真实场景中的优势

在真实神经数据上：
1. **模型误差**：WC模型是简化，真实大脑更复杂
2. **参数不确定性**：tau, w_ee等参数可能不准确
3. **未建模动力学**：外部输入、疲劳、适应等

此时残差网络将学习这些偏差，提供更准确的预测。

## 后续改进方向

### 1. 使用噪声数据

```python
# 添加观测噪声
noisy_next_states = next_states + 0.01 * torch.randn_like(next_states)

# 添加过程噪声（参数扰动）
perturbed_model = WilsonCowanODE(
    tau_e=tau_e + 0.001 * np.random.randn(),
    w_ee=w_ee + 0.5 * np.random.randn(),
    ...
)
```

### 2. 不确定性估计

```python
class BayesianResidualNetwork(nn.Module):
    """使用dropout或变分推断估计不确定性"""
    def forward(self, state, action):
        # MC dropout for uncertainty
        ...
```

### 3. 自适应物理权重

```python
# 训练早期：高物理权重 → 依赖先验
# 训练后期：低物理权重 → 数据驱动微调
physics_weight = max(0.01, 1.0 * (1 - epoch/n_epochs))
```

### 4. 在线学习

```python
def update_from_real_data(self, real_trajectory):
    """从真实交互数据增量更新模型"""
    ...
```

## 文件位置

- **实现**: `src/models/world_model.py`
- **测试**: `examples/test_pirl_world_model.py`
- **依赖**: `src/envs/wilson_cowan.py`

## 依赖项

```
torch >= 2.0
torchdiffeq
numpy
```

## 参考文献

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. Journal of Computational Physics.
2. Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. Biophysical Journal.
3. Ha, D., & Schmidhuber, J. (2018). World Models. arXiv preprint arXiv:1803.10122.

## 总结

PIRL 世界模型成功实现了：
- ✓ 物理先验与数据驱动学习的结合
- ✓ 残差网络的正确初始化（std=0.01）
- ✓ 双重损失函数（预测 + 物理）
- ✓ 极高的预测精度（MSE < 0.01目标达成）
- ✓ 完整的训练和测试流程

该模型为后续的强化学习代理提供了物理一致的动力学预测器，可用于模型预测控制(MPC)、基于模型的RL等任务。
