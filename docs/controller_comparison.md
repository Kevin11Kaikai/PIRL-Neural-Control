# Controller Comparison Experiment

## 实验设计

### 目标
全面对比 PhIHP Agent 与传统控制器的性能，验证物理信息强化学习的优势。

### 参与控制器

#### 1. PhIHP Agent (Physics-Informed Hierarchical Planning)
- **类型**: 强化学习
- **特点**:
  - Actor-Critic 架构 (TD3)
  - 世界模型想象展开
  - 安全约束层
  - 多目标奖励优化
- **训练**: 50 episodes

#### 2. PID Controller
- **类型**: 经典反馈控制
- **参数**: Kp=2.0, Ki=0.5, Kd=0.1
- **特点**: 比例-积分-微分控制

#### 3. Bang-Bang Controller
- **类型**: 二值控制
- **参数**: threshold=0.05
- **特点**: 最大功率正负切换

#### 4. Open Loop Stimulator
- **类型**: 开环控制
- **参数**: 10Hz 正弦波
- **特点**: 无反馈，固定刺激模式

#### 5. Random Controller
- **类型**: 随机基线
- **特点**: 均匀分布随机动作

## 评估指标

### 主要指标

#### 1. Episode Reward
```
R = R_task + R_energy + R_oscillation + R_safety
```
综合性能指标，越高越好。

#### 2. Final Error
```
error = |E_final - E_target|
```
最终状态与目标的偏差，越低越好。

### 详细指标

#### 控制性能
1. **Mean Error**: 轨迹平均误差
2. **RMSE**: 均方根误差
3. **Settling Time**: 稳定时间（首次进入±5%并保持）
4. **Overshoot**: 超调量
5. **Undershoot**: 下冲量

#### 控制代价
1. **Control Energy**: ∫u²dt，控制能量
2. **Mean Absolute Control**: 平均绝对控制值
3. **Control Smoothness**: 控制平滑度（变化率）

#### 状态稳定性
1. **Oscillation**: 状态振荡幅度
2. **Convergence**: 收敛速度

## 实验流程

### Phase 1: 训练 PhIHP Agent
```
Episodes: 50
Max steps: 500 per episode
Batch size: 64
Exploration: Gaussian noise (σ=0.1)
```

### Phase 2: 评估所有控制器
```
Evaluation episodes: 20 each
Environment: same for all
Target state: E=0.15, I=0.1
Max steps: 500
```

### Phase 3: 统计分析
- 计算均值和标准差
- Mann-Whitney U 检验（非参数）
- Cohen's d 效应量
- 显著性水平：*** p<0.001, ** p<0.01, * p<0.05

### Phase 4: 可视化
1. **comparison_bar.png**: 6个关键指标对比柱状图
2. **trajectory_comparison.png**: 状态和动作轨迹
3. **phase_portrait_comparison.png**: 相空间轨迹

## 统计显著性检验

### Mann-Whitney U 检验
非参数检验，不假设数据正态分布：
- H0: 两组数据来自相同分布
- Ha: 两组数据来自不同分布

### 效应量 (Cohen's d)
```
d = (μ1 - μ2) / σ_pooled
```

解释：
- |d| < 0.2: 小效应
- 0.2 ≤ |d| < 0.5: 中效应
- 0.5 ≤ |d| < 0.8: 大效应
- |d| ≥ 0.8: 极大效应

## 预期结果

### 最佳场景 (PhIHP 成功)
```
排名:
1. PhIHP
2. Bang-Bang
3. PID
4. Open Loop
5. Random

PhIHP 优势:
- 最高奖励
- 最低误差
- 最平滑控制
- 最低能量消耗
```

### 基线场景 (Bang-Bang 最优)
```
排名:
1. Bang-Bang
2. PhIHP
3. PID
4. Open Loop
5. Random

分析:
- Bang-Bang 简单有效
- PhIHP 需要更多训练
- 或超参数需要调优
```

## 关键问题

### 1. PhIHP 是否优于 Bang-Bang？
- 统计显著性：p < 0.05?
- 效应量：|d| > 0.5?

### 2. PhIHP 的优势在哪里？
- 控制平滑度？
- 能量效率？
- 泛化能力？

### 3. 训练是否充分？
- 50 episodes 足够吗？
- 是否需要更多训练？

## 改进方向

### 如果 PhIHP 表现不佳

#### 超参数调整
```python
# 增加训练
n_episodes = 100-200

# 调整学习率
actor_lr = 5e-5
critic_lr = 1e-4

# 调整想象权重
imagination_weight = 0.1-1.0

# 调整探索噪声
noise_scale = 0.05-0.2
```

#### 奖励函数优化
```python
# 增加平滑度奖励
R_smoothness = -α * |du/dt|

# 调整权重
R_task_weight = 1.0
R_energy_weight = 0.01-0.5
```

#### 网络架构
```python
# 增加容量
hidden_dim = 256

# 更深的网络
n_layers = 4
```

## 生成的文件

### 报告
- `results/comparison_report.txt`: 详细文本报告

### 可视化
- `figures/comparison_bar.png`: 指标对比
- `figures/trajectory_comparison.png`: 轨迹对比
- `figures/phase_portrait_comparison.png`: 相空间对比

## 使用方法

### 运行实验
```bash
cd PIRL_claude
python examples/compare_all_controllers.py
```

### 预计时间
- 训练 PhIHP: ~5-10 分钟
- 评估所有控制器: ~2-3 分钟
- 总计: ~10-15 分钟

## 参考文献

1. **PID Control**: Åström & Hägglund (1995). "PID Controllers."
2. **Bang-Bang Control**: Pontryagin (1962). "Mathematical Theory of Optimal Processes."
3. **TD3**: Fujimoto et al. (2018). "Addressing Function Approximation Error in Actor-Critic Methods."
4. **Statistical Tests**: Mann & Whitney (1947). "On a test of whether one of two random variables is stochastically larger."
