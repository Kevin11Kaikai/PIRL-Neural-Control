# PIRL_Claude 项目总结与文献指南

---

## 一、四个 Phase 完成内容

### Phase 0–1：环境搭建与仿真环境

**完成内容**
- 项目结构搭建
- Wilson–Cowan 神经质量模型实现
- Gym 风格仿真环境封装

**技术要点**
- Python 工程化
- Conda 环境管理
- 极限环振荡（10 Hz Alpha 波）建模
- 强化学习接口标准化

**核心方程**

\[
\tau_E \cdot \frac{dE}{dt} = -E + S(w_{EE} \cdot E - w_{EI} \cdot I + P + u)
\]

\[
\tau_I \cdot \frac{dI}{dt} = -I + S(w_{IE} \cdot E - w_{II} \cdot I + Q)
\]

---

### Phase 2：观测器开发

**完成内容**
- Neural CDE 观测器实现
- 隐变量状态重构

**技术要点**
- 处理不规则时序与部分可观测数据
- 从可观测的 \(E(t)\) 推断隐藏状态 \(I(t)\)

**核心思想**
> 使用连续时间微分方程对稀疏 EEG 观测进行建模与状态重构。

---

### Phase 3：控制器开发

**完成内容**
- PINN 世界模型构建
- PhIHP Agent 设计与实现
- 多种基线控制器对比

**技术要点**
- 物理先验 + 神经网络残差（相比纯学习模型提升约 500×）
- Actor–Critic 架构
- CEM 规划与想象训练（Imagined Rollouts）

**基线控制器**
- PID
- Open-loop
- Bang-Bang

**关键发现**
> 世界模型效果极好，但 PhIHP Agent 由于奖励函数设计问题，最终学到了常数控制策略。

---

### Phase 4：实验与分析

**实验结果（状态控制对比）**
- Bang-Bang：-49
- PID：-2277
- Random：-2034

**失败分析**
- PhIHP Agent 收敛到常数策略 \(u = -2.0\)

**代码备份**
- GitHub 仓库：
  https://github.com/Kevinl1Kaikai/PIRL-Neural-Control

---

## 二、系统架构图（逻辑结构）

```
┌─────────────────────────────────────────────────────────────┐
│                    完整系统架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐     ┌──────────────┐     ┌────────────────┐   │
│  │ EEG观测 │────▶│ Neural CDE   │────▶│ 状态估计 [E,I] │   │
│  │ E(t)    │     │ 观测器       │     └───────┬────────┘   │
│  └─────────┘     └──────────────┘             │             │
│                                               ▼             │
│  ┌─────────┐     ┌──────────────┐     ┌────────────────┐   │
│  │ 刺激输出 │◀────│ 安全层       │◀────│ PhIHP Agent    │   │
│  │ u(t)    │     └──────────────┘     │ Actor + CEM    │   │
│  └─────────┘                          └───────┬────────┘   │
│       │                                       │             │
│       │         ┌──────────────────────────┐  │             │
│       └────────▶│ PINN 世界模型            │◀─┘             │
│                 │ F_physics + F_residual   │                │
│                 │ （想象训练）              │                │
│                 └──────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、参考文献与代码对应

1. **Wilson, H. R. & Cowan, J. D. (1972)**  
   *Excitatory and Inhibitory Interactions in Localized Populations of Model Neurons*  
   *Biophysical Journal*, 12(1), 1–24.  
   → `src/envs/wilson_cowan.py`, `src/envs/wilson_cowan_freq.py`

2. **Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018)**  
   *Neural Ordinary Differential Equations*  
   NeurIPS 2018.  
   → `torchdiffeq` 使用，`src/models/physics_informed.py` 中的 ODE 积分

3. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019)**  
   *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs*  
   *Journal of Computational Physics*, 378, 686–707.  
   → `src/models/physics_informed.py`

4. **Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020)**  
   *Neural Controlled Differential Equations for Irregular Time Series*  
   NeurIPS 2020.  
   → `src/models/neural_cde.py`

5. **Fujimoto, S., van Hoof, H., & Meger, D. (2018)**  
   *Addressing Function Approximation Error in Actor–Critic Methods*  
   ICML 2018.  
   → `src/agents/phihp_agent.py`, `src/agents/phihp_freq_agent.py`（TD3）

6. **PhIHP (2024)**  
   *Physics-Informed Model and Hybrid Planning for Efficient Dyna-Style Reinforcement Learning*  
   RLC / NeurIPS 2024.

