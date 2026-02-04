"""
可视化 PIRL 世界模型性能

生成图表展示:
1. 预测 vs 真实轨迹
2. 残差网络输出分析
3. 物理模型 vs 完整模型对比
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.envs.wilson_cowan import WilsonCowanODE
from src.models.world_model import PIRLWorldModel, generate_wc_trajectory


def visualize_pirl_performance():
    """生成性能可视化图表"""
    print("Generating PIRL World Model Visualizations...")
    print("=" * 70)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 创建模型
    physics_model = WilsonCowanODE(device=device)
    world_model = PIRLWorldModel(
        physics_model=physics_model,
        hidden_dim=64,
        physics_weight=0.1,
        device=device
    )

    # 生成测试轨迹
    print("\nGenerating test trajectory...")
    n_steps = 500
    dt = 0.001

    states, actions, next_states = generate_wc_trajectory(
        physics_model=physics_model,
        n_steps=n_steps,
        dt=dt,
        device=device
    )

    # 使用世界模型预测
    print("Computing predictions...")
    with torch.no_grad():
        # 完整模型预测
        pred_next_states = world_model.predict_next_state(states, actions, dt)

        # 获取导数分解
        total_derivs, physics_derivs, residuals = world_model.predict_derivative(
            states, actions
        )

        # 纯物理模型预测（不含残差）
        physics_pred_next_states = states + dt * physics_derivs

    # 转换为 numpy
    states_np = states.cpu().numpy()
    next_states_np = next_states.cpu().numpy()
    pred_next_states_np = pred_next_states.cpu().numpy()
    physics_pred_np = physics_pred_next_states.cpu().numpy()
    actions_np = actions.cpu().numpy()
    residuals_np = residuals.cpu().numpy()

    # 计算误差
    pred_errors = np.linalg.norm(pred_next_states_np - next_states_np, axis=1)
    physics_errors = np.linalg.norm(physics_pred_np - next_states_np, axis=1)

    # 创建图表
    print("Creating visualizations...")
    os.makedirs('figures', exist_ok=True)

    fig = plt.figure(figsize=(16, 12))

    # 1. E 状态轨迹对比
    ax1 = plt.subplot(3, 3, 1)
    time = np.arange(n_steps) * dt
    ax1.plot(time, states_np[:, 0], 'b-', label='Current State', alpha=0.7, linewidth=1.5)
    ax1.plot(time, next_states_np[:, 0], 'g-', label='True Next', alpha=0.7, linewidth=1.5)
    ax1.plot(time, pred_next_states_np[:, 0], 'r--', label='PIRL Pred', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('E (Excitatory)')
    ax1.set_title('Excitatory Population Prediction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. I 状态轨迹对比
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time, states_np[:, 1], 'b-', label='Current State', alpha=0.7, linewidth=1.5)
    ax2.plot(time, next_states_np[:, 1], 'g-', label='True Next', alpha=0.7, linewidth=1.5)
    ax2.plot(time, pred_next_states_np[:, 1], 'r--', label='PIRL Pred', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('I (Inhibitory)')
    ax2.set_title('Inhibitory Population Prediction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 动作序列
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(time, actions_np, 'k-', alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Action u')
    ax3.set_title('Control Input Sequence')
    ax3.grid(True, alpha=0.3)

    # 4. 预测误差对比
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(time, pred_errors, 'r-', label='PIRL Model', alpha=0.7, linewidth=1.5)
    ax4.plot(time, physics_errors, 'b--', label='Physics Only', alpha=0.7, linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Prediction Error (L2 norm)')
    ax4.set_title('Prediction Error Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # 5. 残差分析 - E 分量
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(time, residuals_np[:, 0], 'purple', alpha=0.7, linewidth=1.5)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Residual dE/dt')
    ax5.set_title('Residual Network Output (E)')
    ax5.grid(True, alpha=0.3)

    # 6. 残差分析 - I 分量
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(time, residuals_np[:, 1], 'orange', alpha=0.7, linewidth=1.5)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Residual dI/dt')
    ax6.set_title('Residual Network Output (I)')
    ax6.grid(True, alpha=0.3)

    # 7. 相空间：真实 vs 预测
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(next_states_np[:, 0], next_states_np[:, 1], 'g-',
             label='True', alpha=0.5, linewidth=2)
    ax7.plot(pred_next_states_np[:, 0], pred_next_states_np[:, 1], 'r--',
             label='PIRL Pred', alpha=0.5, linewidth=2)
    ax7.set_xlabel('E')
    ax7.set_ylabel('I')
    ax7.set_title('Phase Space Trajectory')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. 误差直方图
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(pred_errors, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax8.set_xlabel('Prediction Error')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Error Distribution')
    ax8.axvline(x=np.mean(pred_errors), color='blue', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(pred_errors):.6f}')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. 统计摘要
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    stats_text = f"""
    Performance Statistics
    ───────────────────────

    Prediction Error:
      Mean: {np.mean(pred_errors):.6f}
      Std:  {np.std(pred_errors):.6f}
      Max:  {np.max(pred_errors):.6f}

    Residual Statistics:
      E mean: {np.mean(residuals_np[:, 0]):.6f}
      E std:  {np.std(residuals_np[:, 0]):.6f}
      I mean: {np.mean(residuals_np[:, 1]):.6f}
      I std:  {np.std(residuals_np[:, 1]):.6f}

    Model Comparison:
      PIRL MSE:    {np.mean(pred_errors**2):.6f}
      Physics MSE: {np.mean(physics_errors**2):.6f}
      Improvement: {(1 - np.mean(pred_errors**2)/np.mean(physics_errors**2))*100:.2f}%

    Target: MSE < 0.01
    Status: {"PASS" if np.mean(pred_errors**2) < 0.01 else "FAIL"}
    """

    ax9.text(0.1, 0.9, stats_text, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax9.transAxes)

    plt.tight_layout()

    # 保存图表
    save_path = 'figures/pirl_performance.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to: {save_path}")

    plt.close()

    # 打印统计摘要
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"\nPrediction Error:")
    print(f"  Mean: {np.mean(pred_errors):.6f}")
    print(f"  Std:  {np.std(pred_errors):.6f}")
    print(f"  Max:  {np.max(pred_errors):.6f}")
    print(f"\nMSE: {np.mean(pred_errors**2):.6f}")
    print(f"Status: {'PASS' if np.mean(pred_errors**2) < 0.01 else 'FAIL'} (target: MSE < 0.01)")

    print("\nResidual Network Output:")
    print(f"  Average norm: {np.mean(np.linalg.norm(residuals_np, axis=1)):.6f}")
    print(f"  Max absolute: {np.max(np.abs(residuals_np)):.6f}")

    print("\nModel Comparison:")
    print(f"  PIRL MSE:    {np.mean(pred_errors**2):.6f}")
    print(f"  Physics MSE: {np.mean(physics_errors**2):.6f}")
    improvement = (1 - np.mean(pred_errors**2)/np.mean(physics_errors**2))*100
    print(f"  Improvement: {improvement:.2f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    visualize_pirl_performance()
