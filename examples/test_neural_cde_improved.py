"""
Improved Neural CDE Observer with better hyperparameters

改进版本的 Neural CDE 观测器测试
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.envs.wilson_cowan import WilsonCowanEnv
from src.models.neural_cde_observer import (
    NeuralCDEObserver,
    ObserverLoss,
    generate_observer_training_data
)


def test_improved_observer():
    """
    改进版测试：
    - 降低物理损失权重 0.01
    - 增加网络容量 hidden_dim=64
    - 降低观测噪声 0.005
    - 增加训练轮数 100
    - 学习率调度
    - 梯度裁剪
    """
    print("=" * 70)
    print("Testing Improved Neural CDE State Observer")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 创建环境
    env = WilsonCowanEnv(
        dt=0.001,
        max_steps=100,
        device=device,
        reward_type='none'
    )
    print("[OK] Environment created")

    # 生成训练数据（更大数据集，更小噪声）
    print("\n" + "-" * 70)
    print("Generating training data...")
    print("  Improvements: noise_std=0.005 (vs 0.01), n_traj=200 (vs 100)")

    times_train, obs_train, states_train, actions_train = generate_observer_training_data(
        env=env,
        n_trajectories=200,  # 增加数据
        n_steps=100,
        noise_std=0.005,     # 降低噪声
        device=device
    )

    print(f"[OK] Generated {len(times_train)} trajectories")

    # 创建改进的观测器
    observer = NeuralCDEObserver(
        input_dim=1,
        hidden_dim=64,       # 增加容量
        output_dim=2,
        interpolation='cubic',
        device=device
    )
    print(f"\n[OK] Observer created (hidden_dim=64, was 32)")

    # 创建损失函数
    loss_fn = ObserverLoss(
        physics_model=env.model,
        physics_weight=0.01,  # 降低物理约束权重
        device=device
    )
    print("[OK] Loss function (physics_weight=0.01, was 0.1)")

    # 训练配置
    print("\n" + "-" * 70)
    print("Training configuration:")
    print("  Optimizer: Adam(lr=1e-3, weight_decay=1e-5)")
    print("  Scheduler: CosineAnnealing")
    print("  Gradient clipping: max_norm=1.0")
    print("  Epochs: 100 (was 50)")
    print("  Batch size: 16")

    optimizer = optim.Adam(
        observer.parameters(),
        lr=1e-3,
        weight_decay=1e-5  # L2 正则化
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-5
    )

    n_epochs = 100
    batch_size = 16

    # 评估初始性能
    print("\nTraining...")
    with torch.no_grad():
        pred_states = observer(times_train, obs_train)
        initial_losses = loss_fn(pred_states, states_train, times_train, actions_train)
        print(f"\nInitial (epoch 0):")
        print(f"  Recon: {initial_losses['reconstruction'].item():.6f}")
        print(f"  Physics: {initial_losses['physics'].item():.6f}")

    # 训练循环
    train_losses = []
    best_loss = float('inf')

    for epoch in range(n_epochs):
        perm = torch.randperm(len(times_train))
        epoch_losses = []

        for i in range(0, len(times_train), batch_size):
            batch_idx = perm[i:i+batch_size]

            batch_times = times_train[batch_idx]
            batch_obs = obs_train[batch_idx]
            batch_states = states_train[batch_idx]
            batch_actions = actions_train[batch_idx]

            optimizer.zero_grad()
            pred_states = observer(batch_times, batch_obs)
            losses = loss_fn(pred_states, batch_states, batch_times, batch_actions)

            losses['total'].backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(observer.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_losses.append({k: v.item() for k, v in losses.items()})

        # 学习率调度
        scheduler.step()

        # 计算平均损失
        avg_losses = {
            k: np.mean([loss[k] for loss in epoch_losses])
            for k in epoch_losses[0].keys()
        }
        train_losses.append(avg_losses)

        # 保存最佳模型
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            best_epoch = epoch + 1

        # 打印进度
        if (epoch + 1) % 20 == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs} (lr={scheduler.get_last_lr()[0]:.6f}):")
            print(f"  Recon: {avg_losses['reconstruction']:.6f}")
            print(f"  Physics: {avg_losses['physics']:.6f}")

    print(f"\n[OK] Training complete! Best epoch: {best_epoch}")

    # 最终评估
    print("\n" + "-" * 70)
    print("Final Evaluation")
    print("-" * 70)

    with torch.no_grad():
        pred_states = observer(times_train, states_train)
        final_losses = loss_fn(pred_states, states_train, times_train, actions_train)

        E_mse = torch.mean((pred_states[:, :, 0] - states_train[:, :, 0]) ** 2).item()
        I_mse = torch.mean((pred_states[:, :, 1] - states_train[:, :, 1]) ** 2).item()
        E_mae = torch.mean(torch.abs(pred_states[:, :, 0] - states_train[:, :, 0])).item()
        I_mae = torch.mean(torch.abs(pred_states[:, :, 1] - states_train[:, :, 1])).item()

        print(f"\n1. Reconstruction Performance:")
        print(f"   E MSE: {E_mse:.6f}  MAE: {E_mae:.6f}")
        print(f"   I MSE: {I_mse:.6f}  MAE: {I_mae:.6f} (unobserved!)")
        print(f"   Overall MSE: {final_losses['reconstruction'].item():.6f}")

        print(f"\n2. Physics Constraint:")
        print(f"   Physics loss: {final_losses['physics'].item():.6f}")

        # 与初始版本比较
        print(f"\n3. Comparison with baseline:")
        print(f"   Baseline E MSE: 0.213 -> Improved: {E_mse:.6f}")
        print(f"   Baseline I MSE: 0.160 -> Improved: {I_mse:.6f}")
        improvement_E = (0.213 - E_mse) / 0.213 * 100
        improvement_I = (0.160 - I_mse) / 0.160 * 100
        print(f"   E improvement: {improvement_E:+.1f}%")
        print(f"   I improvement: {improvement_I:+.1f}%")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

    return observer, train_losses


if __name__ == "__main__":
    try:
        test_improved_observer()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install torchcde:")
        print("  pip install torchcde")
