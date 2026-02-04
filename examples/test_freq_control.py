"""
Test Frequency Control Components

Verify that all new frequency control components work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.utils.frequency_estimator import FrequencyEstimator
from src.envs.wilson_cowan_freq import WilsonCowanFreqEnv
from src.agents.baselines_freq import (
    PIFrequencyController,
    OpenLoopFrequencyController,
    RandomFrequencyController,
)
from src.agents.phihp_freq_agent import PhIHPFreqAgent


def test_frequency_estimator():
    """Test frequency estimator on synthetic signal."""
    print("\n" + "=" * 80)
    print("TEST 1: Frequency Estimator")
    print("=" * 80)

    estimator = FrequencyEstimator(window_size=500, dt=0.001)

    # Generate 10Hz sine wave
    t = np.arange(0, 1.0, 0.001)
    signal = 0.5 + 0.3 * np.sin(2 * np.pi * 10.0 * t)

    # Feed samples
    f_estimates = []
    for E in signal:
        f_hat = estimator.update(E)
        f_estimates.append(f_hat)

    # Check final estimate
    final_f = f_estimates[-1]
    print(f"Generated: 10.0 Hz")
    print(f"Estimated: {final_f:.2f} Hz")
    print(f"Error: {abs(final_f - 10.0):.2f} Hz")

    if abs(final_f - 10.0) < 0.5:
        print("[OK] Frequency estimator working correctly\n")
        return True
    else:
        print("[FAIL] Frequency estimator error too large\n")
        return False


def test_environment():
    """Test frequency control environment."""
    print("=" * 80)
    print("TEST 2: Frequency Control Environment")
    print("=" * 80)

    env = WilsonCowanFreqEnv(
        f_target=9.0, f_natural=12.0, dt=0.001, episode_length=2.0
    )

    obs, info = env.reset()
    print(f"Initial observation: E={obs[0]:.3f}, f_hat={obs[1]:.2f} Hz")

    # Run 100 steps with fixed stimulation
    total_reward = 0.0
    for _ in range(100):
        action = np.array([9.0])  # Stimulate at target frequency
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"After 100 steps: E={obs[0]:.3f}, f_hat={obs[1]:.2f} Hz")
    print(f"Total reward: {total_reward:.2f}")

    if obs[0] > 0 and obs[0] < 1 and obs[1] > 0:
        print("[OK] Environment working correctly\n")
        return True
    else:
        print("[FAIL] Environment state out of bounds\n")
        return False


def test_baseline_controllers():
    """Test baseline frequency controllers."""
    print("=" * 80)
    print("TEST 3: Baseline Controllers")
    print("=" * 80)

    env = WilsonCowanFreqEnv(f_target=9.0, f_natural=12.0, episode_length=1.0)

    controllers = {
        "PI": PIFrequencyController(f_target=9.0),
        "OpenLoop": OpenLoopFrequencyController(f_target=9.0),
        "Random": RandomFrequencyController(seed=42),
    }

    all_passed = True

    for name, controller in controllers.items():
        obs, _ = env.reset()
        controller.reset()

        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 100:
            action = controller(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        print(f"{name:12s}: {steps} steps, reward={total_reward:.2f}")

        if steps > 0:
            print(f"  [OK] {name} working")
        else:
            print(f"  [FAIL] {name} failed")
            all_passed = False

    if all_passed:
        print("\n[OK] All baseline controllers working correctly\n")
    else:
        print("\n[FAIL] Some baseline controllers failed\n")

    return all_passed


def test_phihp_agent():
    """Test PhIHP agent initialization and action selection."""
    print("=" * 80)
    print("TEST 4: PhIHP Agent")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    agent = PhIHPFreqAgent(obs_dim=2, action_dim=1, device=device)

    # Test action selection
    obs = np.array([0.3, 10.0])
    action = agent.select_action(obs, explore=False)

    print(f"Observation: E={obs[0]:.3f}, f_hat={obs[1]:.2f} Hz")
    print(f"Action: f_stim={action[0]:.2f} Hz")

    if 4.0 <= action[0] <= 15.0:
        print("[OK] PhIHP agent action in valid range")
    else:
        print(f"[FAIL] PhIHP agent action out of range: {action[0]}")
        return False

    # Test training step
    env = WilsonCowanFreqEnv(f_target=9.0, f_natural=12.0, episode_length=0.5)

    obs, _ = env.reset()
    for _ in range(50):
        action = agent.select_action(obs, explore=True)
        next_obs, reward, done, truncated, info = env.step(action)
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            break

    # Try update
    if len(agent.replay_buffer) >= 32:
        metrics = agent.update(batch_size=32)
        print(f"Update metrics: {metrics}")
        print("[OK] PhIHP agent training step successful\n")
        return True
    else:
        print("[FAIL] Not enough samples for training\n")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("FREQUENCY CONTROL COMPONENT TESTS")
    print("=" * 80)

    results = {
        "Frequency Estimator": test_frequency_estimator(),
        "Environment": test_environment(),
        "Baseline Controllers": test_baseline_controllers(),
        "PhIHP Agent": test_phihp_agent(),
    }

    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{test_name:25s} {status}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n[OK] ALL TESTS PASSED - Ready to run experiments!\n")
        return 0
    else:
        print("\n[FAIL] SOME TESTS FAILED - Fix issues before running experiments\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
