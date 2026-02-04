"""
Unified Experiment Entry Point

Supports two control modes:
1. State control: Control E to target value (original experiments)
2. Frequency control: Control oscillation frequency to target (new experiments)

Usage:
    # Run state control experiments
    python scripts/run_all_experiments.py --mode state

    # Run frequency control experiments
    python scripts/run_all_experiments.py --mode freq

    # Run specific frequency experiment
    python scripts/run_all_experiments.py --mode freq --experiment a_locking

    # Run both modes
    python scripts/run_all_experiments.py --mode all
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import subprocess
from datetime import datetime


def run_state_control_experiments(args):
    """
    Run state control experiments (original implementation).

    Uses compare_all_controllers.py or compare_controllers_quick.py
    """
    print("\n" + "=" * 80)
    print("RUNNING STATE CONTROL EXPERIMENTS")
    print("=" * 80 + "\n")

    if args.quick:
        script = "examples/compare_controllers_quick.py"
        print("Mode: Quick (10 training + 5 eval)")
    else:
        script = "examples/compare_all_controllers.py"
        print("Mode: Full (50 training + 20 eval)")

    cmd = [sys.executable, script]

    print(f"Executing: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("\n[OK] State control experiments completed successfully")
    else:
        print(f"\n[ERROR] State control experiments failed with code {result.returncode}")

    return result.returncode == 0


def run_frequency_control_experiments(args):
    """
    Run frequency control experiments (new implementation).

    Uses run_freq_control.py
    """
    print("\n" + "=" * 80)
    print("RUNNING FREQUENCY CONTROL EXPERIMENTS")
    print("=" * 80 + "\n")

    cmd = [
        sys.executable,
        "scripts/run_freq_control.py",
        "--n_train",
        str(args.n_train),
        "--n_eval",
        str(args.n_eval),
        "--device",
        args.device,
        "--output_dir",
        args.output_dir,
    ]

    if args.experiment != "all":
        cmd.extend(["--experiment", args.experiment])

    print(f"Executing: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("\n[OK] Frequency control experiments completed successfully")
    else:
        print(f"\n[ERROR] Frequency control experiments failed with code {result.returncode}")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified entry point for state and frequency control experiments"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["state", "freq", "all"],
        default="freq",
        help="Control mode: state (amplitude control), freq (frequency control), or all",
    )

    # State control options
    parser.add_argument(
        "--quick",
        action="store_true",
        help="[State mode] Use quick comparison (10+5 episodes)",
    )

    # Frequency control options
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        help="[Freq mode] Specific experiment: a_locking, b_drift, c_extended, d_wrong, or all",
    )
    parser.add_argument(
        "--n_train", type=int, default=50, help="[Freq mode] Training episodes for PhIHP"
    )
    parser.add_argument(
        "--n_eval", type=int, default=10, help="[Freq mode] Evaluation episodes"
    )

    # Common options
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/freq_control",
        help="Output directory for frequency control",
    )

    args = parser.parse_args()

    # Print configuration
    print("\n" + "=" * 80)
    print("PIRL: UNIFIED EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")

    if args.mode in ["state", "all"]:
        print(f"State control: {'Quick' if args.quick else 'Full'}")

    if args.mode in ["freq", "all"]:
        print(f"Frequency control experiments: {args.experiment}")
        print(f"Training: {args.n_train} episodes, Evaluation: {args.n_eval} episodes")

    print("=" * 80 + "\n")

    # Run experiments based on mode
    success = True

    if args.mode == "state":
        success = run_state_control_experiments(args)

    elif args.mode == "freq":
        success = run_frequency_control_experiments(args)

    elif args.mode == "all":
        # Run both
        state_success = run_state_control_experiments(args)
        freq_success = run_frequency_control_experiments(args)
        success = state_success and freq_success

    # Final summary
    print("\n" + "=" * 80)
    if success:
        print("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    else:
        print("✗ SOME EXPERIMENTS FAILED")
    print("=" * 80 + "\n")

    # Print result locations
    print("Results locations:")
    if args.mode in ["state", "all"]:
        print("  State control:")
        print("    - results/comparison_report.txt")
        print("    - figures/comparison_*.png")

    if args.mode in ["freq", "all"]:
        print("  Frequency control:")
        print(f"    - {args.output_dir}/")
        print(f"    - {args.output_dir}/figures/")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
