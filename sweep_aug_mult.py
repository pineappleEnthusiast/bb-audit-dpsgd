#!/usr/bin/env python
"""
Sweep script to test different aug_mult configurations for CIFAR10 training.
Goal: Find configurations that achieve >= 70% test accuracy.
"""
import subprocess
import sys
from datetime import datetime

# =============================================================================
# CONFIGURATION - Edit these values to customize the sweep
# =============================================================================

AUG_MULT_VALUES = [1, 2, 4, 8, 16]
RUN_NON_PRIVATE = True  # Run one experiment without DP (epsilon=None)

# Training parameters
MODEL_NAME = "cnn"
N_REPS = 2
N_EPOCHS = 100
LR = 1.33e-4
BATCH_SIZE = 4000
EPSILON = 10.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
OPTIMIZER = "adam"
SEED = 0
OUT_DIR = "sweep_results"

# Optional parameters (set to None to disable)
MAX_PHYSICAL_BATCH_SIZE = 3000
EARLY_STOPPING = None
FIT_WORLD_ONLY = "in"

# =============================================================================


def get_max_physical_batch_size(aug_mult):
    """Get max physical batch size based on aug_mult."""
    if aug_mult <= 4:
        return 4000
    elif aug_mult == 8:
        return 3000
    else:  # aug_mult >= 16
        return 1500


def run_experiment(aug_mult, epsilon=EPSILON, run_name=None, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, model_name=MODEL_NAME):
    """Run opacus_audit.py with a specific aug_mult value."""
    if run_name is None:
        run_name = f"aug_mult_{aug_mult}"
    out_folder = f"{OUT_DIR}/{run_name}"
    
    cmd = [
        sys.executable, "opacus_audit.py",
        "--data_name", "cifar10",
        "--model_name", model_name,
        "--aug_mult", str(aug_mult),
        "--n_reps", str(N_REPS),
        "--n_epochs", str(n_epochs),
        "--lr", str(LR),
        "--batch_size", str(batch_size),
        "--optimizer", OPTIMIZER,
        "--seed", str(SEED),
        "--out", out_folder,
    ]
    
    # Add privacy parameters only if epsilon is set
    if epsilon is not None:
        cmd.extend(["--epsilon", str(epsilon)])
        cmd.extend(["--delta", str(DELTA)])
        cmd.extend(["--max_grad_norm", str(MAX_GRAD_NORM)])
    else:
        cmd.extend(["--max_grad_norm", "-1"])
    
    max_physical_batch_size = get_max_physical_batch_size(aug_mult)
    cmd.extend(["--max_physical_batch_size", str(max_physical_batch_size)])
    
    if EARLY_STOPPING is not None:
        cmd.extend(["--early_stopping", str(EARLY_STOPPING)])
    
    if FIT_WORLD_ONLY is not None:
        cmd.extend(["--fit_world_only", FIT_WORLD_ONLY])
    
    print(f"\n{'='*60}")
    print(f"Running experiment: aug_mult={aug_mult}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    print("CIFAR10 aug_mult Sweep")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing aug_mult values: {AUG_MULT_VALUES}")
    print(f"Config: model={MODEL_NAME}, epochs={N_EPOCHS}, lr={LR}, "
          f"batch_size={BATCH_SIZE}, epsilon={EPSILON}")
    
    results = {}
    
    # Run non-private baseline first if enabled
    if RUN_NON_PRIVATE:
        returncode = run_experiment(aug_mult=1, epsilon=None, run_name="non_private")
        results["non_private"] = "success" if returncode == 0 else f"failed (code {returncode})"
    
    # Run aug_mult sweep with DP
    for aug_mult in AUG_MULT_VALUES:
        returncode = run_experiment(aug_mult)
        results[f"aug_mult_{aug_mult}"] = "success" if returncode == 0 else f"failed (code {returncode})"
    
    # Run aug_mult=1 with 200 epochs
    returncode = run_experiment(aug_mult=1, epsilon=EPSILON, run_name="aug_mult_1_epochs_200", n_epochs=200)
    results["aug_mult_1_epochs_200"] = "success" if returncode == 0 else f"failed (code {returncode})"
    
    # Run aug_mult=1 with batch_size=2000
    returncode = run_experiment(aug_mult=1, epsilon=EPSILON, run_name="aug_mult_1_batch_2000", batch_size=2000)
    results["aug_mult_1_batch_2000"] = "success" if returncode == 0 else f"failed (code {returncode})"
    
    # Run with WideResNet
    returncode = run_experiment(aug_mult=1, epsilon=EPSILON, run_name="wideresnet", model_name="wideresnet")
    results["wideresnet"] = "success" if returncode == 0 else f"failed (code {returncode})"
    
    # Summary
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    for aug_mult, status in results.items():
        print(f"  aug_mult={aug_mult}: {status}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
