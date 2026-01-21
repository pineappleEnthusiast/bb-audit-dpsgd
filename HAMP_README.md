# HAMP Defense: Testing Soft Label Privacy

This directory contains the HAMP (High-entropy Adversarial Membership Protection) defense implementation.

## Quick Start

Test the implementation with minimal settings:
```bash
./test_hamp.sh
```

## Usage

Train models and compute empirical epsilon:

```bash
# Baseline (no defense)
python hamp_defense.py --n_reps 200 --n_epochs 10 --gamma 0.0 --out exp_data/hamp

# With soft label defense
python hamp_defense.py --n_reps 200 --n_epochs 10 --gamma 0.8 --out exp_data/hamp
```

## Key Arguments

- `--gamma`: Soft label entropy parameter (0=hard labels, 0.8=20% ground-truth prob, 1=uniform)
- `--n_reps`: Number of model repetitions (default: 200)
- `--n_epochs`: Number of training epochs (default: 10)
- `--alpha`: Significance level for empirical epsilon (default: 0.05)
- `--delta`: Privacy parameter delta (default: 1e-5)

## How It Works

1. **Canary**: Uses the last training sample as the canary
2. **Two Worlds**:
   - "in" world: Train with canary, using soft labels (gamma)
   - "out" world: Train without canary, using soft labels (gamma)
3. **MIA**: Computes empirical epsilon using membership inference attack
4. **Defense Effectiveness**: Lower empirical epsilon = better privacy protection

## Output

Results are saved to `{out}/mnist_{model}_gamma{gamma}/`:
- `emp_eps_loss.npy`: Empirical epsilon
- `losses_in.npy`, `losses_out.npy`: Canary losses for each model
- `outputs_in.npy`, `outputs_out.npy`: Model outputs on canary
- `mia_scores.npy`, `mia_labels.npy`: MIA attack data
