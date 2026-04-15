# SLURM Job Specification

This document describes the SLURM job format used to run experiments on TACC (Texas Advanced Computing Center) with the `bb_audit_dpsgd` conda environment.

---

## Environment

- **Cluster**: TACC (uses `$SCRATCH` for working directory)
- **Partition**: `gh`
- **CUDA module**: `cuda/12.4`
- **Conda env**: `bb_audit_dpsgd`
- **Allocation IDs**: `ASC25081`, `ASC25102`

---

## SBATCH Header Parameters

| Flag | Description | Example values |
|------|-------------|----------------|
| `-J` | Job name | `my_experiment` |
| `-o` | Stdout file | `my_job.o%j` or `status/name_%A_%a.o` |
| `-e` | Stderr file | `my_job.e%j` or `status/name_%A_%a.e` |
| `-p` | Partition | `gh` |
| `-N` | Number of nodes | `1`, `5`, `20` |
| `-n` | Total tasks (= nodes for 1 GPU/node) | `1`, `5`, `20` |
| `--ntasks-per-node` | Tasks per node | `1` |
| `-t` | Wall time limit | `03:00:00`, `4:00:00`, `15:00:00` |
| `-A` | Account/allocation | `ASC25081` |
| `--array` | Job array index range | `0-9`, `0-29` |
| `--mail-user` | Email for notifications | `saloni.a.modi@utexas.edu` |

For array jobs, `%A` is the master job ID and `%a` is the array task index.

---

## Environment Setup Boilerplate

Every job starts with the same setup block:

```bash
module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd
```

---

## Job Patterns

### Pattern 1: Single-Node Job

For quick tests or non-distributed runs using `audit_model.py`.

```bash
#!/bin/bash
#SBATCH -J my_job
#SBATCH -o my_job.o%j
#SBATCH -e my_job.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A ASC25081

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

python3 audit_model.py \
    --data_name mnist \
    --model_name cnn \
    --lr 3 \
    --epsilon 10 \
    --fixed_init \
    --out my_output \
    --block_size 4000 \
    --target_type blank \
    --n_reps 400 \
    --batch_size 4000 \
    --defense
```

---

### Pattern 2: Multi-Node Distributed Job

For production runs using `parallel_audit_model.py` with `torchrun` across N nodes (typically 20).

```bash
#!/bin/bash
#SBATCH -J my_distributed_job
#SBATCH -o my_distributed_job.o%j
#SBATCH -e my_distributed_job.e%j
#SBATCH -p gh
#SBATCH -N 20
#SBATCH -n 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -A ASC25081

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  RANK=$SLURM_PROCID;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name cifar10 \
    --model_name cnn \
    --n_reps 400 \
    --n_epochs 100 \
    --lr 3 \
    --batch_size 3125 \
    --block_size 3125 \
    --seed 0 \
    --epsilon 10 \
    --delta 1e-5 \
    --max_grad_norm 1 \
    --aug_mult 1 \
    --fixed_init \
    --holdout_audit \
    --out my_output'
```

---

### Pattern 3: Job Array (Multi-Node per Task)

For running many experiments in parallel. Each array task gets its own set of N nodes and runs `parallel_audit_model.py`.

```bash
#!/bin/bash
#SBATCH -J my_array_job
#SBATCH -o status/my_array_%A_%a.o
#SBATCH -e status/my_array_%A_%a.e
#SBATCH -p gh
#SBATCH -N 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-N   # N+1 experiments total

module load cuda/12.4

set -e
cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd
cd bb-audit-dpsgd

mkdir -p status
mkdir -p my_output_parent

# Define experiments as pipe-separated strings
# Format: "dataset|model|lr|batch_size|block_size|epsilon|aug_mult|defense|output_name"
experiments=(
    "mnist|cnn|3|4000|4000|10.0|1|false|mnist_no_defense"
    "mnist|cnn|3|4000|4000|10.0|1|true|mnist_defense"
    # add more rows...
)

IFS='|' read -r DATA_NAME MODEL_NAME LR BATCH_SIZE BLOCK_SIZE EPSILON AUG_MULT DEFENSE OUT_NAME \
    <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUT_NAME}"

# Build optional defense args
DEFENSE_ARGS=""
if [ "${DEFENSE}" = "true" ]; then
    DEFENSE_ARGS="--defense --defense_k 5 --defense_score_fn grad_norm --defense_score_norm linf --defense_apply_ascent"
fi

# Port offset per task to avoid collisions
srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=$((29500 + '"${SLURM_ARRAY_TASK_ID}"'));
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --n_reps 400 \
    --n_epochs 100 \
    --lr '"${LR}"' \
    --max_grad_norm 1 \
    --epsilon '"${EPSILON}"' \
    --delta 1e-5 \
    --target_type blank \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --seed 0 \
    --fixed_init \
    --aug_mult '"${AUG_MULT}"' \
    '"${DEFENSE_ARGS}"' \
    --holdout_audit \
    --out my_output_parent/'"${OUT_NAME}"''
```

**Key convention**: use `$((29500 + SLURM_ARRAY_TASK_ID))` as the `MASTER_PORT` to avoid port collisions across concurrent array tasks.

**Log output convention**: array job logs go in `status/` with `_%A_%a` suffixes. Create the directory before submitting: `mkdir -p status`.

---

## `parallel_audit_model.py` Argument Reference

| Argument | Type | Description |
|----------|------|-------------|
| `--data_name` | str | Dataset: `mnist`, `cifar10`, `purchase` |
| `--model_name` | str | Model: `cnn`, `mlp`, `wideresnet` |
| `--n_reps` | int | Number of audit repetitions (e.g. 400) |
| `--n_epochs` | int | Training epochs (e.g. 100) |
| `--lr` | float | Learning rate (e.g. `3` for MNIST/CIFAR, `10` for Purchase) |
| `--batch_size` | int | Training batch size (MNIST: 4000, CIFAR-10: 3125, Purchase: 12143) |
| `--block_size` | int | Audit block size (often = batch_size; smaller for WideResNet/augmult) |
| `--seed` | int | Random seed (typically `0`) |
| `--epsilon` | float | DP epsilon target (e.g. `6.0`, `8.0`, `10.0`) |
| `--delta` | float | DP delta (typically `1e-5`) |
| `--max_grad_norm` | float | DP gradient clipping norm (typically `1`) |
| `--aug_mult` | int | Augmentation multiplicity (1, 2, 4, 8, 16) |
| `--target_type` | str | Canary type: `blank`, `mislabeled`, `fgsm`, `gradient_ascent`, `clipbkd` |
| `--blank_alpha` | float | Interpolation weight for blank canary (0.0–1.0) |
| `--fixed_init` | flag | Use fixed model initialization across reps |
| `--holdout_audit` | flag | Hold out audit samples from training |
| `--n_df` | int | Dataset size override (Purchase: `153800`) |
| `--out` | str | Output directory path |
| `--defense` | flag | Enable gradient norm defense |
| `--defense_k` | int | Defense filter size (typically `5`) |
| `--defense_score_fn` | str | Defense scoring: `grad_norm` |
| `--defense_score_norm` | str | Defense norm: `linf` |
| `--defense_apply_ascent` | flag | Apply gradient ascent in defense |

### Dataset-specific hyperparameters

| Dataset | Model | LR | Batch size | Block size |
|---------|-------|----|------------|------------|
| MNIST | cnn | 3 | 4000 | 4000 |
| CIFAR-10 | cnn | 3 | 3125 | 3125 |
| CIFAR-10 + augmult=4 | cnn | 3 | 3125 | 2000 |
| CIFAR-10 | wideresnet | 3 | 3125 | 1000 |
| Purchase | mlp | 10 | 12143 | 12143 |

---

## Submitting Jobs

```bash
# Submit a single job
sbatch my_job.slurm

# Submit a job array
sbatch my_array.slurm

# Check job status
squeue -u $USER

# Cancel a job
scancel <job_id>
```
