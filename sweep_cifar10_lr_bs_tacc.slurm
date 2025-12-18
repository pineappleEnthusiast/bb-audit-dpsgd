#!/bin/bash
#SBATCH -J sweep_cifar10_lr_bs     # Job name
#SBATCH -o sweep_cifar10_lr_bs.o%j # Output file (%j = job ID)
#SBATCH -e sweep_cifar10_lr_bs.e%j # Error file
#SBATCH -p gpu-a100                # Partition (queue) name
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 1                       # Number of tasks (MPI ranks)
#SBATCH -t 20:00:00                # Run time (hh:mm:ss)
#SBATCH -A CCR23048                # Allocation/Project ID (required on TACC)
#SBATCH --mail-user=saloni.a.modi@utexas.edu

# Print some info
echo "Running on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

set -e

cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd

cd bb-audit-dpsgd

DATA_NAME=cifar10
MODEL_NAME=cnn
AUG_MULT=1
N_REPS=2
N_EPOCHS=100
SEED=0
EPSILON=10.0
MAX_GRAD_NORM=1
MAX_PHYSICAL_BATCH_SIZE=1000
FIT_WORLD_ONLY=in

OPTIMIZERS=(sgd adam)

LRS=(0.5 1 3 5 7 9)
BATCH_SIZES=(500 1000 2000 4000 8000)

N_OPTIMIZERS=${#OPTIMIZERS[@]}
N_LRS=${#LRS[@]}
N_BS=${#BATCH_SIZES[@]}
N_COMBOS=$((N_OPTIMIZERS * N_LRS * N_BS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_COMBOS" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) >= N_COMBOS ($N_COMBOS)"
  exit 1
fi

OPT_IDX=$((SLURM_ARRAY_TASK_ID / (N_LRS * N_BS)))
REM=$((SLURM_ARRAY_TASK_ID % (N_LRS * N_BS)))
LR_IDX=$((REM / N_BS))
BS_IDX=$((REM % N_BS))

OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
LR=${LRS[$LR_IDX]}
BATCH_SIZE=${BATCH_SIZES[$BS_IDX]}

OUT_DIR="hp_tune/${DATA_NAME}_${MODEL_NAME}_lr${LR}_bs${BATCH_SIZE}_ep${N_EPOCHS}_eps${EPSILON}_${OPTIMIZER}"

echo "Running: data=${DATA_NAME} model=${MODEL_NAME} lr=${LR} batch_size=${BATCH_SIZE} out=${OUT_DIR}"

python opacus_audit.py \
  --data_name ${DATA_NAME} \
  --model_name ${MODEL_NAME} \
  --aug_mult ${AUG_MULT} \
  --n_reps ${N_REPS} \
  --n_epochs ${N_EPOCHS} \
  --lr ${LR} \
  --batch_size ${BATCH_SIZE} \
  --optimizer ${OPTIMIZER} \
  --seed ${SEED} \
  --out ${OUT_DIR} \
  --epsilon ${EPSILON} \
  --max_grad_norm ${MAX_GRAD_NORM} \
  --max_physical_batch_size ${MAX_PHYSICAL_BATCH_SIZE} \
  --fit_world_only ${FIT_WORLD_ONLY}
