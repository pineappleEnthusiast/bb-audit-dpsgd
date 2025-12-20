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

echo "Beginning nested sweep over ${#OPTIMIZERS[@]} optimizers x ${#LRS[@]} lrs x ${#BATCH_SIZES[@]} batch_sizes"

for OPTIMIZER in "${OPTIMIZERS[@]}"; do
  for LR in "${LRS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      OUT_DIR="hp_tune/${DATA_NAME}_${MODEL_NAME}_lr${LR}_bs${BATCH_SIZE}_ep${N_EPOCHS}_eps${EPSILON}_${OPTIMIZER}"
      mkdir -p "${OUT_DIR}"

      echo ""
      echo "=================================================="
      echo "Running: data=${DATA_NAME} model=${MODEL_NAME} lr=${LR} batch_size=${BATCH_SIZE} optimizer=${OPTIMIZER} out=${OUT_DIR}"
      echo "=================================================="

      set +e
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
      rc=$?
      set -e

      if [ $rc -ne 0 ]; then
        echo "[WARN] Run failed (rc=$rc): optimizer=${OPTIMIZER} lr=${LR} batch_size=${BATCH_SIZE}"
      else
        echo "[OK] Completed: optimizer=${OPTIMIZER} lr=${LR} batch_size=${BATCH_SIZE}"
      fi
    done
  done
done

echo "Nested sweep complete."
