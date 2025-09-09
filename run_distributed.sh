#!/bin/bash

# Get the list of nodes
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODES=($NODES)  # Convert to array
MASTER_ADDR=${NODES[0]}  # First node is the master
MASTER_PORT=12345
GPUS_PER_NODE=4  # Adjust based on your nodes
NNODES=${#NODES[@]}  # Number of nodes

# For each node, launch the training script
for ((i=0; i<NNODES; i++)); do
    NODE_RANK=$i
    NODE_NAME=${NODES[$i]}
    
    # Launch on each node
    srun --nodes=1 --ntasks=1 --nodelist=$NODE_NAME \
        python -m torch.distributed.run \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        audit_model.py \
        --data_name mnist \
        --model_name cnn \
        --lr 1.33e-4 \
        --epsilon 10 \
        --fixed_init \
        --out debug \
        --block_size 4096 \
        --target_type blank \
        --fit_world_only in \
        --n_reps 2 \
        --batch_size 4096 \
        &
done

# Wait for all background processes to complete
wait