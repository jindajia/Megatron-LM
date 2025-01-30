#!/bin/bash

CHECKPOINT_PATH=$1
TENSOR_PARALLEL_SIZE=$2
PIPELINE_PARALLEL_SIZE=$3
MICRO_BATCH_SIZE=$4
GLOBAL_BATCH_SIZE=$5
MODEL_ARG_PATH=$6
TRAINING_ARG_PATH=$7
OUTPUT_DIR=$8
MEGATRON_PATH=$9
RUNNING_NODES=${10}
RUNNING_GPUS_PER_NODE=${11}
EXIT_INTERVAL=${12}
LOG_INTERVAL=${13}
TENSORBOARD_DIR=${14}
WANDB_DIR=${15}
WANDB_PROJECT=${16}
WANDB_NAME=${17}

# export NCCL_SOCKET_IFNAME=eno
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NNODES=$RUNNING_NODES  # Number of node you want to use, must be less than or equal to the number of nodes allocated by SLURM
GPUS_PER_NODE=$RUNNING_GPUS_PER_NODE # Number of gpus you want to use
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=6002
TOTAL_NODELIST=$(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')

selected_nodes=$(echo "$TOTAL_NODELIST" | tr ' ' '\n' | head -n "$NNODES" | tr '\n' ' ')

set -x

rank=0
for node in $selected_nodes; do
    echo "Executinging on $node with rank $rank and $NNODES nodes..."

    if (( rank == $NNODES-1 )); then
        echo "Rank: $rank, NNODES: $NNODES, run_in_background: FALSE"
        srun --nodes=1 --nodelist=$node --gres=gpu:$GPUS_PER_NODE bash ${SCRIPT_DIR}/common/container_starter.sh \
            "$NNODES" \
            "$GPUS_PER_NODE" \
            "$MASTER_ADDR" \
            "$MASTER_PORT" \
            $rank \
            "$CHECKPOINT_PATH" \
            "$MODEL_ARG_PATH" \
            "$TRAINING_ARG_PATH" \
            "$OUTPUT_DIR" \
            "$MEGATRON_PATH" \
            "$TENSOR_PARALLEL_SIZE" \
            "$PIPELINE_PARALLEL_SIZE" \
            "$MICRO_BATCH_SIZE" \
            "$GLOBAL_BATCH_SIZE" \
            "$EXIT_INTERVAL" \
            "$LOG_INTERVAL" \
            "$TENSORBOARD_DIR" \
            "$WANDB_DIR" \
            "$WANDB_PROJECT" \
            "$WANDB_NAME" \
            "$OMP_NUM_THREADS"
    else
        echo "Rank: $rank, NNODES: $NNODES, run_in_background: TRUE"
        srun --nodes=1 --nodelist=$node --gres=gpu:$GPUS_PER_NODE bash ${SCRIPT_DIR}/common/container_starter.sh \
            "$NNODES" \
            "$GPUS_PER_NODE" \
            "$MASTER_ADDR" \
            "$MASTER_PORT" \
            $rank \
            "$CHECKPOINT_PATH" \
            "$MODEL_ARG_PATH" \
            "$TRAINING_ARG_PATH" \
            "$OUTPUT_DIR" \
            "$MEGATRON_PATH" \
            "$TENSOR_PARALLEL_SIZE" \
            "$PIPELINE_PARALLEL_SIZE" \
            "$MICRO_BATCH_SIZE" \
            "$GLOBAL_BATCH_SIZE" \
            "$EXIT_INTERVAL" \
            "$LOG_INTERVAL" \
            "$TENSORBOARD_DIR" \
            "$WANDB_DIR" \
            "$WANDB_PROJECT" \
            "$WANDB_NAME" \
            "$OMP_NUM_THREADS" \
            &
    fi

    ((rank++))
done