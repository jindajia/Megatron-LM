#!/bin/bash

NNODES=$1
GPUS_PER_NODE=$2
MASTER_ADDR=$3
MASTER_PORT=$4
NODE_RANK=$5
CHECKPOINT_PATH=$6
MODEL_ARG_PATH=$7
TRAINING_ARG_PATH=$8
OUTPUT_DIR=$9
MEGATRON_PATH=${10}
TENSOR_PARALLEL_SIZE=${11}
PIPELINE_PARALLEL_SIZE=${12}
MICRO_BATCH_SIZE=${13}
GLOBAL_BATCH_SIZE=${14}
EXIT_INTERVAL=${15}
LOG_INTERVAL=${16}
TENSORBOARD_DIR=${17}
WANDB_DIR=${18}
WANDB_PROJECT=${19}
WANDB_NAME=${20}
OMP_NUM_THREADS=${21}

echo "MASTER_ADDR: ${MASTER_ADDR}, slum_nodeid: ${SLURM_NODEID}, NODE_RANK: $NODE_RANK NNODES: ${NNODES}, GPUS_PER_NODE: ${GPUS_PER_NODE}, OUTPUT_DIR: ${OUTPUT_DIR}"
echo "RUNING IN BACKGROUND: $run_in_background"

# IMAGE_PATH=/N/slate/jindjia/bash_scripts/icml/env/build-docker/1-build-docker/network-sif
# IMAGE_NAME=megatron-lm-env_2025.sif

IMAGE_PATH=/N/slate/jindjia/bash_scripts/icml/env/build-docker/1-build-docker/megatron-lm-env_jan_27.sif
module purge
module load apptainer
module list

set -x
# cd /N/u/jindjia/BigRed200
export APPTAINER_TMPDIR=/N/slate/jindjia/apptainer_temp_home

apptainer exec \
    --nv \
    --containall \
    --home /N/slate/jindjia/apptainer_temp_home/:/home/user \
    --bind /N/scratch/jindjia:/N/scratch/jindjia \
    --bind /N/slate/jindjia:/N/slate/jindjia \
    --bind /tmp:/tmp \
    --env NNODES=$NNODES \
    --env GPUS_PER_NODE=$GPUS_PER_NODE \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    --env NODE_RANK=$NODE_RANK \
    --env CHECKPOINT_PATH=$CHECKPOINT_PATH \
    --env PROFILER_ARGS="$PROFILER_ARGS" \
    --env MODEL_ARG_PATH=$MODEL_ARG_PATH \
    --env TRAINING_ARG_PATH=$TRAINING_ARG_PATH \
    --env OUTPUT_DIR=$OUTPUT_DIR \
    --env MEGATRON_PATH=$MEGATRON_PATH \
    --env TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE \
    --env PIPELINE_PARALLEL_SIZE=$PIPELINE_PARALLEL_SIZE \
    --env MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE \
    --env GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE \
    --env EXIT_INTERVAL=$EXIT_INTERVAL \
    --env LOG_INTERVAL=$LOG_INTERVAL \
    --env TENSORBOARD_DIR=$TENSORBOARD_DIR \
    --env WANDB_DIR=$WANDB_DIR \
    --env WANDB_PROJECT=$WANDB_PROJECT \
    --env WANDB_NAME=$WANDB_NAME \
    --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
    --env NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
    --env SSL_CERT_FILE=$SSL_CERT_FILE \
    --env BUCKET_SIZE=$BUCKET_SIZE \
    $IMAGE_PATH \
    bash ${SCRIPT_DIR}/common/run.sh