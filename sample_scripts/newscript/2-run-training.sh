#!/usr/bin/env bash
set -x

######################################
# 1. ENVIRONMENT VARIABLES & DEFAULTS
######################################

# If your environment does not set these automatically, change them as needed
export MASTER_ADDR="${MASTER_ADDR:-localhost}" #TODO
export MASTER_PORT="${MASTER_PORT:-6002}"
export NODE_RANK="${NODE_RANK:-0}"        # TODO
export WORLD_SIZE="${WORLD_SIZE:-1}"   # TODO
export NNODES="${NNODES:-1}"           # TODO
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}" # TODO


# Basic training parameters
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export EXIT_INTERVAL="${EXIT_INTERVAL:-20}"
export WANDB_PROJECT="${WANDB_PROJECT:-icml-performance-test-new}"


# Data or paths
export VOCAB_FILE="${VOCAB_FILE:-/N/scratch/jindjia/thepile/vocab.json}" # TODO
export MERGE_FILE="${MERGE_FILE:-/N/scratch/jindjia/thepile/merges.txt}" # TODO
# export DATA_PATH="${DATA_PATH:-/path/to/pile_text_document}"

export MEGATRON_PATH="${MEGATRON_PATH:-"/N/slate/jindjia/bash_scripts/DUO-debug/repo/Megatron-LM_DUO-speed"}" # TODO

# HPC or environment variables that might or might not be needed
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" 
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-"hsn0"}" #TODO
export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-"hsn1"}" #TODO

echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"

# Other logging or profiling arguments
# export PROFILER_ARGS="${PROFILER_ARGS:-}"   # e.g. "--profile --use-pytorch-profiler ..."

# You can override or set these externally. If not set, we'll default them here.
export BUCKET_SIZE="${BUCKET_SIZE:-40000000}"
export ACCUMULATION_STEP="${ACCUMULATION_STEP:-32}"

# The main output directory
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-"/N/slate/jindjia/bash_scripts/icml/testnew-script"}" #TODO


######################################
# 2. LISTS OF MODELS / TRAINING CONFIGS
######################################



# Example: training config variants
TRAIN_CONFIG_LIST=(
    "DUO4Bit"
    # "SDP4Bit"
)


######################################
# 3. DEFINE PER-MODEL AND PER-TRAINING ARGS
######################################

##
# For illustration, we define "functions" that set environment variables
# to replicate your separate `model-cards` or `training-config` files.
# You can define multiple model_xxx() or train_config_xxx() variants
# for different configurations. Then just call them as needed in your loops.
##

function set_model() {

# Example "350M" content
MODEL_NAME="350M"
MODEL_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
"
export TENSOR_PARALLEL_SIZE=4
export PIPELINE_PARALLEL_SIZE=1
export MICRO_BATCH_SIZE=1

# Example "6.7B" content
# MODEL_NAME="6_7B"
# MODEL_ARGS="
#     --num-layers 32 \
#     --hidden-size 4096 \
#     --num-attention-heads 32 \
#     --seq-length 4096 \
#     --max-position-embeddings 4096 \
# "
# export TENSOR_PARALLEL_SIZE=8
# export PIPELINE_PARALLEL_SIZE=1
# export MICRO_BATCH_SIZE=1

# Example "13B" content
# MODEL_NAME="13B"
# MODEL_ARGS="
#     --num-layers 40 \
#     --hidden-size 5120 \
#     --num-attention-heads 40 \
#     --seq-length 4096 \
#     --max-position-embeddings 4096 \
# "
# export TENSOR_PARALLEL_SIZE=8
# export PIPELINE_PARALLEL_SIZE=1
# export MICRO_BATCH_SIZE=1

# Example "18B" content
# MODEL_NAME="18B"
# export MODEL_ARGS="
#     --num-layers 40 \
#     --hidden-size 6144 \
#     --num-attention-heads 48 \
#     --seq-length 4096 \
#     --max-position-embeddings 4096 \
# "
# export TENSOR_PARALLEL_SIZE=8
# export PIPELINE_PARALLEL_SIZE=1
# export MICRO_BATCH_SIZE=1

  export OPTIMIZER_ARGS="
    --lr 0.0001 \
    --lr-decay-iters 70000 \
    --lr-decay-style cosine \
    --min-lr 0.00001 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-08 \
    --weight-decay .1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --loss-scale 0 \
    --loss-scale-window 1000 \
    --hysteresis 2 \
    --min-loss-scale 1 \
  "


  # Recompute the effective world size
  export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))

  # Compute global batch size
  export GLOBAL_BATCH_SIZE=$((WORLD_SIZE / (TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE) * MICRO_BATCH_SIZE * ACCUMULATION_STEP))

  # Print for debugging
  echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
  echo "Accumulation Step: $ACCUMULATION_STEP"
}

##
# Training configs
##

function set_train_config_DUO4Bit() {
  # Example "DUO4Bit.sh"
  export QUANTIZE_ARGS="
    --quantized-gradients \
    --gq-group-size-inter 128 \
    --gradient-quantization-bits-inter 4 \
    --gq-group-size-intra 128 \
    --gradient-quantization-bits-intra 8 \
    --gradient-alltoall-pipeline 1 \
    --quantized-weights \
    --weight-quantization-bits 4 \
    --wq-group-size 2048 \
  "

  export DUO_ARGS="
    --fast-slow-grad-reduce \
    --high-precision-grad-device cpu \
  "

  echo "Training Config: DUO4Bit"
}

function set_train_config_SDP4Bit() {
  # Example "SDP4Bit.sh" – you might have different flags
  export QUANTIZE_ARGS="
    --quantized-gradients \
    --gq-group-size-inter 128 \
    --gradient-quantization-bits-inter 4 \
    --gq-group-size-intra 128 \
    --gradient-quantization-bits-intra 4 \
    --gradient-alltoall-pipeline 1 \
    --quantized-weights \
    --weight-quantization-bits 4 \
    --wq-group-size 2048 \
  "
  export DUO_ARGS=""  # or other relevant flags

  echo "Training Config: SDP4Bit"
}


######################################
# 4. MAIN TRAINING LOOP
######################################

# We iterate over the node counts, the model, and the training config
# "torchrun" is used directly (instead of `srun`). You can supply
# "--nnodes", "--nproc_per_node", etc. to torchrun. If you *do* want multi-node,
# ensure environment variables MASTER_ADDR, MASTER_PORT, NODE_RANK, etc. are set.
######################################


set_model

for train_config_name in "${TRAIN_CONFIG_LIST[@]}"; do

    if [ "$train_config_name" == "DUO4Bit" ]; then
    set_train_config_DUO4Bit
    elif [ "$train_config_name" == "SDP4Bit" ]; then
    set_train_config_SDP4Bit
    else
    echo "Unknown training config: $train_config_name"
    exit 1
    fi

    echo "Running $MODEL_NAME / $train_config_name on $NNODES nodes"
    export WANDB_NAME="${MODEL_NAME}_${train_config_name}_${NNODES}_NODES"

    # Compose an OUTPUT_DIR per run
    # If you rely on SLURM_JOB_ID, define it or remove references
    # (below we just use a dummy "job_id_1234" for demonstration)
    job_id="$(date '+%Y%m%d_%H%M%S')"
    export OUTPUT_DIR="${OUTPUT_BASE_DIR}/${NNODES}_NODES/${job_id}/${MODEL_NAME}/${train_config_name}"
    export WANDB_DIR="${OUTPUT_DIR}/wandb_logs"
    export TENSORBOARD_DIR="${OUTPUT_DIR}/tb_logs"
    mkdir -p "${OUTPUT_DIR}"

    ##################################
    # REPLICATES run.sh main content
    ##################################
    export DATA_ARGS="
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --mock-data
    "

    export OUTPUT_ARGS="
    --log-interval ${LOG_INTERVAL} \
    --timing-log-level 0 \
    --log-timers-to-tensorboard \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 1 \
    --save-interval 5000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-save-dir ${WANDB_DIR} \
    --wandb-exp-name ${WANDB_NAME} \
    "

    export TRAINING_ARGS="
    --bf16 \
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 80000 \
    "

    export ADVANCED_ARGS="
    --use-flash-attn \
    --no-async-tensor-model-parallel-allreduce \
    --recompute-activations \
    --recompute-granularity selective \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --bucket-size $BUCKET_SIZE \
    "

    # Construct the torchrun arguments
    DISTRIBUTED_ARGS="
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT
    "

    cd "${MEGATRON_PATH}"  # If you need to cd into Megatron-lm folder

    # Finally run it:
    torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
        $MODEL_ARGS \
        $TRAINING_ARGS \
        $OPTIMIZER_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $QUANTIZE_ARGS \
        $PROFILER_ARGS \
        $ADVANCED_ARGS \
        $DUO_ARGS \
        --distributed-backend "cpu:gloo,cuda:nccl" \
        --exit-interval "${EXIT_INTERVAL}" \
        2>&1 | tee -a "${OUTPUT_DIR}/train.log"

done

echo "All training runs completed."
