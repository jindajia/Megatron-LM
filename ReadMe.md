### DUO is a high precision training framework, currently can reduce accuracy loss involved by compressed training.

### Here is a sample script for running GPT-350M training with three differnt way: 
    - 1. full precision training (Baseline)
    - 2. compression for weight and gradient (SDP4Bit)
    - 3. improve accuracy by enable DUO (DUO)

*Baseline, SDP4Bit, DUO4Bit*


```
#!/usr/bin/env bash
set -x


######################################
# 1. ENVIRONMENT VARIABLES & DEFAULTS
######################################

# If your environment does not set these automatically, change them as needed
export MASTER_ADDR=
export MASTER_PORT=
export NODE_RANK=
export NNODES=
export GPUS_PER_NODE=
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Basic training parameters
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export WANDB_PROJECT=

# Data or paths
VOCAB_FILE=
MERGE_FILE=
DATA_PATH=
DATA_INDEX_CACHE_PATH=

mkdir $DATA_INDEX_CACHE_PATH

export MEGATRON_PATH=
cd $MEGATRON_PATH

# HPC or environment variables that might or might not be needed
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" 

# You can override or set these externally. If not set, we'll default them here.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"

# The main output directory
export OUTPUT_BASE_DIR=

######################################
# 2. LISTS OF MODELS & TRAINING CONFIGS
######################################

# Define the list of models to run
MODEL_LIST=(
    "6_7B"
)

# Example: training config variants
TRAIN_CONFIG_LIST=(
    "Baseline"
    "SDP4Bit"
    "DUO4Bit"
)

######################################
# 3. DEFINE PER-MODEL AND PER-TRAINING ARGS
######################################

function set_model_6_7B() {
    MODEL_NAME="6_7B"
    MODEL_ARGS="
        --num-layers 32 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --lr 0.00012 \
        --min-lr 0.000012 \
    "
    export TENSOR_PARALLEL_SIZE=8
    export PIPELINE_PARALLEL_SIZE=1
    export MICRO_BATCH_SIZE=4
    echo "Model Config: 7B"
}

function set_model_13B() {
    MODEL_NAME="13B"
    MODEL_ARGS="
        --num-layers 40 \
        --hidden-size 5120 \
        --num-attention-heads 40 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --lr 0.0001 \
        --min-lr 0.00001 \
    "
    export TENSOR_PARALLEL_SIZE=8
    export PIPELINE_PARALLEL_SIZE=1
    export MICRO_BATCH_SIZE=2
    echo "Model Config: 13B"
}

function set_model_18B() {
    MODEL_NAME="18B"
    export MODEL_ARGS="
        --num-layers 40 \
        --hidden-size 6144 \
        --num-attention-heads 48 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --lr 0.000097 \
        --min-lr 0.0000097 \
    "
    export TENSOR_PARALLEL_SIZE=8
    export PIPELINE_PARALLEL_SIZE=1
    export MICRO_BATCH_SIZE=2
    echo "Model Config: 18B"
}

function set_optimizer_args() {
  export OPTIMIZER_ARGS="
    --lr-decay-iters 70000 \
    --lr-decay-style cosine \
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
}

function set_train_config_DUO0Bit() {
  # Example "DUO4Bit.sh"
  export QUANTIZE_ARGS="
    --quantized-gradients \
    --gq-group-size-inter 128 \
    --gradient-quantization-bits-inter 0 \
    --gq-group-size-intra 128 \
    --gradient-quantization-bits-intra 0 \
    --hadamard-transform \
    --gradient-alltoall-pipeline 1 \
    --quantized-weights \
    --weight-quantization-bits 4 \
    --wq-group-size 2048 \
  "

  export DUO_ARGS="
    --fast-slow-grad-reduce \
  "

  echo "Training Config: DUO0Bit"
}

function set_train_config_DUO1Bit() {
  # Example "DUO4Bit.sh"
  export QUANTIZE_ARGS="
    --quantized-gradients \
    --gq-group-size-inter 64 \
    --gradient-quantization-bits-inter 1 \
    --gq-group-size-intra 128 \
    --gradient-quantization-bits-intra 8 \
    --hadamard-transform \
    --gradient-alltoall-pipeline 1 \
    --quantized-weights \
    --weight-quantization-bits 4 \
    --wq-group-size 2048 \
  "

  export DUO_ARGS="
    --fast-slow-grad-reduce \
  "

  echo "Training Config: DUO1Bit"
}

function set_train_config_DUO4Bit() {
  # Example "DUO4Bit.sh"
  export QUANTIZE_ARGS="
    --quantized-gradients \
    --gq-group-size-inter 128 \
    --gradient-quantization-bits-inter 4 \
    --gq-group-size-intra 128 \
    --gradient-quantization-bits-intra 8 \
    --hadamard-transform \
    --gradient-alltoall-pipeline 1 \
    --quantized-weights \
    --weight-quantization-bits 4 \
    --wq-group-size 2048 \
  "

  export DUO_ARGS="
    --fast-slow-grad-reduce \
  "

  echo "Training Config: DUO4Bit"
}

function set_train_config_SDP1Bit() {
  # Example "DUO4Bit.sh"
  export QUANTIZE_ARGS="
    --quantized-gradients \
    --gq-group-size-inter 64 \
    --gradient-quantization-bits-inter 1 \
    --gq-group-size-intra 128 \
    --gradient-quantization-bits-intra 8 \
    --hadamard-transform \
    --gradient-alltoall-pipeline 1 \
    --quantized-weights \
    --weight-quantization-bits 4 \
    --wq-group-size 2048 \
  "

  export DUO_ARGS=""

  echo "Training Config: SDP1Bit"
}

function set_train_config_SDP4Bit() {
  # Example "SDP4Bit.sh" – you might have different flags
  export QUANTIZE_ARGS="
    --quantized-gradients \
    --gq-group-size-inter 128 \
    --gradient-quantization-bits-inter 4 \
    --gq-group-size-intra 128 \
    --gradient-quantization-bits-intra 8 \
    --hadamard-transform \
    --gradient-alltoall-pipeline 1 \
    --quantized-weights \
    --weight-quantization-bits 4 \
    --wq-group-size 2048 \
  "
  export DUO_ARGS=""  # or other relevant flags

  echo "Training Config: SDP4Bit"
}

function set_train_config_Baseline() {
  # Example "SDP4Bit.sh" – you might have different flags
  export QUANTIZE_ARGS=""
  export DUO_ARGS=""  # or other relevant flags

  echo "Training Config: Baseline"
}
######################################
# 4. MAIN TRAINING LOOP
######################################

# Loop through all models
for model_name in "${MODEL_LIST[@]}"; do
    # Set model-specific parameters
    case $model_name in
        "1_3B")
            set_model_1_3B
            ;;
        "2_7B")
            set_model_2_7B
            ;;
        "6_7B")
            set_model_6_7B
            ;;
        "13B")
            set_model_13B
            ;;
        "18B")
            set_model_18B
            ;;
        *)
            echo "Unknown model: $model_name"
            exit 1
            ;;
    esac

    # Set optimizer arguments (common for all models)
    set_optimizer_args

    # Recompute the effective world size
    export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))

    # Print for debugging
    echo "Global Batch Size: $GLOBAL_BATCH_SIZE"

    # Loop through all training configurations
    for train_config_name in "${TRAIN_CONFIG_LIST[@]}"; do
        if [ "$train_config_name" == "DUO4Bit" ]; then
            set_train_config_DUO4Bit
        elif [ "$train_config_name" == "SDP4Bit" ]; then
            set_train_config_SDP4Bit
        elif [ "$train_config_name" == "SDP1Bit" ]; then
            set_train_config_SDP1Bit
        elif [ "$train_config_name" == "DUO1Bit" ]; then
            set_train_config_DUO1Bit
        elif [ "$train_config_name" == "DUO0Bit" ]; then
            set_train_config_DUO0Bit
        elif [ "$train_config_name" == "Baseline" ]; then
            set_train_config_Baseline
        else
            echo "Unknown training config: $train_config_name"
            exit 1
        fi

        echo "Running $MODEL_NAME / $train_config_name on $NNODES nodes"
        export WANDB_NAME="${MODEL_NAME}_${train_config_name}_${NNODES}_NODES"

        # Compose an OUTPUT_DIR per run
        job_id="$(date '+%Y%m%d_%H%M%S')"
        export OUTPUT_DIR="${OUTPUT_BASE_DIR}/${NNODES}_NODES/${job_id}/${MODEL_NAME}/${train_config_name}"
        export WANDB_DIR="${OUTPUT_DIR}/wandb_logs"
        export TENSORBOARD_DIR="${OUTPUT_DIR}/tb_logs"
        mkdir -p "${OUTPUT_DIR}"

        ##################################
        # REPLICATES run.sh main content
        ##################################
        export DATA_ARGS="
        --data-path ${DATA_PATH} \
        --data-cache-path ${DATA_INDEX_CACHE_PATH} \
        --vocab-file ${VOCAB_FILE} \
        --merge-file ${MERGE_FILE} \
        --distributed-storage \
        "

        export OUTPUT_ARGS="
        --log-interval ${LOG_INTERVAL} \
        --timing-log-level 2 \
        --log-timers-to-tensorboard \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --tensorboard-log-interval 1 \
        --save-interval 5000 \
        --eval-interval 100 \
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
        "

        # Construct the torchrun arguments
        DISTRIBUTED_ARGS="
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT
        "

        export CKPT_DIR=
        mkdir -p "${CKPT_DIR}"

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
            --distributed-backend "nccl" \
            --save $CKPT_DIR \
            --load $CKPT_DIR \
            2>&1 | tee -a "${OUTPUT_DIR}/train.log"
    done
done

echo "All training runs completed."
```