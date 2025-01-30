#!/bin/bash

unset -f which

set -x

source $TRAINING_ARG_PATH
source $MODEL_ARG_PATH

# export HOME=/N/scratch/jindjia/runningcache/
# export NCCL_SOCKET_IFNAME=eno
# export HOME=$HOME

VOCAB_FILE=/N/scratch/jindjia/thepile/vocab.json
MERGE_FILE=/N/scratch/jindjia/thepile/merges.txt
DATA_PATH=/N/scratch/jindjia/thepile/pile_text_document

set -x

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
"

OUTPUT_ARGS="
    --log-interval ${LOG_INTERVAL} \
    --timing-log-level 0 \
    --log-timers-to-tensorboard \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 1 \
    --save-interval 5000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-save-dir ${WANDB_DIR} \
    --wandb-exp-name ${WANDB_NAME} \
"

TRAINING_ARGS="
    --bf16 \
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 80000 \
"

    # --use-flash-attn \

: "${BUCKET_SIZE:=10000000}"
echo "BUCKET_SIZE is $BUCKET_SIZE"

ADVANCED_ARGS="
    --use-flash-attn \
    --no-async-tensor-model-parallel-allreduce \
    --recompute-activations \
    --recompute-granularity selective \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --bucket-size $BUCKET_SIZE \
"

cd $MEGATRON_PATH


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
    --distributed-backend cpu:gloo,cuda:nccl \
    --exit-interval ${EXIT_INTERVAL}
