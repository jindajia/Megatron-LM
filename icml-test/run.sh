#!/bin/bash

# set -x

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

OTHER_ARGS="
    --use-distributed-optimizer \
    --no-async-tensor-model-parallel-allreduce \
    --recompute-activations \
    --recompute-granularity selective \
    --use-flash-attn \
    --overlap-grad-reduce \
    --overlap-param-gather \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-cache-path $DATA_CACHE_DIR \
    --distributed-storage \
"

OUTPUT_ARGS="
    --log-interval 100 \
    --timing-log-level 2 \
    --log-timers-to-tensorboard \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 1 \
    --save-interval ${CHECKPOINT_SAVING_INTERVAL} \
    --eval-interval 100 \
    --eval-iters 10 \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
    --wandb-project ${WANDB_PROJECT_NAME} \
    --wandb-save-dir ${WANDB_DIR} \
    --wandb-exp-name ${WANDB_EXP_NAME}\
"

QUANTIZE_ARGS="
    --quantized-weights \
    --weight-quantization-bits 4 \
    --wq-group-size 512 \
    --quantized-gradients \
    --gq-group-size-inter 64 \
    --gradient-quantization-bits-inter 1 \
    --gq-group-size-intra 128 \
    --gradient-quantization-bits-intra 8 \
    --gradient-alltoall-pipeline 8 \
"

if [ "$ENABLE_DUO" -eq 1 ]; then
    QUANTIZE_ARGS+="--fast-slow-grad-reduce "
fi

echo "
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $OPTIMIZER_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $QUANTIZE_ARGS \
    $OTHER_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --distributed-backend nccl \
    --exit-duration-in-mins $EXIT_AFTER_MINS
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $OPTIMIZER_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $QUANTIZE_ARGS \
    $OTHER_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --distributed-backend nccl \
    --exit-duration-in-mins $EXIT_AFTER_MINS