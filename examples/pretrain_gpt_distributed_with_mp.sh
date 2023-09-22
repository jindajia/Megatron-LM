#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/ocean/projects/asc200010p/jjia1/Developer/LLM/gpt2/checkpoint/demo
VOCAB_FILE=/ocean/projects/asc200010p/jjia1/Developer/LLM/gpt2/vocab_file/gpt2-vocab.json
MERGE_FILE=/ocean/projects/asc200010p/jjia1/Developer/LLM/gpt2/merge_file/gpt2-merges.txt
DATA_PATH=/ocean/projects/asc200010p/jjia1/Developer/LLM/gpt2/datapath/gpt2_text_document
TENSORBOARD_DIR=/ocean/projects/asc200010p/jjia1/Developer/LLM/gpt2/tensorboard3

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --train-iters 300 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 5
"

torchrun $DISTRIBUTED_ARGS ../pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH
