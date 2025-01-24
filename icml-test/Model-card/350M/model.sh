
MODEL_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
"

TRAINING_ARGS="
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 80000 \
"

OPTIMIZER_ARGS="
    --lr 0.0003 \
    --lr-decay-iters 70000 \
    --lr-decay-style cosine \
    --min-lr 0.00003 \
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
    --bf16 \
"