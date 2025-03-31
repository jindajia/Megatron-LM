MODEL_ARGS="
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
"

OPTIMIZER_ARGS="
    --lr 0.0002 \
    --lr-decay-iters 70000 \
    --lr-decay-style cosine \
    --min-lr 0.00002 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-08 \
    --weight-decay .1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 0 \
    --loss-scale 0 \
    --loss-scale-window 1000 \
    --hysteresis 2 \
    --min-loss-scale 1 \
"

export TENSOR_PARALLEL_SIZE=8
export PIPELINE_PARALLEL_SIZE=1
export MICRO_BATCH_SIZE=4

WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
ACCUMULATION_STEP=${ACCUMULATION_STEP:-32}
export GLOBAL_BATCH_SIZE=$(($WORLD_SIZE / ($TENSOR_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE) * $MICRO_BATCH_SIZE * $ACCUMULATION_STEP))

# Print results
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Accumulation Step: $ACCUMULATION_STEP"