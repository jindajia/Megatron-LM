
QUANTIZE_ARGS="
    --quantized-gradients \
    --gq-group-size-inter 64 \
    --gradient-quantization-bits-inter 1 \
    --gq-group-size-intra 128 \
    --gradient-quantization-bits-intra 8 \
    --gradient-alltoall-pipeline 1 \
    --quantized-weights \
    --weight-quantization-bits 4 \
    --wq-group-size 2048 \
"