
QUANTIZE_ARGS="
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

DUO_ARGS="
    --fast-slow-grad-reduce \
    --high-precision-grad-device cpu \
"