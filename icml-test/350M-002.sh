# This is a starter script for 350M model
# Quantization Strategy: Wdiff 4bit, Grad 1bit, without using DUO and Hadamard Transformation
# Enable DUO twice Gradient Reduction

# ------------------ Common Args ------------------ #
source common_var.sh

# ------------------ Our Method DUO Args ------------------ #
ENABLE_DUO=1

# ------------------ Output Args ------------------ #
CHECKPOINT_PATH=/N/scratch/jindjia/checkpoints/icml/350M-1bitGrad-4bitWdiff-DUO/checkpoints
CHECKPOINT_SAVING_INTERVAL=5002
WANDB_PROJECT_NAME=ICML
WANDB_EXP_NAME=350M-1bitGrad-4bitWdiff-DUO
WANDB_DIR=/tmp/350M-1bitGrad-4bitWdiff-DUO/wandb
TENSORBOARD_DIR=/tmp/350M-1bitGrad-4bitWdiff-DUO/tensorboard


# ------------------ Quantization Args ------------------ #
GRAD_INTRA_QUANT_BIT=8
GRAD_INTRA_QUANT_BUCKET_SIZE=128
GRAD_INTER_QUANT_BIT=1
GRAD_INTER_QUANT_BUCKET_SIZE=64
WDIFF_QUANT_BIT=4
WDIFF_QUANT_BUCKET_SIZE=512

# ------------------ Model Training Args ------------------ #
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=512
source Model-card/350M/model.sh

source ./run.sh