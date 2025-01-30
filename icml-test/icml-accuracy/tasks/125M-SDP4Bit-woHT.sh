#!/bin/bash
#SBATCH -J 2xA100Nodes
#SBATCH -p gpu-debug
#SBATCH -A r01156
#SBATCH -o /N/slate/jindjia/bash_scripts/bytedance2/icml-performance-bytedance/tasks/2xA100Nodes/batch_output_%j.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=240g
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiajinda001@gmail.com

set -x
export MEGATRON_PATH="/N/slate/jindjia/RepeatComm/dev/Fast-Slow-performance"

export LOG_INTERVAL=1
export EXIT_INTERVAL=40
export WANDB_PROJECT=icml-performance-test
export SCRIPT_DIR=/N/slate/jindjia/bash_scripts/bytedance2/icml-performance-bytedance
export OUTPUT_BASE_DIR=/N/slate/jindjia/bash_scripts/bytedance2/icml-performance-bytedance/tasks/2xA100Nodes/output_dir-acc32-setting1

# ----------------- Numerber of GPUs  -----------------

NUM_NODES_LIST=(2 ) 
export RUNNING_GPUS_PER_NODE=4


# ----------------- Strat srun script -----------------

export OMP_NUM_THREADS=$OMP_NUM_THREADS
export SSL_CERT_FILE=/N/slate/jindjia/cacert.pem # for wandb login with singularity, I encountered a problem, you may not need this


nvidia-smi

# ----------------- Model and Training Config -----------------
MODEL_LIST=(
    "125M" 
)

TRAIN_CONFIG_LIST=(
    "SDP4Bit" 
) 


# ----------------- Model and Training Config -----------------

for num_nodes in "${NUM_NODES_LIST[@]}"; do
    export RUNNING_NODES=$num_nodes
    for model in "${MODEL_LIST[@]}"; do
        MODEL_NAME=$model
        for train_confit_name in "${TRAIN_CONFIG_LIST[@]}"; do
            echo "Running $MODEL_NAME, $train_confit_name on $RUNNING_NODES nodes"
            TRAIN_CONFIG_NAME=$train_confit_name
            export WANDB_NAME=${TRAIN_CONFIG_NAME}_${RUNNING_NODES}_NODES
            export MODEL_ARG_PATH=${SCRIPT_DIR}/model-cards/${MODEL_NAME}.sh
            export TRAINING_ARG_PATH=${SCRIPT_DIR}/training-config/${TRAIN_CONFIG_NAME}.sh
            export OUTPUT_DIR=${OUTPUT_BASE_DIR}/${SLURM_JOB_ID}/${RUNNING_NODES}_NODES/${MODEL_NAME}/${TRAIN_CONFIG_NAME}
            export TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
            export WANDB_DIR=${OUTPUT_DIR}/wandb
            mkdir -p $OUTPUT_DIR

            bash ${SCRIPT_DIR}/starter.sh > $OUTPUT_DIR/${TRAIN_CONFIG_NAME}_logfile.log 2>&1
        done
    done

done
