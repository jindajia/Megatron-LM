#!/bin/bash
#SBATCH -J starter
#SBATCH -p general
#SBATCH -A r01156
#SBATCH -o starter_%j.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10g
#SBATCH --time=00:01:00


module purge
module load apptainer
module list

# ------------------ Set Dist arguments ------------------ #
NNODES=$SLURM_NNODES
GPUS_PER_NODE=0
MASTER_PORT=6000
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

HOSTNAMES=$(scontrol show hostnames | sort -u)
HOSTLIST=""
FIRST_HOST=""
for HOST in $HOSTNAMES; do
  if [ -z "$FIRST_HOST" ]; then
    FIRST_HOST=$HOST
  fi
  HOST_ARRAY+=($HOST)
  HOSTLIST="${HOSTLIST}${HOST},"
done
HOSTLIST=${HOSTLIST%,}
MASTER_ADDR=$FIRST_HOST

# ------------------ Set running time ------------------ #
# Exit and Save Checkpoint after 2850 minutes (Please reserve 30 minutes for checkpoint saving) \
# For example, if you reserved 48 hours gpu times, you can set this to (48*60 - 30 = 2850) minutes.
EXIT_AFTER_MINS=2850


set -x

IMAGE_SOURCE=/N/slate/jindjia/bash_scripts/icml/env/build-docker/1-build-docker/pytorch_23.10-py3-quartz.sif

srun --nodes=$NNODES --gres=gpu:$GPUS_PER_NODE apptainer exec \
  --nv \
  --bind /N/scratch/jindjia/:/N/scratch/jindjia/ \
  --bind /N/slate/jindjia/:/N/slate/jindjia/ \
  --bind /tmp:/tmp \
  --env NNODES=$NNODES \
  --env GPUS_PER_NODE=$GPUS_PER_NODE \
  --env MASTER_ADDR=$MASTER_ADDR \
  --env MASTER_PORT=$MASTER_PORT \
  --env EXIT_AFTER_MINS=$EXIT_AFTER_MINS \
  $IMAGE_SOURCE \
  bash 1_3B-002.sh