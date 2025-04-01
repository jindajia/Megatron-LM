
# Start container with interactive shell

IMAGE_PATH=/N/slate/jindjia/bash_scripts/icml/env/build-docker/1-build-docker/megatron-lm-env_jan_27.sif
module purge
module load apptainer
module list

set -x
export APPTAINER_TMPDIR=/N/slate/jindjia/apptainer_temp_home

apptainer shell \
    --nv \
    --containall \
    --home /N/slate/jindjia/apptainer_temp_home/:/home/user \
    --bind /N/scratch/jindjia:/N/scratch/jindjia \
    --bind /N/slate/jindjia:/N/slate/jindjia \
    --bind /tmp:/tmp \    
    "$IMAGE_PATH"
