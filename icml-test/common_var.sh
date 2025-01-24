# ------------------ Distributed Config ------------------ #
NODE_RANK=$SLURM_NODEID

# ------------------ MEGATRON Path ------------------ #
MEGATRON_PATH=/N/slate/jindjia/RepeatComm/accuracy/Megatron-LM

# ------------------ Dataset Args ------------------ #
VOCAB_FILE=/N/scratch/jindjia/thepile/vocab.json
MERGE_FILE=/N/scratch/jindjia/thepile/merges.txt
DATA_PATH=/N/scratch/jindjia/thepile/pile_text_document
DATA_CACHE_DIR=/tmp/thepile/data_cache # set to your GPU nodes temporary directory
