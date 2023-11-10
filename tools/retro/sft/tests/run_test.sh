bash tools/retro/sft/tests/sft_retro_lm.sh   qc               843m            128    5e-6  /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting

bash tools/retro/sft/tests/sft_retro_lm.sh   open_inst        843m            128    5e-6  /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting


bash tools/retro/sft/tests/sft_retro_lm.sh   qc               43b            128    5e-6  /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed

bash tools/retro/sft/tests/sft_retro_lm.sh   open_inst        43b            128    5e-6  /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed


# single node script
#export CUDA_DEVICE_MAX_CONNECTIONS=1
#python -m torch.distributed.run --nproc_per_node 8 \
#                  --nnodes 1 \
#                  --node_rank 0 \
#                  --master_addr localhost \
#                  --master_port 6000  /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/tools/retro/sft/sft_retro.py --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --pipeline-model-parallel-size 1 --tensor-model-parallel-size 1 --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 4096 --max-position-embeddings 4096 --lr-decay-style cosine --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model --clip-grad 1.0 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 --log-params-norm --log-num-zeros-in-grad --bf16 --use-distributed-optimizer --retro-workdir /lustre/fsw/adlr/adlr-nlp/boxinw/next-llm --retro-add-retriever --retro-num-neighbors 2 --retro-attention-gate 0 --data-path 1.0 open_inst --data-folder /lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ --recompute-activations --lr 5e-6 --micro-batch-size 1 --global-batch-size 128 --min-lr 5e-6 --retro-cyclic-train-iters 1000 --train-iters 1000 --dataloader-type cyclic --save /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_843m_128_5e-6 --log-interval 10 --save-interval 500 --eval-interval 200 --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/tensorboard/retro-open_inst_pp1_same_format_ctx1_843m_128_5e-6 --log-validation-ppl-to-tensorboard --eval-iters 100 --eod-mask-loss --answer-loss-only --ft_neighbours 1 --task none --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting --finetune --no-load-rng --no-load-optim
#
#python -u /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/tools/retro/sft/sft_retro.py --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --pipeline-model-parallel-size 1 --tensor-model-parallel-size 1 --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 4096 --max-position-embeddings 4096 --lr-decay-style cosine --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model --clip-grad 1.0 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 --log-params-norm --log-num-zeros-in-grad --bf16 --use-distributed-optimizer --retro-workdir /lustre/fsw/adlr/adlr-nlp/boxinw/next-llm --retro-add-retriever --retro-num-neighbors 2 --retro-attention-gate 0 --data-path 1.0 open_inst --data-folder /lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ --recompute-activations --lr 5e-6 --micro-batch-size 1 --global-batch-size 128 --min-lr 5e-6 --retro-cyclic-train-iters 1000 --train-iters 1000 --dataloader-type cyclic --save /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_843m_128_5e-6 --log-interval 10 --save-interval 500 --eval-interval 200 --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/tensorboard/retro-open_inst_pp1_same_format_ctx1_843m_128_5e-6 --log-validation-ppl-to-tensorboard --eval-iters 100 --eod-mask-loss --answer-loss-only --ft_neighbours 1 --task none --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting --finetune --no-load-rng --no-load-optim
#
#python -u /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/tools/retro/sft/sft_retro.py --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --pipeline-model-parallel-size 1 --tensor-model-parallel-size 1 --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 4096 --max-position-embeddings 4096 --lr-decay-style cosine --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model --clip-grad 1.0 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 --log-params-norm --log-num-zeros-in-grad --bf16 --use-distributed-optimizer --retro-workdir /lustre/fsw/adlr/adlr-nlp/boxinw/next-llm --retro-add-retriever --retro-num-neighbors 2 --retro-attention-gate 0 --data-path 1.0 quiet-cockatoo_commercial --data-folder /lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ --recompute-activations --lr 5e-6 --micro-batch-size 1 --global-batch-size 128 --min-lr 5e-6 --retro-cyclic-train-iters 1000 --train-iters 1000 --dataloader-type cyclic --save /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_843m_128_5e-6 --log-interval 10 --save-interval 500 --eval-interval 200 --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/tensorboard/retro-open_inst_pp1_same_format_ctx1_843m_128_5e-6 --log-validation-ppl-to-tensorboard --eval-iters 100 --eod-mask-loss --answer-loss-only --ft_neighbours 1 --task none --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting --finetune --no-load-rng --no-load-optim
#
#
#
