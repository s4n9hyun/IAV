#!/bin/bash

# Cache reference logits for IAV training

MODEL="argsearch/llama-7b-sft-float32"
SEQ_LENGTH=1024

echo "Caching reference logits..."

python src/caching/cache_reference_logits.py \
    --model_name $MODEL \
    --split train \
    --seq_length $SEQ_LENGTH \
    --batch_size 16 \
    --cache_file cache.pkl

echo "Caching complete."