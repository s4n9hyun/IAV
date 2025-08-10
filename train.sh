#!/bin/bash

# Multi-GPU training script for IAV using accelerate
# Usage: ./train.sh [any python arguments]
# Example: ./train.sh --save_steps 250 --eval_steps 100
# Example: ./train.sh --model_name_or_path custom-model --save_steps 500

echo "Starting IAV training with accelerate..."

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using multi-GPU training with accelerate"
    
    # Set reasonable defaults for checkpointing (can be overridden by command line args)
    # With grad_accum=1, global_step increments every batch
    # save_steps=50 means checkpoint every 50 batches (~2.5 min at 3s/batch)
    accelerate launch \
        --config_file accelerate_config.yaml \
        src/training/train_main.py \
        --bf16 \
        --cache_file cache_argsearch_llama-7b-sft-float32_seq1024.pkl \
        --save_steps 1000 \
        --eval_steps 500 \
        "$@"
else
    echo "Only 1 GPU detected, falling back to single GPU training"
    CUDA_VISIBLE_DEVICES=0 python src/training/train_main.py \
        --bf16 \
        --cache_file cache_argsearch_llama-7b-sft-float32_seq1024.pkl \
        --save_steps 100 \
        --eval_steps 200 \
        "$@"
fi