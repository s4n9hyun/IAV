#!/bin/bash

# Simple training script for IAV
# Passes all arguments directly to Python script
# Usage: ./train.sh [any python arguments]
# Example: ./train.sh --save_steps 250 --eval_steps 100
# Example: ./train.sh --model_name_or_path custom-model --save_steps 500

echo "Starting IAV training..."

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

# Force single GPU training for better memory efficiency
echo "Using single GPU training (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python src/training/train_main.py "$@"