#!/bin/bash

# Train IAV model with parameter-based naming (following GenARM convention)

MODEL_NAME="argsearch/llama-7b-sft-float32"
OUTPUT_DIR="./outputs"
CACHE_FILE="cache.pkl"

# Hyperparameters (modify these for different experiments)
BETA=${BETA:-0.1}
LAMBDA_KL=${LAMBDA_KL:-0.05}
LAMBDA_L2=${LAMBDA_L2:-0.005}
LR=${LR:-5e-6}

echo "Beta: $BETA, Lambda_KL: $LAMBDA_KL, Lambda_L2: $LAMBDA_L2, LR: $LR"

# Use single GPU to avoid NCCL timeout issues
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
    src/training/train_main.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --use_param_naming \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1\
    --learning_rate $LR \
    --warmup_steps 100 \
    --save_steps 2000 \
    --eval_steps 999999 \
    --max_prompt_length 512 \
    --max_length 1024 \
    --beta $BETA \
    --lambda_kl $LAMBDA_KL \
    --lambda_l2 $LAMBDA_L2 \
    --bf16 \
    --cache_file $CACHE_FILE \
    --resume_from_checkpoint True

echo "IAV training complete."