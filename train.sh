#!/bin/bash

# Training script for IAV
# Default model: llama-7b-sft
# Default dataset: hh-rlhf

MODEL_NAME=${1:-"argsearch/llama-7b-sft-float32"}
DATASET_NAME=${2:-"Anthropic/hh-rlhf"}
OUTPUT_DIR=${3:-"./outputs/"}

echo "Starting IAV training..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Output directory: $OUTPUT_DIR"

python src/training/train_main.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --max_seq_length 1024 \
    --eval_steps 500 \
    --save_steps 1000 \
    --logging_steps 10 \
    --seed 42 \
    --bf16 \
    --tf32 \
    --gradient_checkpointing \
    --beta 0.5 \
    --lambda_kl 0.01 \
    --lambda_l2 0.1