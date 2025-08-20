#!/bin/bash

# Train MAV - Multi-Layer Alignment Model

MODEL_NAME="argsearch/llama-7b-sft-float32"
MODEL_NAME_SCRIPT="mav-llama-7b-sft"  # Simplified name for directory
LAYER_SELECTION=${LAYER_SELECTION:-auto}  # auto or uniform
DATASET_NAME=${DATASET_NAME:-"Dahoas/full-hh-rlhf"}  # Dataset name

# Hyperparameters  
EPOCH=${EPOCH:-1}
BETA=${BETA:-0.05} 
LAMBDA_L2=${LAMBDA_L2:-0.05}  # L2 regularization
LR=${LR:-5e-4}  # Learning rate
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}

# Automatically generate experiment name
DATASET_SHORT=$(echo $DATASET_NAME | cut -d'/' -f2 | cut -d'-' -f1-2)  # "full-hh" from "Dahoas/full-hh-rlhf"
EXP_NAME=${MODEL_NAME_SCRIPT}-${DATASET_SHORT}-epoch_${EPOCH}-beta_${BETA}-lr_${LR}-l2_${LAMBDA_L2}

OUTPUT_DIR="./outputs/mav/${EXP_NAME}"

# Check if directory exists
if [ -d "${OUTPUT_DIR}" ]; then
    echo -e "\n\n"
    echo "Error: Directory '${OUTPUT_DIR}' already exists. Please delete it or change hyperparameters." >&2
    exit 1
fi

echo "🚀 Training MAV - Multi-Layer Alignment"
echo "Base model: $MODEL_NAME (FROZEN)"
echo "Dataset: $DATASET_NAME"
echo "Experiment: $EXP_NAME"
echo "Output dir: $OUTPUT_DIR"
echo "Beta: $BETA, Lambda_L2: $LAMBDA_L2, LR: $LR"
echo "Batch size: ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"

# Use single GPU with smaller batch size
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 train.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --layer_selection $LAYER_SELECTION \
    --dataset_name $DATASET_NAME \
    --num_epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --learning_rate $LR \
    --warmup_steps 200 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --max_length 1024 \
    --beta $BETA \
    --lambda_l2 $LAMBDA_L2 \
    --bf16 \
    --resume_from_checkpoint True

echo "✅ MAV training complete! Saved to: $OUTPUT_DIR"