#!/bin/bash

# Train MAV - Multi-Layer Alignment Model

MODEL_NAME="argsearch/llama-7b-sft-float32"
MODEL_NAME_SCRIPT="simplified-mav-llama-7b-sft"  # Simplified cross-attention version
DATASET_NAME=${DATASET_NAME:-"Dahoas/full-hh-rlhf"}  # Dataset name

# Hyperparameters - Conservative settings for stable training
EPOCH=${EPOCH:-1}  
BETA=${BETA:-0.1}  # Standard DPO beta
LAMBDA_L2=${LAMBDA_L2:-0.001}  # Reduced L2 regularization for stronger alignment
LR=${LR:-5e-5}  # Increased learning rate for better alignment learning
BATCH_SIZE=${BATCH_SIZE:-2}  # Smaller batch for stable gradients
GRAD_ACCUM=${GRAD_ACCUM:-16}  # Effective batch size of 16
# REAL BATCH SIZE = BATCH_SIZE*GRAD_ACCUM
# Automatically generate experiment name
DATASET_SHORT=$(echo $DATASET_NAME | cut -d'/' -f2 | cut -d'-' -f1-2)  # "full-hh" from "Dahoas/full-hh-rlhf"
EXP_NAME=${MODEL_NAME_SCRIPT}-${DATASET_SHORT}-epoch_${EPOCH}-beta_${BETA}-lr_${LR}-l2_${LAMBDA_L2}

OUTPUT_DIR="./outputs/mav/${EXP_NAME}"

# Check if directory exists and handle resume
RESUME_CHECKPOINT=""
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Directory '${OUTPUT_DIR}' already exists."
    
    # Find latest checkpoint
    LATEST_CHECKPOINT=$(ls -t ${OUTPUT_DIR}/alignment_step_*.pt 2>/dev/null | head -n1)
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        STEP=$(basename $LATEST_CHECKPOINT | grep -oP 'step_\K[0-9]+')
        echo "Found checkpoint at step $STEP"
        echo "Will resume from: $LATEST_CHECKPOINT"
        RESUME_CHECKPOINT=$LATEST_CHECKPOINT
    elif [ -f "${OUTPUT_DIR}/best_alignment.pt" ]; then
        echo "Found best_alignment.pt, will resume from it"
        RESUME_CHECKPOINT="${OUTPUT_DIR}/best_alignment.pt"
    else
        echo "No checkpoints found in existing directory. Starting fresh training."
        echo "Delete the directory if you want to start completely fresh."
    fi
    echo ""
fi

echo "🚀 Training Simplified Cross-Attention MAV (~200M params)"
echo "Base model: $MODEL_NAME (FROZEN)"
echo "Dataset: $DATASET_NAME"
echo "Experiment: $EXP_NAME"
echo "Output dir: $OUTPUT_DIR"
echo "Beta: $BETA, Lambda_L2: $LAMBDA_L2, LR: $LR"
echo "Batch size: ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"

# Build command
CMD="CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 train.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATASET_NAME \
    --num_epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --learning_rate $LR \
    --warmup_steps 200 \
    --save_steps 1000 \
    --eval_steps 2000 \
    --max_length 1024 \
    --beta $BETA \
    --lambda_l2 $LAMBDA_L2 \
    --bf16"

# Add resume flag if checkpoint exists
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume_from_checkpoint $RESUME_CHECKPOINT"
    echo "🔄 Resuming training from checkpoint..."
else
    echo "🆕 Starting fresh training..."
fi

# Execute training
eval $CMD

echo "✅ Simplified Cross-Attention MAV training complete! Saved to: $OUTPUT_DIR"