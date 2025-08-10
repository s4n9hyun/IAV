#!/bin/bash

# Generate validation cache for IAV training
# This creates reference logits cache for validation data (test split, first 500 samples)

echo "=== Generating Validation Cache for IAV Training ==="
echo "This will create reference logits for validation data to avoid OOM issues."
echo ""

CACHE_FILE="cache_val_dual_argsearch_llama-7b-sft-float32_seq1024.pkl"

if [ -f "$CACHE_FILE" ]; then
    echo "Validation cache file already exists: $CACHE_FILE"
    read -p "Do you want to regenerate it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping cache generation."
        exit 0
    fi
fi

echo "Configuration:"
echo "  Model: argsearch/llama-7b-sft-float32"
echo "  Split: test (used as validation set)"
echo "  Samples: ALL 8552 test samples"
echo "  Sequence length: 1024"
echo "  Batch size: 8"
echo "  Output file: $CACHE_FILE"
echo ""

echo "Running cache generation for validation data (full test split)..."
CUDA_VISIBLE_DEVICES=0 python src/caching/cache_reference_logits.py \
    --model_name "argsearch/llama-7b-sft-float32" \
    --seq_length 1024 \
    --cache_file "$CACHE_FILE" \
    --split "test" \
    --batch_size 8

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Validation cache generated successfully: $CACHE_FILE"
    echo "File size: $(ls -lh $CACHE_FILE | awk '{print $5}')"
else
    echo ""
    echo "❌ Cache generation failed!"
    exit 1
fi