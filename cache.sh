#!/bin/bash

# Cache Reference Logits for IAV Training
# This creates a cache file that speeds up training by eliminating reference model forward passes
# Usage: ./cache.sh [options]

echo "=== IAV Reference Logits Caching ==="
echo "This will create a cache file to speed up training"
echo ""

# Default values
MODEL_NAME="argsearch/llama-7b-sft-float32"
SEQ_LENGTH=1024
BATCH_SIZE=16
MAX_SAMPLES=""
CACHE_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --cache_file)
            CACHE_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_name MODEL     Model name (default: argsearch/llama-7b-sft-float32)"
            echo "  --seq_length LENGTH    Sequence length (default: 1024)"
            echo "  --batch_size SIZE      Batch size for caching (default: 4)"
            echo "  --max_samples N        Limit cache to N samples (default: all)"
            echo "  --cache_file FILE      Output cache file (default: auto-generated)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./cache.sh                                    # Cache with defaults"
            echo "  ./cache.sh --seq_length 512 --batch_size 8   # Custom settings"
            echo "  ./cache.sh --max_samples 10000               # Limit to 10k samples"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Generate default cache filename if not provided
if [ -z "$CACHE_FILE" ]; then
    MODEL_SAFE=$(echo "$MODEL_NAME" | sed 's/\//_/g')
    if [ -n "$MAX_SAMPLES" ]; then
        CACHE_FILE="cache_${MODEL_SAFE}_seq${SEQ_LENGTH}_${MAX_SAMPLES}samples.pkl"
    else
        CACHE_FILE="cache_${MODEL_SAFE}_seq${SEQ_LENGTH}.pkl"
    fi
fi

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Sequence Length: $SEQ_LENGTH"
echo "  Batch Size: $BATCH_SIZE"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  Max Samples: $MAX_SAMPLES"
else
    echo "  Max Samples: All"
fi
echo "  Cache File: $CACHE_FILE"
echo ""

# Check if cache file already exists
if [ -f "$CACHE_FILE" ]; then
    echo "WARNING: Cache file '$CACHE_FILE' already exists!"
    echo "This will overwrite the existing cache."
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Check available GPU memory
echo "Checking GPU status..."
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits

echo ""
echo "Starting cache generation..."
echo "This may take several hours for the full dataset."
echo "Press Ctrl+C to cancel."
echo ""

# Build the command
CMD="python src/caching/cache_reference_logits.py"
CMD="$CMD --model_name $MODEL_NAME"
CMD="$CMD --seq_length $SEQ_LENGTH"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --cache_file $CACHE_FILE"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Run the caching command
echo "Running: $CMD"
echo ""

eval $CMD

# Check if caching was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Caching Complete ==="
    echo "Cache file: $CACHE_FILE"
    if [ -f "$CACHE_FILE" ]; then
        SIZE=$(du -h "$CACHE_FILE" | cut -f1)
        echo "Cache size: $SIZE"
    fi
    echo ""
    echo "You can now use this cache in training by modifying train_main.py"
    echo "to load the cached dataset instead of computing reference logits on-the-fly."
else
    echo ""
    echo "=== Caching Failed ==="
    echo "Check the error messages above for details."
    exit 1
fi