# Inherent Alignment Vectors (IAV)

Implementation of the Inherent Alignment Vectors framework for efficient test-time alignment of Large Language Models without external reward models.

## Overview

IAV introduces a novel dual-head architecture where a single LLM learns to generate both:
- **Base logits** (z_base): Raw knowledge representation from frozen backbone
- **Alignment vectors** (a_t): Learned directional adjustments for human preferences

The final output is computed as: `z_final = z_base + α * a_t`

This achieves ~50% reduction in inference costs compared to methods like GenARM while maintaining competitive alignment performance.

## Key Features

- **Single-model architecture** eliminating external reward models
- **Controllable alignment strength** via α parameter at inference time
- **DPO-style training** with dual KL regularization (chosen + rejected)
- **Efficient caching system** for reference model logits
- **Multi-GPU support** with automatic device detection
- **Memory-efficient training** with gradient checkpointing

## Installation

```bash
# Create conda environment
conda create -n iav python=3.10
conda activate iav

# Install dependencies
pip install torch transformers datasets accelerate tqdm wandb pyyaml
```

## Project Structure

```
IAV/
├── src/
│   ├── models/
│   │   └── iav_model.py          # Core IAV dual-head architecture
│   ├── training/
│   │   ├── train_iav.py          # IAVLoss and IAVTrainer implementation
│   │   └── train_main.py         # Main training entry point
│   ├── data/
│   │   ├── data_utils.py         # Dataset utilities
│   │   └── datasets.py           # HH-RLHF dataset loader
│   ├── caching/
│   │   ├── cache_reference_logits.py  # Generate reference logits cache
│   │   └── cached_preference_dataset.py # Cached dataset wrapper
│   └── inference.py              # Inference with controllable alpha
├── train.sh                      # Training launcher script
├── cache.sh                      # Cache generation script
└── requirements.txt              # Python dependencies
```

## Quick Start

### 1. Generate Reference Logits Cache (Recommended)

Before training, generate a cache of reference model logits to significantly speed up training:

```bash
# Generate dual cache (chosen + rejected) for KL regularization
./cache.sh

# Or manually:
python src/caching/cache_reference_logits.py \
    --model_name argsearch/llama-7b-sft-float32 \
    --dataset_name Dahoas/full-hh-rlhf \
    --split train \
    --max_seq_length 2048 \
    --output_path ./cache/reference_logits_dual_hh_train_2048.pt
```

This creates a cache containing reference model logits for both chosen and rejected responses, enabling efficient dual KL regularization during training.

### 2. Training

```bash
# Use default settings (auto-detects GPU count)
./train.sh

# Custom parameters
./train.sh <model_name> <dataset_name> <output_dir>

# Example with specific configuration
./train.sh argsearch/llama-7b-sft-float32 hh-rlhf ./outputs/iav_training
```

The training script automatically:
- Detects available GPUs and uses `accelerate` for multi-GPU training
- Falls back to single GPU if only one is available
- Uses cached reference logits if available
- Saves checkpoints every 500 steps

### 3. Command-line Training Options

```bash
python src/training/train_main.py \
    --model_name_or_path argsearch/llama-7b-sft-float32 \
    --output_dir ./outputs \
    --cache_file ./cache/reference_logits_dual_hh_train_2048.pt \
    --num_epochs 1 \
    --batch_size 4 \
    --grad_accum 1 \
    --learning_rate 5e-5 \
    --beta 0.1 \
    --lambda_kl 0.1 \
    --lambda_l2 0.01 \
    --seq_length 2048 \
    --warmup_ratio 0.1 \
    --save_steps 500 \
    --eval_steps 100 \
    --bf16 \
    --gradient_checkpointing
```

## Training Details

### Loss Function

The IAV training objective combines three components:

```
L_total = L_pref + λ_kl * L_kl + λ_l2 * L_l2
```

Where:
- **L_pref**: DPO-style preference loss
- **L_kl**: KL divergence from reference model (applied to both chosen and rejected)
- **L_l2**: L2 regularization on alignment vectors

### Dual KL Regularization

IAV applies KL regularization to **both chosen and rejected responses** to prevent the model from gaming the loss:

```python
L_kl = (L_kl_chosen + L_kl_rejected) / 2.0
```

This ensures the base head maintains consistency with the reference model for all types of responses.

### Memory Optimization

- **Frozen Backbone**: Only trains the dual heads (~2% of parameters)
- **Gradient Checkpointing**: Reduces memory usage during backprop
- **Reference Logits Caching**: Eliminates need for reference model in memory
- **BFloat16 Training**: Uses mixed precision for efficiency

## Inference

```python
from src.models.iav_model import IAVModel
from transformers import AutoTokenizer

# Load model and tokenizer
model = IAVModel.from_pretrained("./outputs/iav_training/checkpoint-5000")
tokenizer = AutoTokenizer.from_pretrained("argsearch/llama-7b-sft-float32")

# Generate with different alignment strengths
prompt = "How can I help you today?"

# No alignment (raw model)
response_raw = model.generate(prompt, alpha=0.0)

# Standard alignment
response_aligned = model.generate(prompt, alpha=1.0)

# Strong alignment
response_strong = model.generate(prompt, alpha=1.5)
```

## Alpha Parameter Guide

The α parameter controls the strength of alignment at inference time:

- `α = 0.0`: Raw base model (no alignment)
- `α = 0.5`: Light alignment
- `α = 1.0`: Standard alignment (training default)
- `α = 1.5`: Strong alignment
- `α = 2.0`: Maximum alignment

## Multi-GPU Training

The training script automatically detects and uses available GPUs:

```bash
# Uses all available GPUs with accelerate
./train.sh

# Or manually specify GPUs
CUDA_VISIBLE_DEVICES=0,1 ./train.sh
```

## Evaluation

Generate responses for evaluation:

```bash
cd ../evaluation
python scripts/generate_iav.py 300 --dataset hh-rlhf
python scripts/generate_iav.py 300 --dataset alpaca_eval
python scripts/generate_iav.py 300 --dataset arena_hard
```

## Troubleshooting

### Out of Memory (OOM)

1. **Enable gradient checkpointing**: Add `--gradient_checkpointing`
2. **Reduce batch size**: Use `--batch_size 2` or `--batch_size 1`
3. **Use gradient accumulation**: Set `--grad_accum 4`
4. **Ensure cache is used**: Check that `--cache_file` points to valid cache

### Slow Training

1. **Generate cache first**: Run `./cache.sh` before training
2. **Check cache is loaded**: Look for "Using cached reference logits" in logs
3. **Use multiple GPUs**: Training automatically uses all available GPUs

### Cache Issues

If you see "Cache file uses old single-cache format" error:
- Regenerate cache with dual format using `./cache.sh`
- The new cache includes both chosen and rejected reference logits

## Key Parameters

- `--beta`: Weight for preference loss term (default: 0.1)
- `--lambda_kl`: Weight for KL regularization (default: 0.1)  
- `--lambda_l2`: Weight for L2 regularization on alignment vectors (default: 0.01)
- `--save_steps`: Checkpoint saving frequency (default: 500)
- `--eval_steps`: Validation frequency (default: 100)

## Performance

- **Training Speed**: ~3 seconds/iteration with cache, ~12s without
- **Memory Usage**: ~40GB for 7B model with batch_size=4
- **Alignment Quality**: Competitive with GenARM and DPO
- **Inference Speed**: 2x faster than GenARM (single forward pass)

## License

MIT License