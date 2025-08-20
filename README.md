# Modular Alignment Vectors (MAV): A Plug-and-Play Framework for Test-Time LLM Alignment

MAV implementation with **multi-layer hidden state analysis** that works with **any transformer model** via adaptive pooling. Modular design allows training separate alignment models for different architectures.

## 🧠 Core Innovation: Multi-Layer Analysis

Unlike traditional alignment methods that only see the model's final output, MAV analyzes the **entire reasoning process** by examining hidden states from multiple layers:

- **Traditional Alignment**: Sees only the final sentence → Limited corrections
- **MAV**: Sees the full thought process (first, middle, last layers) → Expert-level alignment

This is like having an expert editor who reads the entire chapter, not just the conclusion.

## Overview

MAV features:
- **Frozen base model**: Any LLaMA-family model (7B, 13B, 70B, Mistral, etc.)
- **591.4M parameter alignment model** for maximum capacity
- **Multi-layer alignment vectors**: Analyzes hidden states from multiple layers
- **Runtime model switching**: Change base models without retraining alignment
- **Layer fusion technology**: Intelligently combines information from different layers

The final output is computed as: `z_final = z_base + α * alignment_vector`

Where `alignment_vector` is derived from fusing multiple layer representations.

## Key Features

- **Maximum capacity** - 591.4M parameters (36.1x more than GenARM)
- **Multi-layer intelligence** - Analyzes model's entire reasoning chain
- **Multi-model compatibility** with any transformer model
- **Adaptive pooling** automatically handles different hidden sizes
- **Runtime base model switching** without retraining alignment
- **Controllable alignment strength** via α parameter at inference time
- **Parameter efficiency**: Only alignment model trains, base stays frozen

## Installation

```bash
# Create conda environment
conda create -n pav python=3.10
conda activate pav

# Install dependencies
pip install torch transformers datasets accelerate tqdm
```

## Project Structure

```
MAV/
├── models.py                     # MAV architecture
├── data.py                       # HH-RLHF dataset utilities  
├── train.py                      # MAV training script
├── inference.py                  # MAV inference with model switching
├── train.sh                      # Training launcher script
├── accelerate_config.yaml        # Accelerate configuration
└── requirements.txt              # Python dependencies
```

## Quick Start

### 1. Training

```bash
# Default training (591.4M params, auto layer selection)
./train.sh

# Train with uniform layer selection
LAYER_SELECTION=uniform ./train.sh

# Custom hyperparameters
BETA=0.2 LAMBDA_L2=0.001 LR=5e-5 ./train.sh
```

### 2. Inference with Model Switching

```bash
# Basic inference
python inference.py \
    --alignment_checkpoint ./outputs/pav/best_alignment.pt \
    --prompt "How can I help you?"

# Switch base models at runtime
python inference.py \
    --alignment_checkpoint ./outputs/pav/best_alignment.pt \
    --base_model mistralai/Mistral-7B-v0.1 \
    --prompt "How can I help you?"

# Demo model switching capabilities
python inference.py \
    --alignment_checkpoint ./outputs/pav/best_alignment.pt \
    --demo_model_switching \
    --prompt "How can I help you?"
```

### 3. Advanced Training Options

```bash
python train.py \
    --model_name argsearch/llama-7b-sft-float32 \
    --layer_selection auto \
    --output_dir ./outputs/pav \
    --num_epochs 1 \
    --batch_size 8 \
    --grad_accum 2 \
    --learning_rate 1e-4 \
    --beta 0.1 \
    --lambda_l2 0.001 \
    --max_length 512 \
    --bf16
```

## Architecture Details

### Alignment Architecture

MAV uses a large-capacity architecture for superior alignment:

```
Input: Multi-layer hidden states (3 layers × 4096 dims)
   ↓
Fusion Layer: 12288 → 4096 (with ReLU and Dropout)
   ↓
Alignment Network: 4096 → 16384 → 32000
   ↓
Output: Alignment vector (32000 dims)
```

**Total Parameters**: 591.4M (36.1x more than GenARM)

### Multi-Layer Analysis

MAV extracts and analyzes hidden states from multiple layers:

1. **Layer Selection Strategies**:
   - `auto`: First, middle, and last layers (default)
   - `uniform`: Evenly spaced layers throughout the model
   - `custom`: User-defined layer indices

2. **Fusion Mechanism**:
   ```
   Layer 1 Hidden States ─┐
   Layer N/2 Hidden States ├─→ Fusion Layer → Alignment Network → Alignment Vector
   Layer N Hidden States ──┘
   ```

### Adaptive Pooling

Automatically handles different model sizes:
- **LLaMA-7B (4096)**: No pooling needed
- **LLaMA-13B (5120)**: Down-sample to 4096
- **LLaMA-70B (8192)**: Down-sample to 4096  
- **Smaller models (3072)**: Up-sample to 4096

### Training Efficiency

- **Frozen base model**: 6.7B parameters stay frozen
- **Only alignment trains**: 591.4M trainable parameters  
- **Multi-layer fusion**: Additional fusion layer for combining layer information
- **Memory efficient**: Base model cached, alignment model in GPU memory
- **BFloat16 training**: Mixed precision for efficiency

## Inference

```python
from models import create_mav
from inference import MAVInference

# Create inference model (automatically multi-layer)
inference_model = MAVInference(
    alignment_checkpoint_path="./outputs/mav/best_alignment.pt",
    initial_base_model="argsearch/llama-7b-sft-float32"
)

# Generate with different alignment strengths
prompt = "How can I help you today?"

# No alignment (raw model) 
response_raw = inference_model.generate_response(prompt, alpha=0.0)

# Standard alignment
response_aligned = inference_model.generate_response(prompt, alpha=1.0)

# Strong alignment
response_strong = inference_model.generate_response(prompt, alpha=2.0)

# Switch base model at runtime
inference_model.switch_base_model("mistralai/Mistral-7B-v0.1")
response_mistral = inference_model.generate_response(prompt, alpha=1.0)
```

## Alpha Parameter Guide

The α parameter controls the strength of multi-layer alignment at inference time:

- `α = 0.0`: Raw base model (no alignment)
- `α = 0.5`: Light alignment
- `α = 1.0`: Standard alignment (training default)
- `α = 1.5`: Strong alignment
- `α = 2.0`: Maximum alignment

## Supported Models

MAV supports any transformer model by training separate alignment models:

### ✅ Compatible Models
- **LLaMA family**: LLaMA-7B, LLaMA-13B, LLaMA-70B (vocab_size=32000)
- **CodeLLaMA**: All variants (7B, 13B, 34B) (vocab_size=32000)
- **Mistral**: Mistral-7B-v0.1 (vocab_size=32000)
- **Gemma**: All variants (vocab_size=256000)
- **Phi**: All variants (vocab_size=32064)
- **Qwen**: All variants (vocab_size=151936)
- **Fine-tuned models**: Any architecture with different vocab sizes

### Training Strategy
- **Per-model training**: Train separate MAV for each base model architecture
- **Vocab-size specific**: Each MAV trained for specific vocab_size
- **Cross-model inference**: No runtime switching between different vocab_size models

## Model Switching Examples

```bash
# Train on LLaMA-7B
python train.py --model_name argsearch/llama-7b-sft-float32

# Use trained alignment with different models
python inference.py --base_model meta-llama/Llama-2-13b-hf --alignment_checkpoint ./outputs/pav/best_alignment.pt
python inference.py --base_model codellama/CodeLlama-7b-hf --alignment_checkpoint ./outputs/pav/best_alignment.pt  
python inference.py --base_model mistralai/Mistral-7B-v0.1 --alignment_checkpoint ./outputs/pav/best_alignment.pt
```

## Key Parameters

- `--layer_selection`: Layer selection strategy (auto, uniform)
- `--beta`: Weight for preference loss term (default: 0.1)
- `--lambda_l2`: Weight for L2 regularization on alignment vectors (default: 0.001)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--batch_size`: Batch size (default: 8 for memory efficiency)
- `--max_length`: Sequence length (default: 512 for memory efficiency)

## Performance Comparison

| Method | Parameters | Layer Analysis | Inference | Universal |
|---------|------------|---------------|-----------|-----------|
| **GenARM** | 16.4M | Single | 2x forward pass | ❌ |
| **MAV** | 591.4M + fusion | **Multi-layer** | 1x forward pass | ✅ |

**MAV Advantages**:
- **36.1x more parameters** than GenARM for superior alignment capacity
- **Multi-layer intelligence**: Analyzes entire reasoning process
- **Single forward pass** inference (vs GenARM's dual forward pass)  
- **Multi-model compatibility** across LLaMA family models
- **Runtime model switching** without retraining

## Research Paper

This work will be submitted as:

**"Modular Alignment Vectors (MAV): A Plug-and-Play Framework for Test-Time LLM Alignment"**

MAV introduces a novel approach to test-time alignment that analyzes the model's entire reasoning process through multi-layer hidden state fusion. By examining how the model's understanding evolves from early to late layers, MAV achieves expert-level alignment that can identify and correct issues that begin deep within the model's reasoning chain.

## Technical Innovation

The key innovations of MAV:

1. **Multi-Layer Fusion**: Intelligent combination of layer information
2. **Adaptive Pooling**: Universal compatibility across model sizes
3. **Large Capacity**: 591.4M parameters for superior alignment
4. **Test-Time Control**: Full alignment control at inference

This allows MAV to achieve state-of-the-art alignment performance while maintaining complete flexibility.

## License

MIT License