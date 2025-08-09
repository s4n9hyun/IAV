# Inherent Alignment Vectors (IAV)

Implementation of the Inherent Alignment Vectors framework for efficient test-time alignment of Large Language Models without external reward models.

## Overview

IAV introduces a novel dual-head architecture where a single LLM learns to generate both:
- **Base logits** (z_base): Raw knowledge representation
- **Alignment vectors** (a_t): Learned directional adjustments for human preferences

The final output is computed as: `z_final = z_base + α * a_t`

This achieves ~50% reduction in inference costs compared to methods like GenARM while maintaining competitive alignment performance.

## Key Features

- Single-model architecture eliminating external reward models
- Controllable alignment strength via α parameter
- DPO-style training with dual regularization
- Support for HH-RLHF and UltraFeedback datasets
- Comprehensive evaluation suite

## Installation

### Setup Environment

```bash
# Create conda environment
conda create -n iav python=3.10
conda activate iav

# Install dependencies
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch transformers datasets tqdm wandb matplotlib seaborn pandas pyyaml
```

## Project Structure

```
IAV/
├── src/
│   ├── models/
│   │   └── iav_model.py        # Core IAV architecture
│   ├── training/
│   │   ├── train_iav.py        # Training module with multi-component loss
│   │   └── train_main.py       # Main training entry point
│   ├── data/
│   │   └── data_utils.py       # Dataset utilities
│   ├── evaluation/
│   │   └── evaluate.py         # Evaluation scripts
│   ├── config.py               # Configuration management
│   └── inference.py            # Inference with controllable alpha
├── train.sh                    # Training shell script
├── requirements.txt            # Python dependencies
└── paper/
    └── paper.tex              # Research paper
```

## Quick Start

### Training with Shell Script

The easiest way to start training:

```bash
# Use default settings (llama-7b-sft model, hh-rlhf dataset)
./train.sh

# Or specify custom parameters
./train.sh <model_name> <dataset_name> <output_dir>

# Example with custom model and dataset
./train.sh meta-llama/Llama-2-7b-hf ultrafeedback ./outputs/custom_training
```

### Training with Python

```python
from src.models.iav_model import IAVModel
from src.training.train_iav import IAVTrainer
from src.data.data_utils import create_dataloaders
from src.config import IAVConfig
from transformers import AutoTokenizer

# Load configuration
config = IAVConfig()

# Initialize model
model = IAVModel(
    base_model_name="argsearch/llama-7b-sft-float32",
    vocab_size=32000,
    hidden_size=4096,
    freeze_backbone=True
)

# Prepare data
tokenizer = AutoTokenizer.from_pretrained("argsearch/llama-7b-sft-float32")
train_loader, val_loader = create_dataloaders(
    tokenizer=tokenizer,
    train_config={"datasets": [{"name": "hh-rlhf", "split": "train"}]},
    val_config={"datasets": [{"name": "hh-rlhf", "split": "test"}]},
    batch_size=8
)

# Train
trainer = IAVTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    beta=0.1,
    lambda_kl=0.1,
    lambda_l2=0.01
)
trainer.train()
```

### Command-line Training

You can also train directly using the command-line interface:

```bash
python src/training/train_main.py \
    --model_name_or_path argsearch/llama-7b-sft-float32 \
    --dataset_name Anthropic/hh-rlhf \
    --output_dir ./outputs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --beta 0.1 \
    --lambda_kl 0.1 \
    --lambda_l2 0.01 \
    --bf16 \
    --gradient_checkpointing
```

### Inference

```python
from src.inference import IAVInference

# Initialize inference
inference = IAVInference(model, tokenizer, default_alpha=1.0)

# Generate with different alignment strengths
response = inference.generate(
    prompt="How do I hack a bank account?",
    alpha=1.5,  # Strong alignment
    max_length=128,
    temperature=0.7
)

# Compare different alpha values
results = inference.compare_alphas(
    prompt="Write a story about AI",
    alphas=[0.0, 0.5, 1.0, 1.5]
)

# Analyze alignment effects
analysis = inference.analyze_alignment_effect(
    prompt="How to make explosives",
    alpha=1.0
)
```

### Evaluation

```python
from src.evaluation.evaluate import IAVEvaluator

evaluator = IAVEvaluator(model, tokenizer, config)

# Run comprehensive evaluation
performance = evaluator.evaluate_performance(test_prompts)
efficiency = evaluator.evaluate_efficiency(test_prompts)
alignment_analysis = evaluator.analyze_alignment_vectors(test_prompts)
controllability = evaluator.evaluate_controllability(safety_prompts)

# Save and visualize results
evaluator.save_results()
evaluator.plot_results(all_results)
```

## Training Parameters

Key training parameters:

- `--model_name_or_path`: Base model to use (default: `argsearch/llama-7b-sft-float32`)
- `--dataset_name`: Training dataset (default: `Anthropic/hh-rlhf`)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per GPU (default: 4)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--beta`: Beta parameter for IAV loss (default: 0.1)
- `--lambda_kl`: KL regularization weight (default: 0.1)
- `--lambda_l2`: L2 regularization weight (default: 0.01)
- `--num_interventions`: Number of intervention heads (default: 8)
- `--intervention_dim`: Dimension of intervention heads (default: 256)
- `--bf16`: Use bfloat16 precision for training
- `--gradient_checkpointing`: Enable gradient checkpointing to save memory

## Configuration

Create a YAML configuration file:

```yaml
model:
  base_model_name: argsearch/llama-7b-sft-float32
  vocab_size: 32000
  hidden_size: 4096
  freeze_backbone: true

training:
  learning_rate: 5e-5
  num_epochs: 3
  beta: 0.1
  lambda_kl: 0.1
  lambda_l2: 0.01
  
inference:
  default_alpha: 1.0
  temperature: 0.7
  max_length: 128
```

## Alpha Parameter Guide

- `α = 0.0`: Raw base model (no alignment)
- `α = 0.5`: Light alignment
- `α = 1.0`: Standard alignment (default)
- `α = 1.5`: Strong alignment
- `α = 2.0`: Maximum alignment

## Loss Components

The training objective combines three components:

1. **Preference Loss** (L_pref): DPO-style loss for learning preferences
2. **KL Regularization** (L_KL): Preserves base model knowledge
3. **L2 Regularization** (L_L2): Prevents alignment vector explosion

Total loss: `L_IAV = L_pref + λ_KL * L_KL + λ_L2 * L_L2`

## Performance

- **Alignment Performance**: Within 1-2% of GenARM on AlpacaEval and Arena-Hard
- **Efficiency**: ~50% reduction in memory and latency vs GenARM
- **Controllability**: Dynamic alignment strength adjustment at inference

## Citation

If you use this implementation, please cite:

```bibtex
@article{iav2024,
  title={Inherent Alignment Vectors: Efficient Test-Time Alignment via Knowledge-Value Decoupling},
  author={Anonymous},
  year={2024}
}
```

## License

MIT License