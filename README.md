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
│   │   └── train_iav.py        # Training module with multi-component loss
│   ├── data/
│   │   └── data_utils.py       # Dataset utilities
│   ├── evaluation/
│   │   └── evaluate.py         # Evaluation scripts
│   ├── config.py               # Configuration management
│   └── inference.py            # Inference with controllable alpha
└── paper/
    └── paper.tex              # Research paper
```

## Quick Start

### Training

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