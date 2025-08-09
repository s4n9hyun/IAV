#!/usr/bin/env python
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.train_iav import IAVTrainer, IAVLoss
from src.models.iav_model import IAVModel
from src.data.data_utils import prepare_preference_dataset


def main():
    parser = argparse.ArgumentParser(description="Train IAV model")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="argsearch/llama-7b-sft-float32",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf",
                        help="Dataset name from huggingface datasets")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size per device during evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    # IAV specific arguments
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Beta parameter for IAV loss")
    parser.add_argument("--lambda_kl", type=float, default=0.1,
                        help="KL regularization weight")
    parser.add_argument("--lambda_l2", type=float, default=0.01,
                        help="L2 regularization weight")
    parser.add_argument("--num_interventions", type=int, default=8,
                        help="Number of intervention heads")
    parser.add_argument("--intervention_dim", type=int, default=256,
                        help="Dimension of intervention heads")
    
    # Other arguments
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Run evaluation every X steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--tf32", action="store_true",
                        help="Use tf32 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="iav-training",
                        help="WandB project name")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.bf16 and device.type == "cuda":
        torch.set_default_dtype(torch.bfloat16)
    
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print(f"Loading base model from {args.model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto" if torch.cuda.device_count() > 1 else None
    )
    
    # Create IAV model
    print("Creating IAV model")
    model = IAVModel(
        base_model=base_model,
        num_interventions=args.num_interventions,
        intervention_dim=args.intervention_dim,
        hidden_size=base_model.config.hidden_size
    )
    
    if device.type == "cuda":
        model = model.to(device)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Load and prepare dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    
    # Prepare training data
    train_dataset = prepare_preference_dataset(
        dataset["train"] if "train" in dataset else dataset,
        tokenizer=tokenizer,
        max_length=args.max_seq_length
    )
    
    # Prepare validation data if available
    val_dataset = None
    if "validation" in dataset or "test" in dataset:
        val_split = "validation" if "validation" in dataset else "test"
        val_dataset = prepare_preference_dataset(
            dataset[val_split],
            tokenizer=tokenizer,
            max_length=args.max_seq_length
        )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Create scheduler
    from transformers import get_linear_schedule_with_warmup
    
    num_training_steps = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Create loss function with reference model
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto" if torch.cuda.device_count() > 1 else None
    )
    if device.type == "cuda":
        reference_model = reference_model.to(device)
    
    loss_fn = IAVLoss(
        beta=args.beta,
        lambda_kl=args.lambda_kl,
        lambda_l2=args.lambda_l2,
        reference_model=reference_model
    )
    
    # Create trainer
    trainer = IAVTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        num_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        save_dir=args.output_dir,
        log_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()