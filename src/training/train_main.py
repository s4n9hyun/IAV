#!/usr/bin/env python
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.train_iav import IAVTrainer
from src.models.iav_model import IAVModel
from src.data.data_utils import PreferenceDataset, HHRLHFDataset


def main():
    parser = argparse.ArgumentParser(description="Train IAV model")
    parser.add_argument("--model_name_or_path", default="argsearch/llama-7b-sft-float32")
    parser.add_argument("--dataset_name", default="Anthropic/hh-rlhf")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--num_train_epochs", type=int, default=1)  # Reduce epochs for faster iteration
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)  # Smaller batch for memory
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)  # Increase to maintain effective batch size
    parser.add_argument("--learning_rate", type=float, default=2e-5)  # Lower LR for more stable training
    parser.add_argument("--warmup_ratio", type=float, default=0.03)  # Shorter warmup
    parser.add_argument("--max_seq_length", type=int, default=1024)  # Shorter sequences for efficiency
    parser.add_argument("--beta", type=float, default=0.5)  # Increase beta for stronger preference signal
    parser.add_argument("--lambda_kl", type=float, default=0.01)  # Reduce KL weight (was too high)
    parser.add_argument("--lambda_l2", type=float, default=0.1)  # Increase L2 to regularize alignment vectors
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    
    # Unused but kept for compatibility
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--num_interventions", type=int, default=8)
    parser.add_argument("--intervention_dim", type=int, default=256)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--wandb_project", default="iav-training")
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.bf16 and device.type == "cuda":
        torch.set_default_dtype(torch.bfloat16)
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load tokenizer and config
    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    # Create IAV model
    print("Creating IAV model")
    model = IAVModel(
        base_model_name=args.model_name_or_path,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        device=device,
        freeze_backbone=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    ).to(device)
    
    if args.gradient_checkpointing and hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    train_data = HHRLHFDataset.load_data(split="train")
    train_dataset = PreferenceDataset(train_data, tokenizer, args.max_seq_length)
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        shuffle=True
    )
    
    # Create reference model
    print("Loading reference model for KL regularization")
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    ).to(device)
    
    # Calculate warmup steps
    num_training_steps = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    # Create trainer
    trainer = IAVTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_train_epochs,
        warmup_steps=num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        beta=args.beta,
        lambda_kl=args.lambda_kl,
        lambda_l2=args.lambda_l2,
        reference_model=reference_model,
        device=device.type,
        log_wandb=args.use_wandb,
        save_dir=args.output_dir
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == "__main__":
    main()