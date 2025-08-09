#!/usr/bin/env python
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.train_iav import IAVTrainer
from src.models.iav_model import IAVModel
from src.data.data_utils import PreferenceDataset, HHRLHFDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="argsearch/llama-7b-sft-float32")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lambda_kl", type=float, default=0.01)
    parser.add_argument("--lambda_l2", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    
    torch.manual_seed(42)
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    # Create model
    model = IAVModel(
        base_model_name=args.model_name_or_path,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        device=device,
        freeze_backbone=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    )
    
    if args.gradient_checkpointing and hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable()
    
    # Load dataset
    train_data = HHRLHFDataset.load_data(split="train")
    train_dataset = PreferenceDataset(train_data, tokenizer, args.seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Load reference model for KL regularization
    print("Loading reference model for KL regularization")
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    ).to(device)
    
    # Calculate warmup steps
    num_training_steps = len(train_dataloader) * args.num_epochs // args.grad_accum
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    # Train
    trainer = IAVTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=num_warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
        beta=args.beta,
        lambda_kl=args.lambda_kl,
        lambda_l2=args.lambda_l2,
        reference_model=reference_model,
        device="cuda",
        save_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        log_wandb=args.use_wandb
    )
    
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()