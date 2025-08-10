#!/usr/bin/env python
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.train_iav import IAVTrainer
from src.models.iav_model import IAVModel
from src.data.data_utils import PreferenceDataset, HHRLHFDataset
from src.caching.cached_preference_dataset import CachedPreferenceDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="argsearch/llama-7b-sft-float32")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lambda_kl", type=float, default=0.01)
    parser.add_argument("--lambda_l2", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--cache_file", default="cache_argsearch_llama-7b-sft-float32_seq1024.pkl", help="Path to cached reference logits file")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint file to resume training from")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="bf16" if args.bf16 else "no")
    
    torch.manual_seed(42)
    device = accelerator.device
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print device info on main process only
    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} GPU(s)")
        print(f"Main process device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    # Create model
    model = IAVModel(
        base_model_name=args.model_name_or_path,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        device="cpu",  # Let accelerator handle device placement
        freeze_backbone=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    )
    
    if args.gradient_checkpointing and hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable()
    
    # Load datasets
    train_data = HHRLHFDataset.load_data(split="train")
    val_data = HHRLHFDataset.load_data(split="test")  # Use test split for validation
    
    base_train_dataset = PreferenceDataset(train_data, tokenizer, args.seq_length)
    base_val_dataset = PreferenceDataset(val_data, tokenizer, args.seq_length)
    
    # Use full test split for validation (8552 samples)
    if accelerator.is_main_process:
        print(f"Using full validation set: {len(base_val_dataset)} samples from test split")
    
    # Initialize reference_model as None - will only be loaded if no caches exist
    reference_model = None
    
    # Check for validation cache file
    # Handle both old (cache_dual_*) and new (cache_*) naming patterns
    if "cache_dual_" in args.cache_file:
        val_cache_file = args.cache_file.replace("cache_dual_", "cache_val_dual_")
    else:
        val_cache_file = args.cache_file.replace("cache_", "cache_val_dual_")
    
    # Setup training dataset with cache if available
    if os.path.exists(args.cache_file):
        if accelerator.is_main_process:
            print(f"Using cached reference logits for training: {args.cache_file}")
        train_dataset = CachedPreferenceDataset(base_train_dataset, args.cache_file)
    else:
        if accelerator.is_main_process:
            print(f"WARNING: Training cache file {args.cache_file} not found!")
            print("Run ./cache.sh first to generate cache for faster training")
        train_dataset = base_train_dataset
    
    # Setup validation dataset with cache if available
    if os.path.exists(val_cache_file):
        if accelerator.is_main_process:
            print(f"Using cached reference logits for validation: {val_cache_file}")
        val_dataset = CachedPreferenceDataset(base_val_dataset, val_cache_file)
    else:
        if accelerator.is_main_process:
            print(f"WARNING: Validation cache file {val_cache_file} not found!")
            print("Run ./cache_val.sh to generate validation cache")
        val_dataset = base_val_dataset
    
    # Only load reference model if neither cache exists
    if not os.path.exists(args.cache_file) or not os.path.exists(val_cache_file):
        if accelerator.is_main_process:
            print("Loading reference model (at least one cache is missing)...")
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            attn_implementation="flash_attention_2"
        )
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Calculate warmup steps
    num_training_steps = len(train_dataloader) * args.num_epochs // args.grad_accum
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    # Prepare everything with accelerator
    if reference_model is not None:
        model, reference_model, train_dataloader, val_dataloader = accelerator.prepare(
            model, reference_model, train_dataloader, val_dataloader
        )
    else:
        model, train_dataloader, val_dataloader = accelerator.prepare(
            model, train_dataloader, val_dataloader
        )
    
    # Train
    trainer = IAVTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=num_warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
        beta=args.beta,
        lambda_kl=args.lambda_kl,
        lambda_l2=args.lambda_l2,
        reference_model=reference_model,
        accelerator=accelerator,  # Pass accelerator to trainer
        save_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        log_wandb=args.use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if accelerator.is_main_process:
            print(f"Resuming training from: {args.resume_from_checkpoint}")
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    if accelerator.is_main_process:
        print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()