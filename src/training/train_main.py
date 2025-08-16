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


def create_experiment_name(args):
    """Create experiment name based on hyperparameters (following GenARM convention)"""
    # Extract base model name (e.g., "llama-7b-sft" from "argsearch/llama-7b-sft-float32")
    model_name = args.model_name_or_path.split("/")[-1].replace("-float32", "")
    
    # Create naming convention: model_method_dataset_epoch_beta_lambdakl_lambdal2_lr_bs
    exp_name = f"{model_name}-iav-HH-epoch_{args.num_epochs}-beta_{args.beta}-lambda_kl_{args.lambda_kl}-lambda_l2_{args.lambda_l2}-lr_{args.learning_rate}-bs_{args.batch_size}"
    
    return exp_name

def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_name", "--model_name_or_path", dest="model_name_or_path", default="argsearch/llama-7b-sft-float32")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--use_param_naming", action="store_true", help="Use parameter-based directory naming")
    
    # Training
    parser.add_argument("--num_train_epochs", "--num_epochs", dest="num_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", "--grad_accum", dest="grad_accum", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    
    # Schedule
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    
    # IAV hyperparameters - matching DPO/GenARM argument names
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum prompt length")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum total sequence length")
    parser.add_argument("--beta", type=float, default=0.5, help="DPO beta")
    parser.add_argument("--lambda_kl", type=float, default=0.01, help="KL penalty weight")
    parser.add_argument("--lambda_l2", type=float, default=0.1, help="L2 alignment weight")
    
    # Optimization
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    
    # Checkpoint
    parser.add_argument("--cache_file", default="cache_argsearch_llama-7b-sft-float32_seq1024.pkl")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create experiment-specific output directory if requested
    if args.use_param_naming:
        exp_name = create_experiment_name(args)
        args.output_dir = os.path.join(args.output_dir, exp_name)
        print(f"Using parameter-based naming: {args.output_dir}")
    
    # Convert warmup_steps to ratio if provided
    if args.warmup_steps is not None:
        args.warmup_ratio = 0.1  # Approximation, adjusted with dataset size later
    
    # Enable wandb if requested (kept for backward compatibility)
    # No longer needed since we removed report_to argument
    
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
    
    # Load data: split Dahoas train set into 90% train, 10% val
    all_data = HHRLHFDataset.load_data(split="train")
    n_total = len(all_data)
    n_train = int(n_total * 0.9)
    train_data = all_data[:n_train]
    val_data = all_data[n_train:]
    
    if accelerator.is_main_process:
        print(f"Total data: {n_total}, Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Use max_length for sequence length
    max_seq_length = args.max_length
    
    train_dataset = PreferenceDataset(train_data, tokenizer, max_seq_length)
    val_dataset = PreferenceDataset(val_data, tokenizer, max_seq_length)
    
    if accelerator.is_main_process:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Setup cache if available
    reference_model = None
    
    if os.path.exists(args.cache_file):
        print(f"Using cache: {args.cache_file}")
        
        from src.caching.split_cached_dataset import SplitCachedPreferenceDataset
        
        # Map to cache indices (90/10 split)
        n_total = len(train_data) + len(val_data)
        n_train = len(train_data)
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_total))
        
        train_dataset = SplitCachedPreferenceDataset(train_dataset, args.cache_file, train_indices)
        val_dataset = SplitCachedPreferenceDataset(val_dataset, args.cache_file, val_indices)
    else:
        print(f"No cache found: {args.cache_file}")
        print("Run ./cache.sh to speed up training")
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
    
    # Resume from checkpoint
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        
        # Auto-find latest checkpoint if "True"
        if checkpoint_path.lower() == "true":
            import glob
            checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                checkpoint_path = checkpoints[-1]
                if accelerator.is_main_process:
                    print(f"Found latest checkpoint: {checkpoint_path}")
            else:
                if accelerator.is_main_process:
                    print("No checkpoints found to resume from")
                checkpoint_path = None
        
        if checkpoint_path and checkpoint_path.lower() != "true":
            if accelerator.is_main_process:
                print(f"Resuming from: {checkpoint_path}")
            checkpoint_info = trainer.load_checkpoint(checkpoint_path)
            # Pass checkpoint info to train method
            trainer.train(start_epoch=checkpoint_info["epoch"], start_step=checkpoint_info["global_step"])
        else:
            trainer.train()
    else:
        if accelerator.is_main_process:
            print("Starting training...")
        trainer.train()


if __name__ == "__main__":
    main()