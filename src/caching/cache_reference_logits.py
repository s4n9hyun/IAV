#!/usr/bin/env python3
"""
Cache reference model logits for all training samples to speed up IAV training.
This eliminates the need for reference model forward passes during training.
"""
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.data_utils import PreferenceDataset, HHRLHFDataset


def cache_reference_logits(
    model_name: str = "argsearch/llama-7b-sft-float32",
    seq_length: int = 1024,
    batch_size: int = 4,
    cache_file: str = "reference_logits_cache.pkl",
    device: str = "cuda",
    max_samples: int = None
):
    """Cache reference model logits for all training samples."""
    
    print(f"Loading reference model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    ).to(device)
    reference_model.eval()
    
    print("Loading training dataset...")
    train_data = HHRLHFDataset.load_data(split="train")
    train_dataset = PreferenceDataset(train_data, tokenizer, seq_length)
    
    if max_samples:
        print(f"Limiting to first {max_samples} samples for testing")
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(max_samples, len(train_dataset))))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Caching reference logits for {len(train_dataset)} samples...")
    cache_data = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Caching")):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Compute reference logits for chosen samples
            ref_outputs_chosen = reference_model(
                input_ids=batch["input_ids_chosen"],
                attention_mask=batch["attention_mask_chosen"]
            )
            
            # Compute reference logits for rejected samples  
            ref_outputs_rejected = reference_model(
                input_ids=batch["input_ids_rejected"],
                attention_mask=batch["attention_mask_rejected"]
            )
            
            # Store logits (move to CPU to save GPU memory)
            for i in range(len(batch["input_ids_chosen"])):
                sample_idx = batch_idx * batch_size + i
                cache_data[sample_idx] = {
                    "chosen_logits": ref_outputs_chosen.logits[i].cpu(),
                    "rejected_logits": ref_outputs_rejected.logits[i].cpu(),
                    "chosen_input_ids": batch["input_ids_chosen"][i].cpu(),
                    "rejected_input_ids": batch["input_ids_rejected"][i].cpu(),
                    "chosen_attention_mask": batch["attention_mask_chosen"][i].cpu(),
                    "rejected_attention_mask": batch["attention_mask_rejected"][i].cpu()
                }
    
    print(f"Saving cache to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Cache saved! {len(cache_data)} samples cached.")
    print(f"Estimated cache size: {os.path.getsize(cache_file) / 1024**3:.2f} GB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="argsearch/llama-7b-sft-float32")
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cache_file", default="reference_logits_cache.pkl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, help="Limit cache to N samples for testing")
    
    args = parser.parse_args()
    
    cache_reference_logits(
        model_name=args.model_name,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        cache_file=args.cache_file,
        device=args.device,
        max_samples=args.max_samples
    )