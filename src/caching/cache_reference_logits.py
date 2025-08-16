#!/usr/bin/env python3
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
    max_samples: int = None,
    split: str = "train"  # Add split parameter for train/test
):
    
    print(f"Loading reference model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    ).to(device)
    reference_model.eval()
    
    print(f"Loading {split} dataset...")
    data = HHRLHFDataset.load_data(split=split)
    dataset = PreferenceDataset(data, tokenizer, seq_length)
    
    if max_samples:
        print(f"Limiting to first {max_samples} samples")
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Caching reference logits for {len(dataset)} samples (chosen + rejected) from {split} split...")
    cache_data = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Caching")):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Cache reference logits for chosen responses
            ref_outputs_chosen = reference_model(
                input_ids=batch["input_ids_chosen"],
                attention_mask=batch["attention_mask_chosen"]
            )
            
            # Cache reference logits for rejected responses
            ref_outputs_rejected = reference_model(
                input_ids=batch["input_ids_rejected"],
                attention_mask=batch["attention_mask_rejected"]
            )
            
            for i in range(len(batch["input_ids_chosen"])):
                sample_idx = batch_idx * batch_size + i
                
                # Chosen data
                input_ids_chosen = batch["input_ids_chosen"][i]
                full_logits_chosen = ref_outputs_chosen.logits[i]
                token_logits_chosen = torch.gather(full_logits_chosen, 1, input_ids_chosen.unsqueeze(1)).squeeze(1)
                
                # Rejected data
                input_ids_rejected = batch["input_ids_rejected"][i]
                full_logits_rejected = ref_outputs_rejected.logits[i]
                token_logits_rejected = torch.gather(full_logits_rejected, 1, input_ids_rejected.unsqueeze(1)).squeeze(1)
                
                cache_data[sample_idx] = {
                    # Only store the essential reference logits, not the input data
                    # Input data is already in the dataset
                    "ref_token_logits_chosen": token_logits_chosen.cpu().half(),  # Use half precision to save space
                    "ref_token_logits_rejected": token_logits_rejected.cpu().half(),  # Use half precision to save space
                    
                    # Keep hashes for validation/debugging
                    "input_hash_chosen": hash(tuple(input_ids_chosen.cpu().tolist())),
                    "input_hash_rejected": hash(tuple(input_ids_rejected.cpu().tolist()))
                }
    
    print(f"Saving cache to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Cache saved! {len(cache_data)} samples cached.")
    cache_size_gb = os.path.getsize(cache_file) / 1024**3
    print(f"Cache size: {cache_size_gb:.2f} GB")
    
    # Calculate expected size: samples * seq_length * 2 (chosen+rejected) * 2 bytes (half precision)
    expected_size_mb = len(cache_data) * seq_length * 2 * 2 / 1024 / 1024
    print(f"Expected size: ~{expected_size_mb:.0f} MB (half precision, chosen + rejected logits only)")
    
    if cache_size_gb > 5:
        print("WARNING: Cache too large!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="argsearch/llama-7b-sft-float32")
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cache_file", default="reference_logits_cache.pkl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, help="Limit cache to N samples for testing")
    parser.add_argument("--split", default="train", choices=["train", "test"], 
                        help="Dataset split to cache")
    
    args = parser.parse_args()
    
    cache_reference_logits(
        model_name=args.model_name,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        cache_file=args.cache_file,
        device=args.device,
        max_samples=args.max_samples,
        split=args.split
    )