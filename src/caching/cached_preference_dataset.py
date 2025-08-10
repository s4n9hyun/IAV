import torch
from torch.utils.data import Dataset
import pickle
import os


class CachedPreferenceDataset(Dataset):
    
    def __init__(self, base_dataset, cache_file: str):
        self.base_dataset = base_dataset
        self.cache_file = cache_file
        
        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"Cache file {cache_file} not found. Run ./cache.sh first."
            )
        
        print(f"Loading cached reference logits from {cache_file}")
        with open(cache_file, 'rb') as f:
            self.cache_data = pickle.load(f)
        
        cache_size = len(self.cache_data)
        dataset_size = len(base_dataset)
        print(f"Loaded cache with {cache_size} samples (dataset has {dataset_size})")
        
        if cache_size < dataset_size:
            print(f"WARNING: Using first {cache_size} samples.")
    
    def __len__(self):
        return min(len(self.base_dataset), len(self.cache_data))
    
    def __getitem__(self, idx):
        base_sample = self.base_dataset[idx]
        
        if idx not in self.cache_data:
            raise KeyError(f"Sample {idx} not found in cache")
        
        cached_sample = self.cache_data[idx]
        
        # Only support dual cache format (chosen + rejected)
        if "ref_token_logits_chosen" not in cached_sample:
            raise ValueError(
                f"Cache file uses old single-cache format. "
                f"Please regenerate cache with both chosen and rejected logits."
            )
        
        # Verify hash match for both chosen and rejected
        input_hash_chosen = hash(tuple(base_sample["input_ids_chosen"].tolist()))
        input_hash_rejected = hash(tuple(base_sample["input_ids_rejected"].tolist()))
        
        if input_hash_chosen != cached_sample["input_hash_chosen"]:
            raise ValueError(f"Sample {idx}: chosen hash mismatch")
        if input_hash_rejected != cached_sample["input_hash_rejected"]:
            raise ValueError(f"Sample {idx}: rejected hash mismatch")
        
        return {
            **base_sample,
            "reference_token_logits_chosen": cached_sample["ref_token_logits_chosen"],
            "reference_token_logits_rejected": cached_sample["ref_token_logits_rejected"],
            "cached_input_ids_chosen": cached_sample["input_ids_chosen"],
            "cached_attention_mask_chosen": cached_sample["attention_mask_chosen"],
            "cached_input_ids_rejected": cached_sample["input_ids_rejected"],
            "cached_attention_mask_rejected": cached_sample["attention_mask_rejected"]
        }