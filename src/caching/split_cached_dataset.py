import torch
from torch.utils.data import Dataset
import pickle
import os


class SplitCachedPreferenceDataset(Dataset):
    """Cached dataset with train/val splitting."""
    
    def __init__(self, base_dataset, cache_file: str, split_indices: list):
        """
        Args:
            base_dataset: Base dataset
            cache_file: Cache file path
            split_indices: Indices for this split
        """
        self.base_dataset = base_dataset
        self.cache_file = cache_file
        self.split_indices = split_indices
        
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file {cache_file} not found. Run ./cache.sh first.")
        
        print(f"Loading cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            self.full_cache_data = pickle.load(f)
        
        print(f"Using {len(split_indices)} samples")
    
    def __len__(self):
        return len(self.split_indices)
    
    def __getitem__(self, idx):
        cache_idx = self.split_indices[idx]
        base_sample = self.base_dataset[idx]
        
        if cache_idx not in self.full_cache_data:
            raise KeyError(f"Sample {cache_idx} not in cache")
        
        cached_sample = self.full_cache_data[cache_idx]
        
        if "ref_token_logits_chosen" not in cached_sample:
            raise ValueError("Invalid cache format")
        
        # Verify data integrity
        input_hash_chosen = hash(tuple(base_sample["input_ids_chosen"].tolist()))
        input_hash_rejected = hash(tuple(base_sample["input_ids_rejected"].tolist()))
        
        if input_hash_chosen != cached_sample["input_hash_chosen"]:
            raise ValueError(f"Hash mismatch: chosen {cache_idx}")
        if input_hash_rejected != cached_sample["input_hash_rejected"]:
            raise ValueError(f"Hash mismatch: rejected {cache_idx}")
        
        return {
            **base_sample,
            "reference_token_logits_chosen": cached_sample["ref_token_logits_chosen"],
            "reference_token_logits_rejected": cached_sample["ref_token_logits_rejected"]
        }