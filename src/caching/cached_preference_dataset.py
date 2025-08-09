import torch
from torch.utils.data import Dataset
import pickle
import os


class CachedPreferenceDataset(Dataset):
    """
    Preference dataset that loads cached reference logits instead of computing them on-the-fly.
    This dramatically speeds up training by eliminating reference model forward passes.
    """
    
    def __init__(self, base_dataset, cache_file: str):
        self.base_dataset = base_dataset
        self.cache_file = cache_file
        
        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"Cache file {cache_file} not found. Run cache_reference_logits.py first."
            )
        
        print(f"Loading cached reference logits from {cache_file}")
        with open(cache_file, 'rb') as f:
            self.cache_data = pickle.load(f)
        
        print(f"Loaded cache with {len(self.cache_data)} samples")
        
        # Verify cache matches dataset
        if len(self.cache_data) != len(base_dataset):
            raise ValueError(
                f"Cache size ({len(self.cache_data)}) doesn't match dataset size ({len(base_dataset)}). "
                "Please regenerate the cache."
            )
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base sample
        base_sample = self.base_dataset[idx]
        
        # Get cached reference logits
        if idx not in self.cache_data:
            raise KeyError(f"Sample {idx} not found in cache")
        
        cached_sample = self.cache_data[idx]
        
        # Verify input IDs match (safety check)
        if not torch.equal(base_sample["input_ids_chosen"], cached_sample["chosen_input_ids"]):
            raise ValueError(f"Sample {idx}: chosen input_ids mismatch between dataset and cache")
        
        if not torch.equal(base_sample["input_ids_rejected"], cached_sample["rejected_input_ids"]):
            raise ValueError(f"Sample {idx}: rejected input_ids mismatch between dataset and cache")
        
        # Return sample with cached reference logits
        return {
            **base_sample,
            "reference_logits_chosen": cached_sample["chosen_logits"],
            "reference_logits_rejected": cached_sample["rejected_logits"]
        }