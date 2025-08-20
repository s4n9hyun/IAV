#!/usr/bin/env python3
"""Data utilities."""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List


class PreferenceDataset(Dataset):
    """Preference dataset."""
    
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        chosen_text = f"{prompt} {chosen}"
        rejected_text = f"{prompt} {rejected}"
        
        chosen_encoding = self.tokenizer(
            chosen_text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        
        return {
            "input_ids_chosen": chosen_encoding["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_encoding["attention_mask"].squeeze(0),
            "labels_chosen": chosen_encoding["input_ids"].squeeze(0),
            "input_ids_rejected": rejected_encoding["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_encoding["attention_mask"].squeeze(0),
            "labels_rejected": rejected_encoding["input_ids"].squeeze(0)
        }


def load_preference_data(dataset_name="Dahoas/full-hh-rlhf", split="train"):
    """Load preference dataset with flexible dataset support."""
    hf_dataset = load_dataset(dataset_name, split=split)
    
    data = []
    for item in hf_dataset:
        # Support different dataset formats
        if "prompt" in item and "chosen" in item and "rejected" in item:
            # Standard format (Dahoas/full-hh-rlhf, Anthropic/hh-rlhf)
            data.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            })
        elif "question" in item and "chosen" in item and "rejected" in item:
            # Alternative format with 'question' instead of 'prompt'
            data.append({
                "prompt": item["question"],
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            })
        elif "conversations" in item:
            # OpenAI format or other conversation formats
            # This would need custom parsing based on specific dataset structure
            pass
        else:
            print(f"Warning: Unsupported dataset format for item: {item.keys()}")
            continue
    
    return data


def load_hh_rlhf_data(split="train"):
    """Load HH-RLHF dataset (backward compatibility)."""
    return load_preference_data("Dahoas/full-hh-rlhf", split=split)