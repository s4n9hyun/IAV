import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import json
import random
from datasets import load_dataset
import logging


logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        truncation: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        chosen_text = f"{prompt} {chosen}"
        rejected_text = f"{prompt} {rejected}"
        
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        return {
            "input_ids_chosen": chosen_encoding["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_encoding["attention_mask"].squeeze(0),
            "labels_chosen": chosen_encoding["input_ids"].squeeze(0),
            "input_ids_rejected": rejected_encoding["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_encoding["attention_mask"].squeeze(0),
            "labels_rejected": rejected_encoding["input_ids"].squeeze(0)
        }


class HHRLHFDataset:
    @staticmethod
    def load_data(split: str = "train", sample_size: Optional[int] = None, val_ratio: Optional[float] = None) -> List[Dict]:
        """
        Load HH-RLHF dataset with optional train/validation split.
        
        Args:
            split: "train" or "test"
            sample_size: Max samples to load
            val_ratio: If set, splits train set (e.g., 0.1 = use last 10% as validation)
        """
        dataset = load_dataset("Dahoas/full-hh-rlhf", split=split)
        
        processed_data = []
        for item in dataset:
            # Dahoas format: prompt + chosen/rejected (needs concatenation)
            full_chosen = item["prompt"] + item["chosen"]
            full_rejected = item["prompt"] + item["rejected"]
            
            processed_item = {
                "prompt": item["prompt"],
                "chosen": full_chosen,
                "rejected": full_rejected
            }
            processed_data.append(processed_item)
        
        # Split train set if val_ratio specified
        if split == "train" and val_ratio is not None:
            n_total = len(processed_data)
            if val_ratio < 0.5:  # Return validation portion
                n_train = int(n_total * 0.9)  # Fixed: use 90% for training
                processed_data = processed_data[n_train:]  # Take remainder for validation
                logger.info(f"Validation: {len(processed_data)} samples (10% of train)")
            else:  # Return training portion (90%)
                n_train = int(n_total * val_ratio)
                processed_data = processed_data[:n_train]
                logger.info(f"Training: {len(processed_data)} samples ({val_ratio*100:.0f}% of train)")
        
        if sample_size and sample_size < len(processed_data):
            processed_data = random.sample(processed_data, sample_size)
        
        logger.info(f"Loaded {len(processed_data)} samples from HH-RLHF {split} set")
        return processed_data


class UltraFeedbackDataset:
    @staticmethod
    def load_data(
        data_path: Optional[str] = None,
        split: str = "train",
        sample_size: Optional[int] = None
    ) -> List[Dict]:
        
        if data_path:
            with open(data_path, 'r') as f:
                raw_data = json.load(f)
        else:
            dataset = load_dataset("openbmb/UltraFeedback", split=split)
            raw_data = list(dataset)
        
        processed_data = []
        for item in raw_data:
            if "completions" in item and len(item["completions"]) >= 2:
                completions_sorted = sorted(
                    item["completions"],
                    key=lambda x: x.get("overall_score", 0),
                    reverse=True
                )
                
                processed_item = {
                    "prompt": item["instruction"],
                    "chosen": completions_sorted[0]["response"],
                    "rejected": completions_sorted[-1]["response"]
                }
                processed_data.append(processed_item)
        
        if sample_size and sample_size < len(processed_data):
            processed_data = random.sample(processed_data, sample_size)
        
        logger.info(f"Loaded {len(processed_data)} samples from UltraFeedback {split} set")
        return processed_data


class MixedPreferenceDataset:
    @staticmethod
    def create_mixed_dataset(
        datasets_config: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        mix_ratio: Optional[List[float]] = None
    ) -> PreferenceDataset:
        
        all_data = []
        
        for config in datasets_config:
            dataset_name = config["name"]
            
            if dataset_name == "hh-rlhf":
                data = HHRLHFDataset.load_data(
                    split=config.get("split", "train"),
                    sample_size=config.get("sample_size")
                )
            elif dataset_name == "ultrafeedback":
                data = UltraFeedbackDataset.load_data(
                    data_path=config.get("data_path"),
                    split=config.get("split", "train"),
                    sample_size=config.get("sample_size")
                )
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            all_data.extend(data)
        
        random.shuffle(all_data)
        
        return PreferenceDataset(
            data=all_data,
            tokenizer=tokenizer,
            max_length=max_length
        )


def create_dataloaders(
    tokenizer: AutoTokenizer,
    train_config: Dict,
    val_config: Optional[Dict] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    max_length: int = 512
) -> Tuple[DataLoader, Optional[DataLoader]]:
    
    train_dataset = MixedPreferenceDataset.create_mixed_dataset(
        datasets_config=train_config["datasets"],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = None
    if val_config:
        val_dataset = MixedPreferenceDataset.create_mixed_dataset(
            datasets_config=val_config["datasets"],
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_dataloader, val_dataloader


def prepare_comparison_batch(
    prompts: List[str],
    responses_a: List[str],
    responses_b: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    
    texts_a = [f"{p} {r}" for p, r in zip(prompts, responses_a)]
    texts_b = [f"{p} {r}" for p, r in zip(prompts, responses_b)]
    
    encoding_a = tokenizer(
        texts_a,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    encoding_b = tokenizer(
        texts_b,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    return {
        "input_ids_a": encoding_a["input_ids"],
        "attention_mask_a": encoding_a["attention_mask"],
        "input_ids_b": encoding_b["input_ids"],
        "attention_mask_b": encoding_b["attention_mask"]
    }