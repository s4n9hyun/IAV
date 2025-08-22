#!/usr/bin/env python3
"""MAV training script."""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from typing import Dict, Tuple
from tqdm import tqdm

from models import create_mav
from data import PreferenceDataset, load_preference_data


class AlignmentLoss(nn.Module):
    """Alignment loss function."""
    
    def __init__(self, beta=0.1, lambda_l2=0.001):
        super().__init__()
        self.beta = beta
        self.lambda_l2 = lambda_l2
    
    def compute_preference_loss(self, logits_chosen, logits_rejected, labels_chosen, labels_rejected, mask_chosen, mask_rejected):
        """DPO preference loss."""
        
        log_probs_chosen = self._get_log_probs(logits_chosen, labels_chosen)
        log_probs_rejected = self._get_log_probs(logits_rejected, labels_rejected)
        
        # Shift masks
        mask_chosen_shifted = mask_chosen[:, 1:].contiguous()
        mask_rejected_shifted = mask_rejected[:, 1:].contiguous()
        
        # Average log probs
        log_probs_chosen = (log_probs_chosen * mask_chosen_shifted).sum(dim=1) / mask_chosen_shifted.sum(dim=1)
        log_probs_rejected = (log_probs_rejected * mask_rejected_shifted).sum(dim=1) / mask_rejected_shifted.sum(dim=1)
        
        # DPO loss
        preference_loss = -F.logsigmoid(self.beta * (log_probs_chosen - log_probs_rejected)).mean()
        return preference_loss
    
    def compute_l2_regularization(self, alignment_vector, attention_mask):
        """L2 regularization with target norm to encourage non-zero alignment."""
        l2_norms = alignment_vector.norm(dim=-1)  # [batch_size, seq_len]
        # Encourage norm around 1.0 instead of 0
        target_norm = 1.0
        l2_penalty = ((l2_norms - target_norm) ** 2) * attention_mask
        return l2_penalty.sum() / (attention_mask.sum() + 1e-8)
    
    
    def _get_log_probs(self, logits, labels):
        """Get log probs for labels."""
        log_probs = F.log_softmax(logits, dim=-1)
        labels = labels[:, 1:].contiguous()
        log_probs = log_probs[:, :-1, :].contiguous()
        
        return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    
    def forward(self, final_logits_chosen, final_logits_rejected, labels_chosen, labels_rejected,
               mask_chosen, mask_rejected, alignment_vector_chosen):
        """Total loss."""
        
        # Pref loss
        L_pref = self.compute_preference_loss(
            final_logits_chosen, final_logits_rejected,
            labels_chosen, labels_rejected,
            mask_chosen, mask_rejected
        )
        
        # L2 reg
        L_l2 = self.compute_l2_regularization(alignment_vector_chosen, mask_chosen)
        
        total_loss = L_pref + self.lambda_l2 * L_l2
        
        return total_loss, {
            "preference_loss": L_pref.item(),
            "l2_loss": L_l2.item(),
            "total_loss": total_loss.item()
        }


class AlignmentTrainer:
    """Trains alignment model only."""
    
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader=None,
                 learning_rate=1e-5, num_epochs=2, warmup_steps=1000, gradient_accumulation_steps=2,
                 max_grad_norm=0.5, beta=0.1, lambda_l2=0.01, accelerator=None,
                 save_dir="./checkpoints", save_steps=1000, eval_steps=500):
        
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.accelerator = accelerator
        
        # Only alignment params - handle accelerator wrapping
        if hasattr(self.model, 'alignment_model'):
            alignment_params = list(self.model.alignment_model.parameters())
        else:
            # Model is wrapped by accelerator, unwrap it
            alignment_params = list(self.model.module.alignment_model.parameters())
        
        # Verify frozen - handle accelerator wrapping
        if hasattr(self.model, 'base_model'):
            base_trainable = sum(p.numel() for p in self.model.base_model.parameters() if p.requires_grad)
        else:
            base_trainable = sum(p.numel() for p in self.model.module.base_model.parameters() if p.requires_grad)
        
        if base_trainable > 0:
            raise RuntimeError(f"Base model has {base_trainable} trainable parameters! Should be 0.")
        
        self.optimizer = torch.optim.AdamW(alignment_params, lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        
        # Prep accelerator
        if self.accelerator:
            self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)
        
        self.loss_fn = AlignmentLoss(beta, lambda_l2)
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        
        os.makedirs(save_dir, exist_ok=True)
    
    def train(self, start_epoch=0, start_step=0):
        """Training loop."""
        
        # Set modes - handle accelerator wrapping
        if hasattr(self.model, 'base_model'):
            self.model.base_model.eval()  # Frozen
            self.model.alignment_model.train()  # Trainable
        else:
            self.model.module.base_model.eval()  # Frozen
            self.model.module.alignment_model.train()  # Trainable
        
        global_step = start_step
        best_val_loss = float('inf')
        
        # Calculate steps to skip when resuming
        steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        
        for epoch in range(start_epoch, self.num_epochs):
            epoch_losses = []
            
            # Calculate how many batches to skip
            if epoch == start_epoch and start_step > 0:
                # Calculate number of raw batches to skip (before gradient accumulation)
                batches_to_skip = start_step * self.gradient_accumulation_steps - epoch * len(self.train_dataloader)
                if batches_to_skip < 0:
                    batches_to_skip = 0
            else:
                batches_to_skip = 0
            
            # Adjust progress bar to show resume progress correctly
            initial_step = (start_step * self.gradient_accumulation_steps) if epoch == start_epoch else 0
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch + 1}/{self.num_epochs} (Alignment Only)",
                initial=initial_step if epoch == start_epoch and start_step > 0 else 0
            )
            
            for step, batch in enumerate(progress_bar):
                # Skip already processed batches when resuming
                if batches_to_skip > 0:
                    batches_to_skip -= 1
                    continue
                if not self.accelerator:
                    batch = {k: v.to("cuda") for k, v in batch.items()}
                
                # Batch concatenation optimization: combine chosen/rejected into single forward pass
                batch_size = batch["input_ids_chosen"].size(0)
                
                # Concatenate chosen and rejected examples
                combined_input_ids = torch.cat([
                    batch["input_ids_chosen"], 
                    batch["input_ids_rejected"]
                ], dim=0)
                combined_attention_mask = torch.cat([
                    batch["attention_mask_chosen"], 
                    batch["attention_mask_rejected"]
                ], dim=0)
                
                # Single forward pass for both chosen and rejected
                # Use standard alpha=1.0 for hierarchical attention (no amplification needed)
                combined_outputs = self.model(
                    combined_input_ids, combined_attention_mask,
                    alpha=1.0, return_components=True  # Standard alpha for attention-based alignment
                )
                
                # Split outputs back into chosen and rejected
                outputs_chosen = {
                    "logits": combined_outputs["logits"][:batch_size],
                    "alignment_vector": combined_outputs["alignment_vector"][:batch_size]
                }
                outputs_rejected = {
                    "logits": combined_outputs["logits"][batch_size:],
                    "alignment_vector": combined_outputs["alignment_vector"][batch_size:]
                }
                
                # Loss
                loss, loss_components = self.loss_fn(
                    outputs_chosen["logits"], outputs_rejected["logits"],
                    batch["labels_chosen"], batch["labels_rejected"],
                    batch["attention_mask_chosen"], batch["attention_mask_rejected"],
                    outputs_chosen["alignment_vector"]
                )
                
                loss = loss / self.gradient_accumulation_steps
                
                # Backward
                if self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                
                # Step
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.accelerator:
                        # Accelerator handles the unwrapping internally
                        alignment_params = self.model.alignment_model.parameters() if hasattr(self.model, 'alignment_model') else self.model.module.alignment_model.parameters()
                        self.accelerator.clip_grad_norm_(alignment_params, self.max_grad_norm)
                    else:
                        alignment_params = self.model.alignment_model.parameters() if hasattr(self.model, 'alignment_model') else self.model.module.alignment_model.parameters()
                        torch.nn.utils.clip_grad_norm_(alignment_params, self.max_grad_norm)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # Save and validate ONLY after gradient step
                    if not self.accelerator or self.accelerator.is_main_process:
                        if global_step > 0 and global_step % self.save_steps == 0:
                            self.save_checkpoint(epoch, global_step, best=False)
                        
                        # Validate
                        if self.val_dataloader and global_step > 0 and global_step % self.eval_steps == 0:
                            val_loss = self.validate()
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                self.save_checkpoint(epoch, global_step, best=True)
                
                epoch_losses.append(loss_components["total_loss"])
                progress_bar.set_postfix({
                    "loss": f"{loss_components['total_loss']:.4f}",
                    "pref": f"{loss_components['preference_loss']:.4f}",
                    "l2": f"{loss_components['l2_loss']:.4f}",
                    "step": global_step
                })
            
            # Save final checkpoint
            self.save_checkpoint(epoch, global_step, best=False)
    
    def validate(self):
        """Validation loop."""
        # Set modes - handle accelerator wrapping
        if hasattr(self.model, 'base_model'):
            self.model.base_model.eval()
            self.model.alignment_model.eval()
        else:
            self.model.module.base_model.eval()
            self.model.module.alignment_model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                if not self.accelerator:
                    batch = {k: v.to("cuda") for k, v in batch.items()}
                
                try:
                    # Batch concatenation optimization for validation as well
                    batch_size = batch["input_ids_chosen"].size(0)
                    
                    # Concatenate chosen and rejected examples
                    combined_input_ids = torch.cat([
                        batch["input_ids_chosen"], 
                        batch["input_ids_rejected"]
                    ], dim=0)
                    combined_attention_mask = torch.cat([
                        batch["attention_mask_chosen"], 
                        batch["attention_mask_rejected"]
                    ], dim=0)
                    
                    # Single forward pass for both chosen and rejected
                    combined_outputs = self.model(
                        combined_input_ids, combined_attention_mask,
                        alpha=1.0, return_components=True
                    )
                    
                    # Split outputs back into chosen and rejected
                    outputs_chosen = {
                        "logits": combined_outputs["logits"][:batch_size],
                        "alignment_vector": combined_outputs["alignment_vector"][:batch_size]
                    }
                    outputs_rejected = {
                        "logits": combined_outputs["logits"][batch_size:],
                        "alignment_vector": combined_outputs["alignment_vector"][batch_size:]
                    }
                    
                    loss, loss_components = self.loss_fn(
                        outputs_chosen["logits"], outputs_rejected["logits"],
                        batch["labels_chosen"], batch["labels_rejected"],
                        batch["attention_mask_chosen"], batch["attention_mask_rejected"],
                        outputs_chosen["alignment_vector"]
                    )
                    
                    val_losses.append(loss_components["total_loss"])
                except Exception:
                    continue
        
        # Back to training - handle accelerator wrapping
        if hasattr(self.model, 'alignment_model'):
            self.model.alignment_model.train()
        else:
            self.model.module.alignment_model.train()
        
        if not val_losses:
            return float('inf')
        
        avg_loss = sum(val_losses) / len(val_losses)
        return avg_loss
    
    def save_checkpoint(self, epoch, global_step, best=False):
        """Save alignment model checkpoint."""
        if self.accelerator and not self.accelerator.is_main_process:
            return
        
        # Only save alignment model state - handle accelerator wrapping
        if self.accelerator:
            # Get alignment model, handling nested wrapping
            if hasattr(self.model, 'alignment_model'):
                alignment_state_dict = self.accelerator.unwrap_model(self.model.alignment_model).state_dict()
            else:
                alignment_state_dict = self.accelerator.unwrap_model(self.model.module.alignment_model).state_dict()
        else:
            alignment_state_dict = self.model.alignment_model.state_dict() if hasattr(self.model, 'alignment_model') else self.model.module.alignment_model.state_dict()
        
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "alignment_model_state_dict": alignment_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_config": {
                "alignment_type": "hierarchical_attention",
                "num_alignment_refs": 32,
                "max_seq_len": 2048,
                "hidden_size": 4096,
                "vocab_size": 32000,
            }
        }
        
        if best:
            path = os.path.join(self.save_dir, "best_alignment.pt")
        else:
            path = os.path.join(self.save_dir, f"alignment_step_{global_step}.pt")
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if "alignment_model_state_dict" in checkpoint:
            if self.accelerator:
                # Handle accelerator wrapping
                if hasattr(self.model, 'alignment_model'):
                    unwrapped_model = self.accelerator.unwrap_model(self.model.alignment_model)
                else:
                    unwrapped_model = self.accelerator.unwrap_model(self.model.module.alignment_model)
                unwrapped_model.load_state_dict(checkpoint["alignment_model_state_dict"])
            else:
                alignment_model = self.model.alignment_model if hasattr(self.model, 'alignment_model') else self.model.module.alignment_model
                alignment_model.load_state_dict(checkpoint["alignment_model_state_dict"])
        
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0)
        }


def main():
    parser = argparse.ArgumentParser(description="Train MAV")
    
    # Model
    parser.add_argument("--model_name", default="argsearch/llama-7b-sft-float32", help="Base model (frozen)")
    parser.add_argument("--output_dir", default="./outputs/mav", help="Output directory")
    
    # Dataset
    parser.add_argument("--dataset_name", default="Dahoas/full-hh-rlhf", help="Dataset name for preference data")
    parser.add_argument("--dataset_split", default="train", help="Dataset split to use")
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8, help="Conservative batch size for stable training")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation for effective batch size 16")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for alignment model")
    
    # Schedule
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Longer warmup for stable training")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    
    # Hyperparameters
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--lambda_l2", type=float, default=0.001, help="L2 regularization for alignment vectors")
    
    # Optimization
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    
    # Checkpoint
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    # Init accelerator
    accelerator = Accelerator(mixed_precision="bf16" if args.bf16 else "no")
    
    torch.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if accelerator.is_main_process:
        print(f"🌍 Training MAV - Simplified Cross-Attention Alignment!")
        print(f"💪 Architecture: Cross-Attention Only (~200M parameters)")
        print(f"🧠 Attention: Cross-Attention + Position-aware (simplified)")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    model = create_mav(
        base_model_name=args.model_name,
        device="cpu",  # Let accelerator handle device placement
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    )
    
    if accelerator.is_main_process:
        base_info = model.base_model.get_model_info()
        alignment_info = model.alignment_model.get_architecture_info()
        system_info = model.get_system_info()
        
        print(f"\n📊 Model Information:")
        print(f"  Base Model: {base_info['model_name']}")
        print(f"    - Parameters: {base_info['parameters_B']:.1f}B (FROZEN)")
        print(f"    - Hidden size: {base_info['hidden_size']}")
        print(f"    - Vocab size: {base_info['vocab_size']}")
        
        print(f"  Alignment Model: Simplified Cross-Attention")
        print(f"    - Parameters: {alignment_info['parameters_M']:.1f}M (TRAINABLE)")
        print(f"    - Architecture: Cross-Attention + Position-Aware (no self-attention)")
        print(f"    - Alignment refs: {model.alignment_model.num_alignment_refs} reference vectors")
        
        print(f"  Compatibility:")
        print(f"    - Adaptive pooling: {'✓' if system_info['compatibility']['adaptive_pooling_needed'] else '✗'}")
        print(f"    - Pooling ratio: {system_info['compatibility']['pooling_ratio']:.2f}")
        print(f"    - Vocab compatible: {'✓' if system_info['compatibility']['vocab_compatible'] else '✗'}")
    
    # Data
    all_data = load_preference_data(dataset_name=args.dataset_name, split=args.dataset_split)
    n_total = len(all_data)
    n_train = int(n_total * 0.9)
    train_data = all_data[:n_train]
    val_data = all_data[n_train:]
    
    if accelerator.is_main_process:
        print(f"\n📚 Dataset: {args.dataset_name} ({args.dataset_split})")
        print(f"    - {n_total} total, {len(train_data)} train, {len(val_data)} val")
    
    train_dataset = PreferenceDataset(train_data, tokenizer, args.max_length)
    val_dataset = PreferenceDataset(val_data, tokenizer, args.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Prepare
    model, train_dataloader, val_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader)
    
    # Trainer
    trainer = AlignmentTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
        beta=args.beta,
        lambda_l2=args.lambda_l2,
        accelerator=accelerator,
        save_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps
    )
    
    # Resume
    start_epoch = 0
    start_step = 0
    
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        
        # Auto-find latest
        if checkpoint_path.lower() == "true":
            import glob
            checkpoints = glob.glob(os.path.join(args.output_dir, "alignment_step_*.pt"))
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
                checkpoint_path = checkpoints[-1]
                if accelerator.is_main_process:
                    print(f"Found latest checkpoint: {checkpoint_path}")
            else:
                checkpoint_path = None
        
        if checkpoint_path and checkpoint_path.lower() != "true":
            if accelerator.is_main_process:
                print(f"Resuming alignment model from: {checkpoint_path}")
            checkpoint_info = trainer.load_checkpoint(checkpoint_path)
            start_epoch = checkpoint_info["epoch"]
            start_step = checkpoint_info["global_step"]
    
    # Train
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("🚀 STARTING MAV ALIGNMENT TRAINING")
        print(f"📐 Architecture: {alignment_info['parameters_M']:.1f}M parameters")
        print("=" * 60)
    
    trainer.train(start_epoch=start_epoch, start_step=start_step)
    
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("🎉 MAV ALIGNMENT TRAINING COMPLETE!")
        print(f"✓ Alignment checkpoint saved in: {args.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()