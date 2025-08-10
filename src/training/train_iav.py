import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, Optional, Tuple
import logging
from tqdm import tqdm
import wandb
import os


logger = logging.getLogger(__name__)


class IAVLoss(nn.Module):
    def __init__(
        self,
        beta: float = 0.1,
        lambda_kl: float = 0.1,
        lambda_l2: float = 0.01,
        reference_model: Optional[nn.Module] = None
    ):
        super().__init__()
        self.beta = beta
        self.lambda_kl = lambda_kl
        self.lambda_l2 = lambda_l2
        self.reference_model = reference_model
        
        if reference_model:
            for param in self.reference_model.parameters():
                param.requires_grad = False
    
    def compute_preference_loss(
        self,
        logits_chosen: torch.Tensor,
        logits_rejected: torch.Tensor,
        labels_chosen: torch.Tensor,
        labels_rejected: torch.Tensor,
        mask_chosen: torch.Tensor,
        mask_rejected: torch.Tensor
    ) -> torch.Tensor:
        
        log_probs_chosen = self._get_log_probs(logits_chosen, labels_chosen)
        log_probs_rejected = self._get_log_probs(logits_rejected, labels_rejected)
        
        # Adjust masks to match the shifted sequences (remove first token)
        mask_chosen_shifted = mask_chosen[:, 1:].contiguous()
        mask_rejected_shifted = mask_rejected[:, 1:].contiguous()
        
        log_probs_chosen = (log_probs_chosen * mask_chosen_shifted).sum(dim=1) / mask_chosen_shifted.sum(dim=1)
        log_probs_rejected = (log_probs_rejected * mask_rejected_shifted).sum(dim=1) / mask_rejected_shifted.sum(dim=1)
        
        # Convert to float32 for numerical stability in loss computation
        log_probs_chosen = log_probs_chosen.float()
        log_probs_rejected = log_probs_rejected.float()
        
        preference_loss = -F.logsigmoid(
            self.beta * (log_probs_chosen - log_probs_rejected)
        ).mean()
        
        return preference_loss
    
    def compute_kl_regularization(
        self,
        base_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cached_ref_logits: torch.Tensor = None
    ) -> torch.Tensor:
        
        if self.reference_model is None and cached_ref_logits is None:
            return torch.tensor(0.0, device=base_logits.device)
        
        if cached_ref_logits is not None:
            # Use cached reference token logits
            # base_logits shape: [batch, seq_len, vocab]
            # cached_ref_logits shape: [batch, seq_len] (logits for actual tokens)
            batch_size, seq_len, vocab_size = base_logits.shape
            
            # Gather base model logits for actual tokens
            input_ids_expanded = input_ids.unsqueeze(-1)  # [batch, seq_len, 1]
            base_token_logits = torch.gather(base_logits, dim=2, index=input_ids_expanded).squeeze(-1)  # [batch, seq_len]
            ref_token_logits = cached_ref_logits.to(base_logits.device)  # [batch, seq_len]
            
            # Convert to float32 for numerical stability
            base_token_logits = base_token_logits.float()
            ref_token_logits = ref_token_logits.float()
            attention_mask = attention_mask.float()
            
            # Compute MSE loss between token logits (simpler than full KL divergence)
            kl_div = F.mse_loss(base_token_logits, ref_token_logits, reduction='none')  # [batch, seq_len]
            kl_div = (kl_div * attention_mask).sum() / attention_mask.sum()
            
            # Debug: Print actual values for first few steps
            if hasattr(self, '_debug_step_count'):
                self._debug_step_count += 1
            else:
                self._debug_step_count = 1
            
            if self._debug_step_count <= 3:
                print(f"DEBUG Step {self._debug_step_count}: base_token_logits mean={base_token_logits.mean().item():.6f}, "
                      f"ref_token_logits mean={ref_token_logits.mean().item():.6f}, "
                      f"kl_div={kl_div.item():.6f}")
        else:
            # Use live reference model
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                ref_logits = ref_outputs.logits if hasattr(ref_outputs, 'logits') else ref_outputs

            # Unify KL calculation method - use MSE between token logits (same as cached version)
            batch_size, seq_len, vocab_size = base_logits.shape
            
            # Gather base model logits for actual tokens
            input_ids_expanded = input_ids.unsqueeze(-1)  # [batch, seq_len, 1]
            base_token_logits = torch.gather(base_logits, dim=2, index=input_ids_expanded).squeeze(-1)  # [batch, seq_len]
            
            # Gather reference model logits for actual tokens
            ref_token_logits = torch.gather(ref_logits, dim=2, index=input_ids_expanded).squeeze(-1)  # [batch, seq_len]

            # Convert to float32 for numerical stability
            base_token_logits = base_token_logits.float()
            ref_token_logits = ref_token_logits.float()
            attention_mask = attention_mask.float()
            
            # Compute MSE loss between token logits (unified with cached version)
            kl_div = F.mse_loss(base_token_logits, ref_token_logits, reduction='none')  # [batch, seq_len]
            kl_div = (kl_div * attention_mask).sum() / attention_mask.sum()
        
        return kl_div
    
    def compute_l2_regularization(
        self,
        alignment_vector: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        
        # Convert to float32 for numerical stability
        alignment_vector = alignment_vector.float()
        if attention_mask is not None:
            attention_mask = attention_mask.float()
        
        l2_norms = alignment_vector.norm(dim=-1)
        
        if attention_mask is not None:
            l2_norms = l2_norms * attention_mask
            l2_reg = l2_norms.sum() / attention_mask.sum()
        else:
            l2_reg = l2_norms.mean()
        
        return l2_reg
    
    def _get_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        labels = labels[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        log_probs = log_probs[:, :-1, :].contiguous()
        
        log_probs = torch.gather(
            log_probs,
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)
        
        return log_probs
    
    def forward(
        self,
        model_outputs_chosen: Dict,
        model_outputs_rejected: Dict,
        labels_chosen: torch.Tensor,
        labels_rejected: torch.Tensor,
        mask_chosen: torch.Tensor,
        mask_rejected: torch.Tensor,
        input_ids_chosen: torch.Tensor,
        attention_mask_chosen: torch.Tensor,
        input_ids_rejected: torch.Tensor = None,
        attention_mask_rejected: torch.Tensor = None,
        cached_ref_logits_chosen: torch.Tensor = None,
        cached_ref_logits_rejected: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict]:
        
        L_pref = self.compute_preference_loss(
            model_outputs_chosen["logits"],
            model_outputs_rejected["logits"],
            labels_chosen,
            labels_rejected,
            mask_chosen,
            mask_rejected
        )
        
        # KL regularization computation for both chosen and rejected
        L_kl = torch.tensor(0.0, device=model_outputs_chosen["base_logits"].device)
        
        if self.lambda_kl > 0:
            # 1. Use cache if available (training with cached reference logits)
            if cached_ref_logits_chosen is not None:
                # KL regularization for chosen responses
                L_kl_chosen = self.compute_kl_regularization(
                    model_outputs_chosen["base_logits"],
                    input_ids_chosen,
                    attention_mask_chosen,
                    cached_ref_logits=cached_ref_logits_chosen
                )
                
                # KL regularization for rejected responses
                if cached_ref_logits_rejected is not None:
                    L_kl_rejected = self.compute_kl_regularization(
                        model_outputs_rejected["base_logits"],
                        input_ids_rejected,
                        attention_mask_rejected,
                        cached_ref_logits=cached_ref_logits_rejected
                    )
                    L_kl = (L_kl_chosen + L_kl_rejected) / 2.0  # Average KL loss
                else:
                    # Only chosen has cache (backward compatibility)
                    L_kl = L_kl_chosen
                
            # 2. Use reference model if available (validation without cache)
            elif self.reference_model is not None:
                # KL regularization for chosen responses
                L_kl_chosen = self.compute_kl_regularization(
                    model_outputs_chosen["base_logits"],
                    input_ids_chosen,
                    attention_mask_chosen,
                    cached_ref_logits=None
                )
                # KL regularization for rejected responses
                if input_ids_rejected is not None and attention_mask_rejected is not None:
                    L_kl_rejected = self.compute_kl_regularization(
                        model_outputs_rejected["base_logits"],
                        input_ids_rejected,
                        attention_mask_rejected,
                        cached_ref_logits=None
                    )
                else:
                    # Fallback to chosen if rejected not provided
                    L_kl_rejected = self.compute_kl_regularization(
                        model_outputs_rejected["base_logits"],
                        input_ids_chosen,
                        attention_mask_chosen,
                        cached_ref_logits=None
                    )
                L_kl = (L_kl_chosen + L_kl_rejected) / 2.0  # Average KL loss
        
        L_l2 = self.compute_l2_regularization(
            model_outputs_chosen["alignment_vector"],
            mask_chosen  # Fixed: use mask_chosen instead of undefined attention_mask
        )
        
        total_loss = L_pref + self.lambda_kl * L_kl + self.lambda_l2 * L_l2
        
        loss_components = {
            "preference_loss": L_pref.item(),
            "kl_loss": L_kl.item(),
            "l2_loss": L_l2.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_components


class IAVTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        beta: float = 0.1,
        lambda_kl: float = 0.1,
        lambda_l2: float = 0.01,
        reference_model: Optional[nn.Module] = None,
        accelerator = None,
        log_wandb: bool = False,
        save_dir: str = "./checkpoints",
        save_steps: int = 1000,
        eval_steps: int = 500
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.accelerator = accelerator
        
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Prepare optimizer and scheduler with accelerator
        if self.accelerator:
            self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)
        
        
        self.loss_fn = IAVLoss(
            beta=beta,
            lambda_kl=lambda_kl,
            lambda_l2=lambda_l2,
            reference_model=reference_model
        )
        
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_wandb = log_wandb
        self.save_dir = save_dir
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        
        os.makedirs(save_dir, exist_ok=True)
        
        if log_wandb:
            wandb.init(project="iav-training")
    
    def train(self):
        self.model.train()
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            epoch_losses = []
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}"
            )
            
            for step, batch in enumerate(progress_bar):
                # Skip device transfer if using accelerator (already handled)
                if not self.accelerator:
                    batch = {k: v.to("cuda") for k, v in batch.items()}
                
                # Debug: Check if cached data is present
                if step == 0:
                    print(f"DEBUG: Batch keys: {batch.keys()}")
                    if 'reference_token_logits_chosen' in batch:
                        print(f"DEBUG: reference_token_logits_chosen shape: {batch['reference_token_logits_chosen'].shape}")
                        print(f"DEBUG: reference_token_logits_rejected shape: {batch['reference_token_logits_rejected'].shape}")
                    else:
                        print("DEBUG: No reference_token_logits in batch!")
                
                model_outputs_chosen = self.model(
                    input_ids=batch["input_ids_chosen"],
                    attention_mask=batch["attention_mask_chosen"],
                    alpha=1.0,
                    return_components=True
                )
                
                model_outputs_rejected = self.model(
                    input_ids=batch["input_ids_rejected"],
                    attention_mask=batch["attention_mask_rejected"],
                    alpha=1.0,
                    return_components=True
                )
                
                loss, loss_components = self.loss_fn(
                    model_outputs_chosen=model_outputs_chosen,
                    model_outputs_rejected=model_outputs_rejected,
                    labels_chosen=batch["labels_chosen"],
                    labels_rejected=batch["labels_rejected"],
                    mask_chosen=batch["attention_mask_chosen"],
                    mask_rejected=batch["attention_mask_rejected"],
                    input_ids_chosen=batch["input_ids_chosen"],
                    attention_mask_chosen=batch["attention_mask_chosen"],
                    input_ids_rejected=batch["input_ids_rejected"],
                    attention_mask_rejected=batch["attention_mask_rejected"],
                    cached_ref_logits_chosen=batch.get("reference_token_logits_chosen"),
                    cached_ref_logits_rejected=batch.get("reference_token_logits_rejected")
                )
                
                loss = loss / self.gradient_accumulation_steps
                
                if self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.accelerator:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                epoch_losses.append(loss_components["total_loss"])
                
                progress_bar.set_postfix({
                    "loss": f"{loss_components['total_loss']:.4f}",
                    "pref": f"{loss_components['preference_loss']:.4f}",
                    "kl": f"{loss_components['kl_loss']:.4f}",
                    "l2": f"{loss_components['l2_loss']:.4f}"
                })
                
                # Only log and save on main process
                if not self.accelerator or self.accelerator.is_main_process:
                    if self.log_wandb and global_step % 10 == 0:
                        wandb.log({
                            "train/total_loss": loss_components["total_loss"],
                            "train/preference_loss": loss_components["preference_loss"],
                            "train/kl_loss": loss_components["kl_loss"],
                            "train/l2_loss": loss_components["l2_loss"],
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "global_step": global_step
                        })
                    
                    # Step-based checkpointing (skip step 0)
                    if global_step > 0 and global_step % self.save_steps == 0:
                        self.save_checkpoint(epoch, global_step, best=False)
                    
                    # Step-based evaluation (skip step 0)
                    if self.val_dataloader is not None and global_step > 0 and global_step % self.eval_steps == 0:
                        val_loss = self.validate()
                        logger.info(f"Step {global_step} - Validation Loss: {val_loss:.4f}")
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_checkpoint(epoch, global_step, best=True)
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Save final checkpoint for each epoch
            self.save_checkpoint(epoch, global_step, best=False)
    
    def validate(self) -> float:
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Skip device transfer if using accelerator (already handled)
                if not self.accelerator:
                    batch = {k: v.to("cuda") for k, v in batch.items()}
                
                model_outputs_chosen = self.model(
                    input_ids=batch["input_ids_chosen"],
                    attention_mask=batch["attention_mask_chosen"],
                    alpha=1.0,
                    return_components=True
                )
                
                model_outputs_rejected = self.model(
                    input_ids=batch["input_ids_rejected"],
                    attention_mask=batch["attention_mask_rejected"],
                    alpha=1.0,
                    return_components=True
                )
                
                loss, loss_components = self.loss_fn(
                    model_outputs_chosen=model_outputs_chosen,
                    model_outputs_rejected=model_outputs_rejected,
                    labels_chosen=batch["labels_chosen"],
                    labels_rejected=batch["labels_rejected"],
                    mask_chosen=batch["attention_mask_chosen"],
                    mask_rejected=batch["attention_mask_rejected"],
                    input_ids_chosen=batch["input_ids_chosen"],
                    attention_mask_chosen=batch["attention_mask_chosen"],
                    input_ids_rejected=batch["input_ids_rejected"],
                    attention_mask_rejected=batch["attention_mask_rejected"],
                    cached_ref_logits_chosen=batch.get("reference_token_logits_chosen"),
                    cached_ref_logits_rejected=batch.get("reference_token_logits_rejected")
                )
                
                val_losses.append(loss_components["total_loss"])
        
        self.model.train()
        return sum(val_losses) / len(val_losses)
    
    def save_checkpoint(self, epoch: int, global_step: int, best: bool = False):
        # Only save on main process in multi-GPU training
        if self.accelerator and not self.accelerator.is_main_process:
            return
            
        # Unwrap model to access original attributes (handles DDP wrapper)
        if self.accelerator:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model
            
        # Only save trainable heads, not the frozen backbone
        heads_state_dict = {
            "base_head": unwrapped_model.base_head.state_dict(),
            "alignment_head": unwrapped_model.alignment_head.state_dict()
        }
            
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "heads_state_dict": heads_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }
        
        if best:
            path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, path)
            logger.info(f"Best checkpoint saved to {path}")
        else:
            path = os.path.join(self.save_dir, f"checkpoint_step_{global_step}.pt")
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to {path}")
            
            # Keep only the 3 most recent checkpoints (excluding best_model.pt)
            self._cleanup_old_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training from saved state"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Get unwrapped model for loading state dict
        if self.accelerator:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model
        
        # Load model state (only trainable heads)
        if "heads_state_dict" in checkpoint:
            unwrapped_model.base_head.load_state_dict(checkpoint["heads_state_dict"]["base_head"])
            unwrapped_model.alignment_head.load_state_dict(checkpoint["heads_state_dict"]["alignment_head"])
            logger.info("Loaded model heads state dict")
        else:
            logger.warning("No heads_state_dict found in checkpoint")
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Loaded optimizer state dict")
        
        # Load scheduler state
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Loaded scheduler state dict")
        
        # Return checkpoint info for resuming training
        resume_info = {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0)
        }
        
        logger.info(f"Resuming from epoch {resume_info['epoch']}, step {resume_info['global_step']}")
        return resume_info
    
    def _cleanup_old_checkpoints(self, keep_count: int = 3):
        """Remove old checkpoints, keeping only the most recent ones"""
        import glob
        
        # Only cleanup on main process
        if self.accelerator and not self.accelerator.is_main_process:
            return
            
        # Find all checkpoint files (excluding best_model.pt)
        pattern = os.path.join(self.save_dir, "checkpoint_step_*.pt")
        checkpoints = glob.glob(pattern)
        
        if len(checkpoints) <= keep_count:
            return
        
        # Filter out non-existent files (in case another process already deleted them)
        checkpoints = [cp for cp in checkpoints if os.path.exists(cp)]
        
        if len(checkpoints) <= keep_count:
            return
            
        # Sort by modification time (newest first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        # Remove old checkpoints
        for old_checkpoint in checkpoints[keep_count:]:
            try:
                if os.path.exists(old_checkpoint):  # Double-check before deletion
                    os.remove(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except OSError as e:
                # Not an error if file doesn't exist (another process may have deleted it)
                if e.errno != 2:  # errno 2 = FileNotFoundError
                    logger.warning(f"Failed to remove {old_checkpoint}: {e}")