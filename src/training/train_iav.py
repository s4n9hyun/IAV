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
        
        preference_loss = -F.logsigmoid(
            self.beta * (log_probs_chosen - log_probs_rejected)
        ).mean()
        
        return preference_loss
    
    def compute_kl_regularization(
        self,
        base_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        
        if self.reference_model is None:
            return torch.tensor(0.0, device=base_logits.device)
        
        with torch.no_grad():
            ref_outputs = self.reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, 'logits') else ref_outputs
        
        base_probs = F.softmax(base_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        
        kl_div = F.kl_div(
            base_probs.log(),
            ref_probs,
            reduction='none'
        ).sum(dim=-1)
        
        kl_div = (kl_div * attention_mask).sum() / attention_mask.sum()
        
        return kl_div
    
    def compute_l2_regularization(
        self,
        alignment_vector: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        
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
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        
        L_pref = self.compute_preference_loss(
            model_outputs_chosen["logits"],
            model_outputs_rejected["logits"],
            labels_chosen,
            labels_rejected,
            mask_chosen,
            mask_rejected
        )
        
        L_kl = self.compute_kl_regularization(
            model_outputs_chosen["base_logits"],
            input_ids,
            attention_mask
        )
        
        L_l2 = self.compute_l2_regularization(
            model_outputs_chosen["alignment_vector"],
            attention_mask
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
        device: str = "cuda",
        log_wandb: bool = False,
        save_dir: str = "./checkpoints"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
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
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
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
                    input_ids=batch["input_ids_chosen"],
                    attention_mask=batch["attention_mask_chosen"]
                )
                
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    
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
                
                if self.log_wandb and global_step % 10 == 0:
                    wandb.log({
                        "train/total_loss": loss_components["total_loss"],
                        "train/preference_loss": loss_components["preference_loss"],
                        "train/kl_loss": loss_components["kl_loss"],
                        "train/l2_loss": loss_components["l2_loss"],
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "global_step": global_step
                    })
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
            
            if self.val_dataloader is not None:
                val_loss = self.validate()
                logger.info(f"Validation Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, global_step, best=True)
            
            self.save_checkpoint(epoch, global_step, best=False)
    
    def validate(self) -> float:
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
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
                    input_ids=batch["input_ids_chosen"],
                    attention_mask=batch["attention_mask_chosen"]
                )
                
                val_losses.append(loss_components["total_loss"])
        
        self.model.train()
        return sum(val_losses) / len(val_losses)
    
    def save_checkpoint(self, epoch: int, global_step: int, best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }
        
        if best:
            path = os.path.join(self.save_dir, "best_model.pt")
        else:
            path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")