import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoConfig


class IAVModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        vocab_size: int,
        hidden_size: int,
        device: str = "cuda",
        freeze_backbone: bool = True,
        torch_dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=None
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.torch_dtype = torch_dtype
        
        self.base_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch_dtype)
        self.alignment_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch_dtype)
        
        self._init_heads()
        
        # Move everything to the specified device
        self.to(device)
        self.backbone.to(device)
        
    def _init_heads(self):
        nn.init.xavier_uniform_(self.base_head.weight)
        nn.init.zeros_(self.alignment_head.weight)
        
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.hidden_states[-1]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        hidden_states = self.get_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        z_base = self.base_head(hidden_states)
        
        a_t = self.alignment_head(hidden_states)
        
        z_final = z_base + alpha * a_t
        
        outputs = {
            "logits": z_final,
            "base_logits": z_base,
            "alignment_vector": a_t,
            "hidden_states": hidden_states
        }
        
        if not return_components:
            return {"logits": z_final}
        
        return outputs
    
    def generate_with_alignment(
        self,
        input_ids: torch.Tensor,
        max_length: int = 128,
        temperature: float = 0.7,
        alpha: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        
        self.eval()
        generated = input_ids.clone()
        alignment_strengths = []
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(
                    generated,
                    attention_mask=attention_mask,
                    alpha=alpha,
                    return_components=True
                )
                
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                
                alignment_strengths.append(
                    outputs["alignment_vector"][:, -1, :].norm(dim=-1).mean().item()
                )
                
                if do_sample:
                    filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones_like(next_token)
                    ], dim=1)
                
                if next_token.item() == self.config.eos_token_id:
                    break
        
        stats = {
            "alignment_strengths": alignment_strengths,
            "avg_alignment_strength": sum(alignment_strengths) / len(alignment_strengths) if alignment_strengths else 0
        }
        
        return generated, stats


def top_p_filtering(logits, top_p=0.9, min_tokens_to_keep=1):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    return logits


class DualHeadWrapper:
    def __init__(self, model: IAVModel, alpha: float = 1.0):
        self.model = model
        self.alpha = alpha
    
    def set_alpha(self, alpha: float):
        self.alpha = alpha
    
    def get_model_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            alpha=self.alpha,
            return_components=True
        )
    
    def analyze_alignment_vector(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        top_k: int = 10
    ) -> Dict:
        
        outputs = self.get_model_outputs(input_ids)
        alignment_vector = outputs["alignment_vector"][0, -1, :]
        
        top_promoted = torch.topk(alignment_vector, k=top_k)
        top_demoted = torch.topk(-alignment_vector, k=top_k)
        
        promoted_tokens = [tokenizer.decode(idx.item()) for idx in top_promoted.indices]
        demoted_tokens = [tokenizer.decode(idx.item()) for idx in top_demoted.indices]
        
        return {
            "promoted_tokens": promoted_tokens,
            "promoted_values": top_promoted.values.tolist(),
            "demoted_tokens": demoted_tokens,
            "demoted_values": (-top_demoted.values).tolist(),
            "l2_norm": alignment_vector.norm().item()
        }