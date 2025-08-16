import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoConfig
import copy


class IAVModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        vocab_size: int,
        hidden_size: int,
        device: str = "cuda",
        freeze_backbone: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(base_model_name)
        
        # Use standard attention to avoid FlashAttention-2 compatibility issues
        print("Using standard attention for maximum compatibility")
        full_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=None,
            attn_implementation="eager"  # Force standard attention
        )
        
        # Separate backbone (transformer) and lm_head
        self.backbone = full_model.model  # Transformer layers only
        original_lm_head = full_model.lm_head  # Pre-trained lm_head with knowledge
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.torch_dtype = torch_dtype
        
        # base_head inherits pre-trained weights (knowledge)
        self.base_head = original_lm_head
        
        # alignment_head is a copy that will learn alignment adjustments
        self.alignment_head = copy.deepcopy(original_lm_head)
        nn.init.zeros_(self.alignment_head.weight)  # Start from zero for alignment
        
        # Make both heads trainable
        for param in self.base_head.parameters():
            param.requires_grad = True
        for param in self.alignment_head.parameters():
            param.requires_grad = True
        
        # Move everything to device
        self.to(device)
        
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Check input validity
        if input_ids.numel() == 0 or input_ids.shape[1] == 0:
            # Return empty tensors with proper shapes
            batch_size = input_ids.shape[0] if input_ids.numel() > 0 else 1
            empty_hidden = torch.zeros(batch_size, 0, self.hidden_size, 
                                     dtype=self.torch_dtype, device=input_ids.device)
            return empty_hidden, past_key_values
        
        try:
            # Additional validation for KV cache usage
            if use_cache and past_key_values is not None:
                # When using KV cache with past states, input should be single token
                if input_ids.shape[1] != 1:
                    print(f"WARNING: KV cache expects single token, got {input_ids.shape[1]} tokens. Resetting cache.")
                    past_key_values = None
                    use_cache = False
            
            # Now backbone is just the transformer layers (no lm_head)
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
            # Get the last hidden states and past_key_values if using cache
            hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            new_past_key_values = outputs.past_key_values if use_cache else None
            return hidden_states, new_past_key_values
        except Exception as e:
            print(f"WARNING: Backbone forward failed: {e}")
            # If KV cache is causing issues, try without it
            if use_cache and past_key_values is not None:
                print("WARNING: Retrying without KV cache")
                try:
                    outputs = self.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False
                    )
                    hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                    return hidden_states, None
                except Exception as e2:
                    print(f"WARNING: Backbone forward failed even without KV cache: {e2}")
            
            # Return empty tensors as fallback
            batch_size = input_ids.shape[0]
            empty_hidden = torch.zeros(batch_size, 0, self.hidden_size, 
                                     dtype=self.torch_dtype, device=input_ids.device)
            return empty_hidden, None
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        alpha: float = 1.0,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        hidden_states, new_past_key_values = self.get_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        # Check if hidden_states is empty
        if hidden_states.numel() == 0 or hidden_states.shape[1] == 0:
            # Return empty logits with proper vocab size
            batch_size = hidden_states.shape[0]
            empty_logits = torch.zeros(batch_size, 0, self.vocab_size, 
                                     dtype=self.torch_dtype, device=hidden_states.device)
            outputs = {
                "logits": empty_logits,
                "base_logits": empty_logits,
                "alignment_vector": empty_logits,
                "hidden_states": hidden_states,
                "past_key_values": new_past_key_values
            }
            return outputs if return_components else {"logits": empty_logits, "past_key_values": new_past_key_values}
        
        z_base = self.base_head(hidden_states)
        
        a_t = self.alignment_head(hidden_states)
        
        z_final = z_base + alpha * a_t
        
        outputs = {
            "logits": z_final,
            "base_logits": z_base,
            "alignment_vector": a_t,
            "hidden_states": hidden_states,
            "past_key_values": new_past_key_values
        }
        
        if not return_components:
            return {"logits": z_final, "past_key_values": new_past_key_values}
        
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
        
        # Check for empty input
        if input_ids.numel() == 0 or input_ids.shape[1] == 0:
            return input_ids, {
                "alignment_strengths": [],
                "avg_alignment_strength": 0.0
            }
        
        self.eval()
        generated = input_ids.clone()
        alignment_strengths = []
        
        # Initialize KV cache variables
        past_key_values = None
        use_kv_cache = True  # Start with KV cache enabled
        
        with torch.no_grad():
            for step in range(max_length):
                # For first step, use full input. For subsequent steps, use only the last token
                if step == 0:
                    current_input_ids = generated
                    current_attention_mask = attention_mask
                else:
                    if use_kv_cache:
                        # Use only the last token for KV cache efficiency
                        current_input_ids = generated[:, -1:] 
                        current_attention_mask = None  # Let model handle it automatically
                    else:
                        # Fallback: process full sequence without KV cache
                        current_input_ids = generated
                        current_attention_mask = attention_mask
                
                # Check if current input is valid
                if current_input_ids.numel() == 0 or current_input_ids.shape[1] == 0:
                    print("WARNING: Empty current_input_ids, stopping generation")
                    break
                
                try:
                    outputs = self.forward(
                        current_input_ids,
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values if use_kv_cache else None,
                        use_cache=use_kv_cache,
                        alpha=alpha,
                        return_components=True
                    )
                except Exception as e:
                    print(f"WARNING: Forward pass failed: {e}")
                    if use_kv_cache:
                        print("WARNING: Disabling KV cache and retrying")
                        use_kv_cache = False
                        past_key_values = None
                        # Retry without KV cache
                        try:
                            outputs = self.forward(
                                generated,  # Use full sequence
                                attention_mask=attention_mask,
                                past_key_values=None,
                                use_cache=False,
                                alpha=alpha,
                                return_components=True
                            )
                        except Exception as e2:
                            print(f"WARNING: Forward pass failed even without KV cache: {e2}")
                            break
                    else:
                        break
                
                # Update past_key_values for next iteration (only if using KV cache)
                if use_kv_cache:
                    past_key_values = outputs.get("past_key_values", None)
                
                # Check if outputs contain valid logits
                if "logits" not in outputs or outputs["logits"] is None:
                    print("WARNING: No logits in outputs, stopping generation")
                    break
                
                logits = outputs["logits"]
                if logits.numel() == 0 or logits.shape[0] == 0 or logits.shape[1] == 0:
                    print("WARNING: Empty logits tensor, stopping generation")
                    break
                
                next_token_logits = logits[:, -1, :] / temperature
                
                # Check if logits are empty
                if next_token_logits.numel() == 0 or next_token_logits.shape[-1] == 0:
                    print("WARNING: Empty logits encountered, stopping generation")
                    break
                
                # Calculate alignment strength safely
                alignment_vec = outputs["alignment_vector"][:, -1, :]
                if alignment_vec.numel() > 0:
                    alignment_strength = alignment_vec.norm(dim=-1).mean().item()
                else:
                    alignment_strength = 0.0
                alignment_strengths.append(alignment_strength)
                
                if do_sample:
                    filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
                    if filtered_logits.numel() == 0:
                        print("WARNING: Empty filtered logits, stopping generation")
                        break
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    if next_token_logits.numel() == 0:
                        print("WARNING: Empty logits for argmax, stopping generation")
                        break
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Update attention mask for the new token (if provided initially)
                if attention_mask is not None:
                    # Extend attention mask for each new token
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones_like(next_token, device=attention_mask.device)
                    ], dim=1)
                
                if next_token.item() == self.config.eos_token_id:
                    break
        
        stats = {
            "alignment_strengths": alignment_strengths,
            "avg_alignment_strength": sum(alignment_strengths) / len(alignment_strengths) if alignment_strengths else 0.0
        }
        
        return generated, stats


def top_p_filtering(logits, top_p=0.9, min_tokens_to_keep=1):
    # Check for empty logits
    if logits.numel() == 0 or logits.shape[-1] == 0:
        return logits
    
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
            return_components=True,
            use_cache=False  # For analysis, we don't need cache
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