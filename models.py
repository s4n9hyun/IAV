#!/usr/bin/env python3
"""MAV models: frozen base + trainable alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from transformers import AutoModelForCausalLM


class FrozenBaseModel(nn.Module):
    """Frozen SFT model with runtime switching capability."""
    
    def __init__(self, model_name="argsearch/llama-7b-sft-float32", device="cuda", torch_dtype=torch.bfloat16):
        super().__init__()
        
        self.device = device
        self.torch_dtype = torch_dtype
        self.current_model_name = None
        
        # Load initial model
        self.load_model(model_name)
    
    def load_model(self, model_name):
        """Load a new base model at runtime."""
        # Clear existing model
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()
        
        # Load new model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.torch_dtype, device_map=None, attn_implementation="eager"
        )
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Update config
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.current_model_name = model_name
        
        # Store vocab_size for alignment model compatibility
        # No restrictions - support any vocab_size
        
        self.to(self.device)
        self.eval()
        
        print(f"Loaded base model: {model_name} (hidden_size={self.hidden_size}, vocab_size={self.vocab_size})")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_info(self):
        """Get current model information."""
        return {
            "model_name": self.current_model_name,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "parameters": self.count_parameters(),
            "parameters_B": self.count_parameters() / 1e9
        }
    
    def get_base_outputs(self, input_ids, attention_mask=None, position_ids=None, 
                        past_key_values=None, use_cache=False, layer_indices=None):
        """Get logits and hidden states from multiple layers."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=use_cache, output_hidden_states=True
            )
            
            # extract multi-layer hidden states
            num_layers = len(outputs.hidden_states)
            multi_layer_hidden_states = {}
            
            if layer_indices:
                for layer_name, layer_idx in layer_indices.items():
                    # Convert negative indices to positive
                    if layer_idx < 0:
                        actual_idx = num_layers + layer_idx
                    else:
                        actual_idx = layer_idx
                    
                    # Clamp to valid range
                    actual_idx = max(0, min(actual_idx, num_layers - 1))
                    multi_layer_hidden_states[layer_name] = outputs.hidden_states[actual_idx]
            else:
                # Default: first, middle, last
                multi_layer_hidden_states = {
                    "first": outputs.hidden_states[1],  # Skip embedding
                    "middle": outputs.hidden_states[num_layers // 2],
                    "last": outputs.hidden_states[-1]
                }
            
            return {
                "base_logits": outputs.logits,
                "multi_layer_hidden_states": multi_layer_hidden_states,
                "past_key_values": outputs.past_key_values if use_cache else None
            }



class AlignmentModel(nn.Module):
    """Alignment model with adaptive pooling."""
    
    def __init__(self, target_size=4096, vocab_size=None, 
                 device="cuda", torch_dtype=torch.bfloat16, num_layers=3):
        super().__init__()
        
        self.target_size = target_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Multi-layer fusion layer
        fusion_input_size = target_size * num_layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, target_size, dtype=torch_dtype),
            nn.ReLU(),
            nn.Dropout(0.1)  # Add some regularization for the fusion layer
        )
        
        # Alignment architecture: 4096 -> 16384 -> 32000 (591.4M params)
        self.alignment = nn.Sequential(
            nn.Linear(target_size, 16384, dtype=torch_dtype),
            nn.ReLU(),
            nn.Linear(16384, vocab_size, dtype=torch_dtype)
        )
        
        # Zero init final layer
        with torch.no_grad():
            nn.init.zeros_(self.alignment[-1].weight)
            if self.alignment[-1].bias is not None:
                nn.init.zeros_(self.alignment[-1].bias)
        
        self.to(device)
    
    def adaptive_pool(self, hidden_states):
        """Convert any hidden_size to target_size via adaptive pooling."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if hidden_size == self.target_size:
            return hidden_states  # No pooling needed
        elif hidden_size < self.target_size:
            # Up-sample using linear layer (simple but effective)
            # This is more reliable than interpolation
            if not hasattr(self, '_upsample_layer') or self._upsample_layer.in_features != hidden_size:
                self._upsample_layer = nn.Linear(hidden_size, self.target_size, dtype=hidden_states.dtype).to(hidden_states.device)
                nn.init.xavier_uniform_(self._upsample_layer.weight)
            return self._upsample_layer(hidden_states)
        else:
            # Down-sample using adaptive average pooling
            # [B, L, H] -> [B, L, target_size]
            reshaped = hidden_states.view(batch_size * seq_len, 1, hidden_size)  # [B*L, 1, H]
            downsampled = F.adaptive_avg_pool1d(reshaped, self.target_size)  # [B*L, 1, target]
            return downsampled.squeeze(1).view(batch_size, seq_len, self.target_size)  # [B, L, target]
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_info(self):
        """Get architecture information."""
        param_count = self.count_parameters()
        return {
            "target_size": self.target_size,
            "vocab_size": self.vocab_size,
            "parameters": param_count,
            "parameters_M": param_count / 1e6,
        }
    
    def forward(self, multi_layer_hidden_states):
        """Generate alignment vector from multi-layer hidden states."""
        
        # Step 1: Apply adaptive pooling to each layer individually
        pooled_layers = []
        for layer_name in sorted(multi_layer_hidden_states.keys()):
            layer_states = multi_layer_hidden_states[layer_name]
            pooled_layer = self.adaptive_pool(layer_states)  # [B, L, target_size]
            pooled_layers.append(pooled_layer)
        
        # Step 2: Concatenate pooled layers
        concatenated_states = torch.cat(pooled_layers, dim=-1)  # [B, L, target_size * num_layers]
        
        # Step 3: Fuse multi-layer information
        fused_states = self.fusion_layer(concatenated_states)  # [B, L, target_size]
        
        # Step 4: Generate alignment vector from fused representation
        alignment_vector = self.alignment(fused_states)  # [B, L, vocab_size]
        
        return alignment_vector


class MAV(nn.Module):
    """MAV: Multi-layer alignment for any compatible base model."""
    
    def __init__(self, base_model, alignment_model, device="cuda", layer_indices=None):
        super().__init__()
        
        self.base_model = base_model
        self.alignment_model = alignment_model
        self.device = device
        self.layer_indices = layer_indices  # For multi-layer extraction
        
        # No dimension checks needed - adaptive pooling handles it
        # Only verify vocab_size compatibility
        assert base_model.vocab_size == alignment_model.vocab_size, \
            f"vocab_size mismatch: base={base_model.vocab_size}, alignment={alignment_model.vocab_size}"
    
    def switch_base_model(self, model_name):
        """Switch to a different base model at runtime."""
        print(f"Switching base model to: {model_name}")
        self.base_model.load_model(model_name)
        
        # Verify compatibility
        if self.base_model.vocab_size != self.alignment_model.vocab_size:
            raise ValueError(f"New base model vocab_size={self.base_model.vocab_size} incompatible with alignment vocab_size={self.alignment_model.vocab_size}. Train separate PAV for different vocab_size models.")
    
    def get_system_info(self):
        """Get complete system information."""
        base_info = self.base_model.get_model_info()
        alignment_info = self.alignment_model.get_architecture_info()
        
        return {
            "base_model": base_info,
            "alignment_model": alignment_info,
            "compatibility": {
                "adaptive_pooling_needed": base_info["hidden_size"] != alignment_info["target_size"],
                "pooling_ratio": base_info["hidden_size"] / alignment_info["target_size"],
                "vocab_compatible": base_info["vocab_size"] == alignment_info["vocab_size"]
            }
        }
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None,
               use_cache=False, alpha=1.0, return_components=False):
        """Forward with multi-layer alignment."""
        
        # Base outputs
        base_outputs = self.base_model.get_base_outputs(
            input_ids, attention_mask, position_ids, past_key_values, use_cache, 
            layer_indices=self.layer_indices
        )
        
        # Multi-layer alignment
        alignment_vector = self.alignment_model(base_outputs["multi_layer_hidden_states"])
        
        # Combine
        final_logits = base_outputs["base_logits"] + alpha * alignment_vector
        
        outputs = {
            "logits": final_logits,
            "past_key_values": base_outputs["past_key_values"]
        }
        
        if return_components:
            outputs.update({
                "base_logits": base_outputs["base_logits"],
                "alignment_vector": alignment_vector,
                "multi_layer_hidden_states": base_outputs["multi_layer_hidden_states"],
                "alpha": alpha
            })
        
        return outputs
    
    def generate_with_alignment(self, input_ids, max_length=128, temperature=0.7, alpha=1.0,
                              do_sample=True, top_p=0.9, attention_mask=None):
        """Generate with alpha control."""
        
        if input_ids.numel() == 0:
            return input_ids, {"alignment_strengths": [], "avg_alignment_strength": 0.0}
        
        self.eval()
        generated = input_ids.clone()
        alignment_strengths = []
        past_key_values = None
        
        with torch.no_grad():
            for step in range(max_length):
                # Prep inputs
                if step == 0:
                    current_input_ids = generated
                    current_attention_mask = attention_mask
                else:
                    current_input_ids = generated[:, -1:]
                    current_attention_mask = None
                
                # Forward
                outputs = self.forward(
                    current_input_ids, current_attention_mask, 
                    past_key_values=past_key_values, use_cache=True,
                    alpha=alpha, return_components=True
                )
                
                past_key_values = outputs.get("past_key_values")
                
                # Next token
                logits = outputs["logits"]
                if logits.numel() == 0 or logits.shape[1] == 0:
                    break
                
                next_token_logits = logits[:, -1, :] / temperature
                
                # Track strength
                alignment_vec = outputs["alignment_vector"][:, -1, :]
                if alignment_vec.numel() > 0:
                    alignment_strength = alignment_vec.norm(dim=-1).mean().item()
                else:
                    alignment_strength = 0.0
                alignment_strengths.append(alignment_strength)
                
                # Sample
                if do_sample:
                    filtered_logits = self._top_p_filtering(next_token_logits, top_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Update mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones_like(next_token, device=attention_mask.device)
                    ], dim=1)
                
                # Check EOS
                if next_token.item() == self.base_model.config.eos_token_id:
                    break
        
        stats = {
            "alignment_strengths": alignment_strengths,
            "avg_alignment_strength": sum(alignment_strengths) / len(alignment_strengths) if alignment_strengths else 0.0,
            "alpha_used": alpha
        }
        
        return generated, stats
    
    def _top_p_filtering(self, logits, top_p=0.9, min_tokens_to_keep=1):
        """Apply top-p filtering."""
        if logits.numel() == 0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits


def create_mav(base_model_name="argsearch/llama-7b-sft-float32", 
               device="cuda", torch_dtype=torch.bfloat16, layer_selection="auto"):
    """Create MAV model with multi-layer alignment.
    
    Args:
        base_model_name: Base model to use
        device: Device to use
        torch_dtype: Data type
        layer_selection: Layer selection strategy ("auto", "uniform", or custom dict)
    """
    
    base_model = FrozenBaseModel(base_model_name, device, torch_dtype)
    
    # Determine layer indices
    num_layers = base_model.config.num_hidden_layers
    
    if layer_selection == "auto":
        # Auto: first, middle, last layers
        layer_indices = {
            "first": 1,  # Skip embedding layer (index 0)
            "middle": num_layers // 2,
            "last": -1  # Last layer
        }
    elif layer_selection == "uniform":
        # Uniform: evenly spaced layers
        layer_indices = {
            "early": num_layers // 4,
            "middle": num_layers // 2,
            "late": 3 * num_layers // 4
        }
    elif isinstance(layer_selection, dict):
        # Custom layer indices
        layer_indices = layer_selection
    else:
        raise ValueError(f"Invalid layer_selection: {layer_selection}")
    
    # Alignment architecture - use base model's vocab_size
    alignment_model = AlignmentModel(
        target_size=4096, vocab_size=base_model.vocab_size,
        device=device, torch_dtype=torch_dtype, 
        num_layers=len(layer_indices)
    )
    
    return MAV(base_model, alignment_model, device, layer_indices=layer_indices)