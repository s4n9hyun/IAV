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
        self.current_model_name = model_name
        
        # Load model
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
    """Simplified cross-attention alignment model using last hidden state."""
    
    def __init__(self, hidden_size=4096, vocab_size=None, 
                 device="cuda", torch_dtype=torch.bfloat16,
                 num_alignment_refs=8, max_seq_len=2048):  # Reduced from 32 to 8
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_alignment_refs = num_alignment_refs
        self.max_seq_len = max_seq_len
        
        # Simplified: Only cross-attention for alignment direction learning
        self.alignment_refs = nn.Parameter(
            torch.randn(num_alignment_refs, hidden_size, dtype=torch_dtype)
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True,
            dtype=torch_dtype
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size, dtype=torch_dtype)
        
        # Position-aware output projection
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_len, hidden_size, dtype=torch_dtype)
        )
        self.output_proj = nn.Linear(hidden_size, vocab_size, dtype=torch_dtype)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize parameters
        self._initialize_parameters()
        
        self.to(device)
    
    def _initialize_parameters(self):
        """Initialize model parameters with improved scaling."""
        with torch.no_grad():
            # Initialize alignment reference vectors with orthogonal initialization for diversity
            # Convert to float32 for orthogonal init, then convert back
            temp_refs = torch.randn_like(self.alignment_refs, dtype=torch.float32)
            nn.init.orthogonal_(temp_refs, gain=0.5)
            self.alignment_refs.data = temp_refs.to(self.alignment_refs.dtype)
            
            # Initialize position encoding with smaller values
            nn.init.normal_(self.pos_encoding, mean=0.0, std=0.01)
            
            # Initialize output projection with very small gain for stable training
            nn.init.xavier_uniform_(self.output_proj.weight, gain=0.01)  # Reduced from 0.02
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)
    
    def adaptive_pool(self, hidden_states):
        """Convert any hidden_size to alignment input size via adaptive pooling."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if hidden_size == self.hidden_size:
            return hidden_states  # No pooling needed
        elif hidden_size < self.hidden_size:
            # Up-sample using linear layer
            if not hasattr(self, '_upsample_layer') or self._upsample_layer.in_features != hidden_size:
                self._upsample_layer = nn.Linear(hidden_size, self.hidden_size, dtype=hidden_states.dtype).to(hidden_states.device)
                nn.init.xavier_uniform_(self._upsample_layer.weight)
            return self._upsample_layer(hidden_states)
        else:
            # Down-sample using adaptive average pooling
            reshaped = hidden_states.view(batch_size * seq_len, 1, hidden_size)
            downsampled = F.adaptive_avg_pool1d(reshaped, self.hidden_size)
            return downsampled.squeeze(1).view(batch_size, seq_len, self.hidden_size)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_info(self):
        """Get architecture information."""
        param_count = self.count_parameters()
        return {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "parameters": param_count,
            "parameters_M": param_count / 1e6,
        }
    
    def forward(self, last_hidden_state):
        """Simplified cross-attention alignment vector generation."""
        
        # Step 0: Apply adaptive pooling if needed
        pooled_state = self.adaptive_pool(last_hidden_state)  # [B, L, hidden_size]
        batch_size, seq_len, hidden_size = pooled_state.shape
        
        # Simplified: Direct cross-attention without self-attention
        # Expand alignment reference vectors for batch processing
        alignment_refs_expanded = self.alignment_refs.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, num_alignment_refs, hidden_size]
        
        aligned_features, cross_attn_weights = self.cross_attention(
            query=pooled_state,             # [B, L, hidden_size] - "What alignment is needed?"
            key=alignment_refs_expanded,    # [B, 8, hidden_size] - "Available alignment directions"  
            value=alignment_refs_expanded   # [B, 8, hidden_size] - "Alignment direction info"
        )
        
        # Residual connection + normalization
        aligned_features = self.cross_attn_norm(aligned_features + pooled_state)
        aligned_features = self.dropout(aligned_features)
        
        # Position-aware output projection
        # Add position encoding (handle variable sequence lengths)
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0)  # [1, L, hidden_size]
        position_aware = aligned_features + pos_enc
        
        # Final projection to vocabulary space
        alignment_vector = self.output_proj(position_aware)  # [B, L, vocab_size]
        
        # Return alignment vector and attention weights for analysis
        attention_info = {
            'cross_attention_weights': cross_attn_weights,    # [B, L, 8] - reduced from 32 to 8
            'alignment_strength': alignment_vector.norm(dim=-1),  # [B, L]
            'active_refs': cross_attn_weights.mean(dim=1).argmax(dim=-1)  # [B] - most used ref per batch
        }
        
        return alignment_vector, attention_info


class MAV(nn.Module):
    """MAV: Single-layer alignment using last hidden state."""
    
    def __init__(self, base_model, alignment_model, device="cuda"):
        super().__init__()
        
        self.base_model = base_model
        self.alignment_model = alignment_model
        self.device = device
        
        # Verify compatibility
        assert base_model.vocab_size == alignment_model.vocab_size, \
            f"vocab_size mismatch: base={base_model.vocab_size}, alignment={alignment_model.vocab_size}"
    
    
    def get_system_info(self):
        """Get complete system information."""
        base_info = self.base_model.get_model_info()
        alignment_info = self.alignment_model.get_architecture_info()
        
        return {
            "base_model": base_info,
            "alignment_model": alignment_info,
            "compatibility": {
                "adaptive_pooling_needed": base_info["hidden_size"] != alignment_info["hidden_size"],
                "pooling_ratio": base_info["hidden_size"] / alignment_info["hidden_size"],
                "vocab_compatible": base_info["vocab_size"] == alignment_info["vocab_size"]
            }
        }
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None,
               use_cache=False, alpha=1.0, return_components=False):
        """Forward with hierarchical attention-based alignment."""
        
        # Get base model outputs (with last hidden state)
        with torch.no_grad():
            base_outputs = self.base_model.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=use_cache, output_hidden_states=True
            )
        
        # Extract last hidden state
        last_hidden_state = base_outputs.hidden_states[-1]  # [B, L, hidden_size]
        
        # Hierarchical attention-based alignment
        alignment_vector, attention_info = self.alignment_model(last_hidden_state)
        
        # Combine
        final_logits = base_outputs.logits + alpha * alignment_vector
        
        outputs = {
            "logits": final_logits,
            "past_key_values": base_outputs.past_key_values if use_cache else None
        }
        
        if return_components:
            outputs.update({
                "base_logits": base_outputs.logits,
                "alignment_vector": alignment_vector,
                "last_hidden_state": last_hidden_state,
                "alpha": alpha,
                "attention_info": attention_info  # New: attention analysis data
            })
        
        return outputs
    
    def generate_with_alignment(self, input_ids, max_length=128, temperature=1.0, alpha=1.0,
                              do_sample=False, top_p=1.0, attention_mask=None):
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
                
                # Track strength and attention info
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
        
        # Add attention info from the last step if available
        if 'attention_info' in outputs:
            stats['attention_info'] = outputs['attention_info']
        
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
               device="cuda", torch_dtype=torch.bfloat16):
    """Create MAV model with single-layer alignment.
    
    Args:
        base_model_name: Base model to use
        device: Device to use
        torch_dtype: Data type
    """
    
    base_model = FrozenBaseModel(base_model_name, device, torch_dtype)
    
    # Simple alignment architecture - use base model's dimensions
    alignment_model = AlignmentModel(
        hidden_size=base_model.hidden_size, 
        vocab_size=base_model.vocab_size,
        device=device, 
        torch_dtype=torch_dtype
    )
    
    return MAV(base_model, alignment_model, device)