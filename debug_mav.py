#!/usr/bin/env python3
"""Debug MAV generation issues."""

import torch
import sys
sys.path.append('/home/ibel/research/MAV')

from inference import MAVInference
import numpy as np

def debug_mav():
    """Debug MAV's repetition issue."""
    
    # Load model - use the newly trained model
    checkpoint_path = "/home/ibel/research/MAV/outputs/mav/simplified-mav-llama-7b-sft-full-hh-epoch_1-beta_0.1-lr_5e-5-l2_0.001/alignment_step_3151.pt"
    
    print("Loading MAV model...")
    inference_model = MAVInference(
        alignment_checkpoint_path=checkpoint_path,
        initial_base_model="argsearch/llama-7b-sft-float32",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16
    )
    
    # Test prompt
    prompt = "What is the meaning of life?"
    
    print("\n" + "="*60)
    print("Testing different alpha values:")
    print("="*60)
    
    # Test with different alphas
    for alpha in [0.0, 0.1, 0.5, 1.0]:
        print(f"\n--- Alpha = {alpha} ---")
        result = inference_model.generate_response(
            prompt=prompt,
            max_length=50,
            alpha=alpha,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        print(f"Response: {result['response']}")
        print(f"Alignment strength: {result['stats']['avg_alignment_strength']:.4f}")
        
        # Check for attention info
        if 'attention_info' in result['stats']:
            attention_info = result['stats']['attention_info']
            if attention_info and 'active_refs' in attention_info:
                active_ref = attention_info['active_refs'][0].item() if hasattr(attention_info['active_refs'][0], 'item') else attention_info['active_refs'][0]
                print(f"Most active reference: #{active_ref}")
    
    # Check alignment vector statistics
    print("\n" + "="*60)
    print("Checking alignment vector statistics:")
    print("="*60)
    
    with torch.no_grad():
        # Generate a simple forward pass
        test_input = inference_model.tokenizer("Hello", return_tensors="pt")
        input_ids = test_input["input_ids"].to(inference_model.model.device)
        attention_mask = test_input["attention_mask"].to(inference_model.model.device)
        
        outputs = inference_model.model(
            input_ids, 
            attention_mask=attention_mask,
            alpha=1.0,
            return_components=True
        )
        
        alignment_vector = outputs["alignment_vector"]
        
        print(f"Alignment vector shape: {alignment_vector.shape}")
        print(f"Alignment vector mean: {alignment_vector.mean().item():.4f}")
        print(f"Alignment vector std: {alignment_vector.std().item():.4f}")
        print(f"Alignment vector min: {alignment_vector.min().item():.4f}")
        print(f"Alignment vector max: {alignment_vector.max().item():.4f}")
        print(f"Alignment vector norm: {alignment_vector.norm(dim=-1).mean().item():.4f}")
        
        # Check top tokens that get boosted
        top_values, top_indices = alignment_vector[0, -1].topk(10)
        print(f"\nTop 10 boosted tokens:")
        for val, idx in zip(top_values, top_indices):
            token = inference_model.tokenizer.decode([idx.item()])
            print(f"  Token '{token}': {val.item():.4f}")
        
        # Check attention info
        if "attention_info" in outputs:
            attention_info = outputs["attention_info"]
            print(f"\nAttention info:")
            print(f"  Cross-attention weights shape: {attention_info['cross_attention_weights'].shape}")
            print(f"  Active refs: {attention_info['active_refs']}")
            
            # Check which reference vectors are most active
            cross_attn = attention_info['cross_attention_weights'][0, -1]  # Last token
            top_refs = cross_attn.topk(5)
            print(f"  Top 5 reference vectors (last token):")
            for val, idx in zip(top_refs.values, top_refs.indices):
                print(f"    Ref #{idx.item()}: {val.item():.4f}")

if __name__ == "__main__":
    debug_mav()