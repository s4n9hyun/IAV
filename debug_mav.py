#!/usr/bin/env python3
"""Debug MAV model to check if alignment is working properly."""

import torch
import sys
sys.path.append('/home/ibel/research/MAV')
from models import create_mav
from transformers import AutoTokenizer

def debug_mav():
    """Debug MAV model components."""
    
    print("=== MAV Debug ===")
    
    # Load checkpoint
    checkpoint_path = "/home/ibel/research/MAV/outputs/mav/mav-llama-7b-sft-full-hh-epoch_1-beta_0.1-lr_5e-5-l2_0.1/best_alignment.pt"
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Check checkpoint contents
    print("\n1. Checkpoint contents:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: dict with {len(checkpoint[key])} keys")
            if key == "alignment_model_state_dict":
                for param_name, param in list(checkpoint[key].items())[:5]:
                    print(f"    {param_name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # Create MAV model
    print("\n2. Creating MAV model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_mav(
        base_model_name="argsearch/llama-7b-sft-float32",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load alignment weights
    print("\n3. Loading alignment weights...")
    model.alignment_model.load_state_dict(checkpoint["alignment_model_state_dict"])
    
    # Check if weights are non-zero
    print("\n4. Checking alignment model parameters:")
    for name, param in model.alignment_model.named_parameters():
        print(f"  {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}, requires_grad={param.requires_grad}")
    
    # Test forward pass with dummy input
    print("\n5. Testing forward pass...")
    tokenizer = AutoTokenizer.from_pretrained("argsearch/llama-7b-sft-float32")
    tokenizer.pad_token = tokenizer.eos_token
    
    test_prompt = "Hello, how are you?"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Test with different alpha values
    for alpha in [0.0, 1.0, 2.0]:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                alpha=alpha,
                return_components=True
            )
            
            print(f"\n  Alpha={alpha}:")
            print(f"    Base logits shape: {outputs['base_logits'].shape}")
            print(f"    Base logits mean: {outputs['base_logits'].mean():.6f}")
            print(f"    Alignment vector shape: {outputs['alignment_vector'].shape}")
            print(f"    Alignment vector mean: {outputs['alignment_vector'].mean():.6f}")
            print(f"    Alignment vector std: {outputs['alignment_vector'].std():.6f}")
            print(f"    Alignment vector norm: {outputs['alignment_vector'].norm(dim=-1).mean():.6f}")
            print(f"    Final logits mean: {outputs['logits'].mean():.6f}")
            
            # Check if alignment actually changes the output
            if alpha > 0:
                diff = (outputs['logits'] - outputs['base_logits']).abs().mean()
                print(f"    Difference from base: {diff:.6f}")
    
    # Test generation with alignment
    print("\n6. Testing generation with alignment tracking...")
    test_prompt = "Human: How can I help you?\n\nAssistant:"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    generated, stats = model.generate_with_alignment(
        input_ids=input_ids,
        max_length=20,
        alpha=1.0,
        temperature=1.0,
        do_sample=False,
        top_p=1.0
    )
    
    print(f"  Generated tokens: {generated.shape}")
    print(f"  Alignment strengths: {stats['alignment_strengths'][:5]}")
    print(f"  Average alignment strength: {stats['avg_alignment_strength']:.6f}")
    
    # Check if model actually produces different outputs with different alphas
    print("\n7. Comparing outputs with different alphas...")
    outputs_by_alpha = {}
    for alpha in [0.0, 1.0, 2.0]:
        generated, _ = model.generate_with_alignment(
            input_ids=input_ids,
            max_length=20,
            alpha=alpha,
            temperature=1.0,
            do_sample=False,
            top_p=1.0
        )
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs_by_alpha[alpha] = decoded
        print(f"  Alpha {alpha}: {decoded[:100]}")
    
    # Check if outputs are different
    if outputs_by_alpha[0.0] == outputs_by_alpha[1.0] == outputs_by_alpha[2.0]:
        print("\n⚠️  WARNING: All outputs are identical! Alignment is not working!")
    else:
        print("\n✅ Outputs vary with alpha - alignment is working!")
    
    return checkpoint, model


if __name__ == "__main__":
    checkpoint, model = debug_mav()