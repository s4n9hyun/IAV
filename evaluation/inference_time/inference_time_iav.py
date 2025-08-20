#!/usr/bin/env python3
"""
IAV (Instance-Adaptive Verifier) Inference Time Measurement
Measures time to generate 128 tokens using IAV method with instance-specific verification
"""

import torch
import time
import json
import argparse
import numpy as np
import os
import sys
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

# Add IAV source to path
sys.path.insert(0, "/home/ibel/research/IAV")

def generate_iav_fixed_tokens(iav_inference, prompt, num_tokens, temperature=0.7, alpha=1.0, do_sample=False):
    """Generate exactly num_tokens using IAV, bypassing EOS stopping"""
    
    # Tokenize input
    inputs = iav_inference.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(iav_inference.device)
    
    generated = inputs["input_ids"].clone()
    alignment_strengths = []
    attention_mask = inputs.get("attention_mask")
    # Disable KV cache for fair comparison with other methods
    past_key_values = None
    use_kv_cache = False
    
    # Generate exactly num_tokens tokens, ignoring EOS
    with torch.no_grad():
        for step in range(num_tokens):  # Force exactly num_tokens
            # Always use full sequence for fair comparison
            current_input_ids = generated
            current_attention_mask = attention_mask
            
            # Forward pass through IAV model (no KV cache for fair comparison)
            outputs = iav_inference.model.forward(
                current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=None,
                use_cache=False,
                alpha=alpha,
                return_components=True
            )
            
            # Get next token logits
            logits = outputs["logits"]
            next_token_logits = logits[:, -1, :] / temperature
            
            # Calculate alignment strength
            alignment_vec = outputs["alignment_vector"][:, -1, :]
            alignment_strength = alignment_vec.norm(dim=-1).mean().item()
            alignment_strengths.append(alignment_strength)
            
            # Sample or select next token
            if do_sample:
                # Apply top-p filtering
                from transformers import top_p_filtering
                filtered_logits = top_p_filtering(next_token_logits, top_p=0.9)
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append token (IGNORE EOS - this is the key difference)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones_like(next_token, device=attention_mask.device)
                ], dim=1)
    
    # Decode result
    generated_text = iav_inference.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    stats = {
        "alignment_strengths": alignment_strengths,
        "avg_alignment_strength": sum(alignment_strengths) / len(alignment_strengths) if alignment_strengths else 0.0
    }
    
    return {
        "generated_text": generated_text,
        "prompt": prompt,
        "alpha": alpha,
        "stats": stats,
        "actual_tokens_generated": num_tokens  # We forced exactly num_tokens
    }

def measure_inference_time(iav_inference, prompt, num_tokens=128, num_runs=10, warmup_runs=2):
    """Measure inference time for IAV generation"""
    
    # Use max_new_tokens approach - force exactly num_tokens generation
    # We'll modify the IAV inference call to force exact token count
    
    # Warmup runs
    print(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        _ = generate_iav_fixed_tokens(
            iav_inference=iav_inference,
            prompt=prompt,
            num_tokens=num_tokens,
            temperature=0.7,
            alpha=1.0,
            do_sample=False
        )
    
    # Actual measurement
    times = []
    generated_token_counts = []
    print(f"Running {num_runs} timed iterations...")
    for _ in tqdm(range(num_runs)):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.perf_counter()
        result = generate_iav_fixed_tokens(
            iav_inference=iav_inference,
            prompt=prompt,
            num_tokens=num_tokens,
            temperature=0.7,
            alpha=1.0,
            do_sample=False
        )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        generation_time = end_time - start_time
        times.append(generation_time)
        
        # Track generated token count for debugging
        # Use the actual tokens generated from the function
        generated_tokens = result.get('actual_tokens_generated', num_tokens)
        generated_token_counts.append(generated_tokens)
        # Allow some variance due to tokenization
        if abs(generated_tokens - num_tokens) > 10:
            print(f"Warning: Generated ~{generated_tokens} tokens instead of {num_tokens}")
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "median_time": np.median(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "all_times": times,
        "tokens_per_second": num_tokens / np.mean(times),
        "generated_token_counts": generated_token_counts,
        "mean_generated_tokens": np.mean(generated_token_counts),
        "expected_tokens": num_tokens
    }

def main():
    parser = argparse.ArgumentParser(description="Measure IAV inference time")
    parser.add_argument("--base_model", type=str, default="argsearch/llama-7b-sft-float32",
                       help="Base model name or path")
    parser.add_argument("--checkpoint_path", type=str, 
                       default="/home/ibel/research/IAV/outputs/llama-7b-sft-iav-HH-epoch_1-beta_0.5-lambda_kl_0.01-lambda_l2_0.1-lr_2e-05-bs_16/best_model.pt",
                       help="Path to IAV checkpoint")
    parser.add_argument("--num_tokens", type=int, default=128,
                       help="Number of tokens to generate")
    parser.add_argument("--num_runs", type=int, default=100,
                       help="Number of runs for timing")
    parser.add_argument("--warmup_runs", type=int, default=2,
                       help="Number of warmup runs")
    parser.add_argument("--prompt", type=str, 
                       default="Human: What are the key benefits of regular exercise? Assistant:",
                       help="Prompt to use for generation")
    parser.add_argument("--output_file", type=str, default="results/iav_times.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda:1",
                       help="Device to use (default cuda:1 to match evaluation scripts)")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="IAV alpha parameter for combining heads")
    
    args = parser.parse_args()
    
    print(f"Loading IAV model from: {args.checkpoint_path}")
    print(f"Base model: {args.base_model}")
    
    try:
        from src.models.iav_model import IAVModel
        from src.inference import IAVInference
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load config
        config = AutoConfig.from_pretrained(args.base_model)
        
        # Use consistent device selection as in evaluation/scripts
        if args.device == "auto":
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                device = torch.device("cuda:1")  # Use GPU 1
            elif torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        
        # Load IAV model (following generate_iav.py exactly)
        print(f"Loading IAV model...")
        model = IAVModel(
            base_model_name=args.base_model,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            device="cpu",  # Load on CPU first
            freeze_backbone=True,
            torch_dtype=torch.bfloat16
        )
        
        # Load checkpoint
        if os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
            if "heads_state_dict" in checkpoint:
                # Load only the trained heads
                model.base_head.load_state_dict(checkpoint["heads_state_dict"]["base_head"])
                model.alignment_head.load_state_dict(checkpoint["heads_state_dict"]["alignment_head"])
                print("Loaded IAV heads from checkpoint")
            else:
                print("WARNING: No heads_state_dict found in checkpoint, trying alternative format")
                # Try loading the full model state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif isinstance(checkpoint, dict):
                    # Try to load what we can
                    model.load_state_dict(checkpoint, strict=False)
        else:
            print(f"WARNING: Checkpoint not found at {args.checkpoint_path}")
        
        # Move to GPU and create inference wrapper
        model = model.to(device)
        iav_inference = IAVInference(
            model=model,
            tokenizer=tokenizer,
            device=str(device),
            default_alpha=args.alpha
        )
        print(f"IAV model loaded successfully on {device}!")
        
    except Exception as e:
        print(f"Failed to load IAV model: {e}")
        import traceback
        traceback.print_exc()
        print("Exiting...")
        return
    
    print(f"\nStarting IAV inference time measurement...")
    print(f"Generating {args.num_tokens} tokens")
    print(f"Alpha: {args.alpha}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Prompt: {args.prompt[:100]}...")
    
    # Measure inference time
    results = measure_inference_time(
        iav_inference=iav_inference,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs
    )
    
    # Add metadata
    results["metadata"] = {
        "base_model": args.base_model,
        "checkpoint": args.checkpoint_path,
        "method": "iav",
        "num_tokens": args.num_tokens,
        "alpha": args.alpha,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "device": str(device)
    }
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS - IAV")
    print("="*50)
    print(f"Mean time: {results['mean_time']:.3f} seconds")
    print(f"Std dev: {results['std_time']:.3f} seconds")
    print(f"Median time: {results['median_time']:.3f} seconds")
    print(f"Min time: {results['min_time']:.3f} seconds")
    print(f"Max time: {results['max_time']:.3f} seconds")
    print(f"Tokens/second: {results['tokens_per_second']:.2f}")
    print("="*50)
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()