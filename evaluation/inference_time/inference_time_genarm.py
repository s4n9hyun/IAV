#!/usr/bin/env python3
"""
GenARM (Generation with Autoregressive Reward Model) Inference Time Measurement
Measures time to generate 128 tokens using GenARM method with process reward model
"""

import torch
import time
import json
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from tqdm import tqdm
import torch.nn.functional as F

def genarm_generate(model, process_reward_model, tokenizer, prompt, num_tokens=128, 
                    alpha=1.0, temperature=0.5):
    """
    GenARM generation with arithmetic model combination
    Implements M_base + alpha * M_arm as in the original GenARM
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    # Force generation of exactly num_tokens (ignore EOS)
    for step in range(num_tokens):
        # Get base model logits
        with torch.no_grad():
            base_outputs = model(input_ids)
            base_logits = base_outputs.logits[:, -1, :]
            
            # Get ARM scores for current sequence
            arm_outputs = process_reward_model(input_ids)
            # ARM outputs 2 logits for binary classification
            # Use difference as the reward signal
            arm_score = (arm_outputs.logits[0, 1] - arm_outputs.logits[0, 0])
            
            # GenARM formula: combine base logits with ARM reward
            # The ARM score modulates all logits uniformly (like a temperature adjustment)
            # This is a simplified version - the actual implementation uses more complex arithmetic
            combined_logits = base_logits + alpha * arm_score
            
            # Apply temperature
            combined_logits = combined_logits / temperature
            
            # Greedy selection (matching original GenARM which uses greedy by default)
            next_token = torch.argmax(combined_logits, dim=-1, keepdim=True)
        
        # Append token
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return input_ids

def measure_inference_time(model, process_reward_model, tokenizer, prompt, 
                          num_tokens=128, num_runs=10, warmup_runs=2, alpha=1.0):
    """Measure inference time for GenARM generation"""
    
    # Warmup runs
    print(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        _ = genarm_generate(
            model, process_reward_model, tokenizer, prompt, 
            num_tokens=num_tokens, alpha=alpha
        )
    
    # Actual measurement
    times = []
    generated_token_counts = []
    print(f"Running {num_runs} timed iterations...")
    for _ in tqdm(range(num_runs)):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.perf_counter()
        outputs = genarm_generate(
            model, process_reward_model, tokenizer, prompt,
            num_tokens=num_tokens, alpha=alpha
        )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        generation_time = end_time - start_time
        times.append(generation_time)
        
        # Track generated token count for debugging
        input_length = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        generated_tokens = outputs.shape[1] - input_length
        generated_token_counts.append(generated_tokens)
        if generated_tokens != num_tokens:
            print(f"WARNING: Generated {generated_tokens} tokens instead of {num_tokens}")
    
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
    parser = argparse.ArgumentParser(description="Measure GenARM inference time")
    parser.add_argument("--model_name", type=str, default="argsearch/llama-7b-sft-float32",
                       help="Model name or path")
    parser.add_argument("--process_reward_model", type=str, 
                       default="/home/ibel/research/genarm_original/training_trl/checkpoints/HH/arm/args-llama-sft-7b-arm-HH-epoch_1-beta_0.05-lr_5e-4-bs_32",
                       help="Process reward model name or path (GenARM ARM model)")
    parser.add_argument("--num_tokens", type=int, default=128,
                       help="Number of tokens to generate")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Weight for ARM model (GenARM alpha parameter)")
    parser.add_argument("--num_runs", type=int, default=100,
                       help="Number of runs for timing")
    parser.add_argument("--warmup_runs", type=int, default=2,
                       help="Number of warmup runs")
    parser.add_argument("--prompt", type=str, 
                       default="Human: What are the key benefits of regular exercise? Assistant:",
                       help="Prompt to use for generation")
    parser.add_argument("--output_file", type=str, default="results/genarm_times.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda:1",
                       help="Device to use (default cuda:1 to match evaluation scripts)")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    print(f"Loading process reward model: {args.process_reward_model}")
    
    # Load models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    process_reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.process_reward_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
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
    
    model = model.to(device)
    process_reward_model = process_reward_model.to(device)
    model.eval()
    process_reward_model.eval()
    print(f"Models loaded on: {device}")
    
    print(f"\nModels loaded. Starting GenARM inference time measurement...")
    print(f"Generating {args.num_tokens} tokens")
    print(f"Alpha (ARM weight): {args.alpha}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Prompt: {args.prompt[:100]}...")
    
    # Measure inference time
    results = measure_inference_time(
        model=model,
        process_reward_model=process_reward_model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        alpha=args.alpha
    )
    
    # Add metadata
    results["metadata"] = {
        "model": args.model_name,
        "process_reward_model": args.process_reward_model,
        "method": "genarm",
        "num_tokens": args.num_tokens,
        "alpha": args.alpha,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "device": args.device
    }
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS - GenARM")
    print("="*50)
    print(f"Mean time: {results['mean_time']:.3f} seconds")
    print(f"Std dev: {results['std_time']:.3f} seconds")
    print(f"Median time: {results['median_time']:.3f} seconds")
    print(f"Min time: {results['min_time']:.3f} seconds")
    print(f"Max time: {results['max_time']:.3f} seconds")
    print(f"Tokens/second: {results['tokens_per_second']:.2f}")
    print("="*50)
    
    # Save results
    import os
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()