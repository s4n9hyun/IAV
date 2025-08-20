#!/usr/bin/env python3
"""
DPO Model Inference Time Measurement
Measures time to generate 128 tokens using the DPO fine-tuned model
"""

import torch
import time
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from tqdm import tqdm

def measure_inference_time(model, tokenizer, prompt, num_tokens=128, num_runs=10, warmup_runs=2):
    """Measure inference time for generating num_tokens"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    # Warmup runs
    print(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                min_new_tokens=num_tokens,  # Force exactly num_tokens
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None,  # Disable EOS to force generation
                use_cache=False,  # Disable KV cache for fair comparison
            )
    
    # Actual measurement
    times = []
    generated_token_counts = []
    print(f"Running {num_runs} timed iterations...")
    for _ in tqdm(range(num_runs)):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                min_new_tokens=num_tokens,  # Force exactly num_tokens
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None,  # Disable EOS to force generation
                use_cache=False,  # Disable KV cache for fair comparison
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        generation_time = end_time - start_time
        times.append(generation_time)
        
        # Track generated token count for debugging
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
    parser = argparse.ArgumentParser(description="Measure DPO model inference time")
    parser.add_argument("--base_model", type=str, default="argsearch/llama-7b-sft-float32",
                       help="Base model name or path")
    parser.add_argument("--checkpoint_path", type=str, default="/home/ibel/research/dpo_1epoch/outputs/dpo",
                       help="Path to DPO checkpoint")
    parser.add_argument("--num_tokens", type=int, default=128,
                       help="Number of tokens to generate")
    parser.add_argument("--num_runs", type=int, default=100,
                       help="Number of runs for timing")
    parser.add_argument("--warmup_runs", type=int, default=2,
                       help="Number of warmup runs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--prompt", type=str, 
                       default="Human: What are the key benefits of regular exercise? Assistant:",
                       help="Prompt to use for generation")
    parser.add_argument("--output_file", type=str, default="results/dpo_times.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda:1",
                       help="Device to use (default cuda:1 to match evaluation scripts)")
    
    args = parser.parse_args()
    
    print(f"Loading base model: {args.base_model}")
    print(f"Loading DPO checkpoint: {args.checkpoint_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
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
    
    base_model = base_model.to(device)
    print(f"Model loaded on: {device}")
    
    # Load DPO adapter
    model = PeftModel.from_pretrained(
        base_model,
        args.checkpoint_path,
        is_trainable=False
    )
    model.eval()
    
    # Merge adapter for faster inference (optional)
    print("Merging adapter weights for faster inference...")
    model = model.merge_and_unload()
    
    print(f"\nDPO model loaded. Starting inference time measurement...")
    print(f"Generating {args.num_tokens} tokens")
    print(f"Number of runs: {args.num_runs}")
    print(f"Prompt: {args.prompt[:100]}...")
    
    # Measure inference time
    results = measure_inference_time(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs
    )
    
    # Add metadata
    results["metadata"] = {
        "base_model": args.base_model,
        "checkpoint": args.checkpoint_path,
        "method": "dpo",
        "num_tokens": args.num_tokens,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "device": args.device,
        "batch_size": args.batch_size
    }
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS - DPO Model")
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