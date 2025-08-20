#!/usr/bin/env python3
"""
Generate responses using MAV (Modular Alignment Vectors) for evaluation.
Supports HH-RLHF, AlpacaEval, MT-Bench datasets.
"""

import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os

# Add MAV directory to path
sys.path.append('/home/ibel/research/MAV')
from inference import MAVInference


def generate_mav_responses(num_samples=300, dataset_name="hh_rlhf", random_seed=42, 
                          max_new_tokens=1024, alpha=1.0, checkpoint_path=None):
    """Generate responses using MAV model."""
    
    print("=== MAV Response Generation ===")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Configuration
    base_model = "argsearch/llama-7b-sft-float32"
    
    # Default checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = "/home/ibel/research/MAV/outputs/mav/mav-llama-7b-sft-full-hh-epoch_1-beta_0.1-lr_5e-5-l2_0.1/best_alignment.pt"
    
    try:
        # Use GPU 1 if available (GPU 0 might be occupied)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device = "cuda:1"  # Use GPU 1
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        
        print(f"Loading MAV model on: {device}")
        print(f"MAV checkpoint: {checkpoint_path}")
        print(f"Base model: {base_model}")
        print(f"Alpha: {alpha}")
        
        # Initialize MAV inference
        mav_inference = MAVInference(
            alignment_checkpoint_path=checkpoint_path,
            initial_base_model=base_model,
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        print("MAV model loaded successfully!")
        
        # Load dataset based on dataset_name
        if dataset_name == "hh_rlhf":
            print("Loading HH-RLHF test dataset...")
            dataset = load_dataset("Dahoas/full-hh-rlhf", split="test")
            prompt_key = "chosen"
            assistant_separator = "\n\nAssistant:"
            # Use 128 tokens for HH-RLHF dataset
            max_new_tokens = 128
        elif dataset_name == "alpaca_eval":
            print("Loading AlpacaEval dataset from local file...")
            with open("/home/ibel/research/MAV/evaluation/alpaca_eval.json", "r") as f:
                alpaca_data = json.load(f)
            dataset = Dataset.from_list(alpaca_data)
            prompt_key = "instruction"
            assistant_separator = ""
        elif dataset_name == "mt_bench":
            print("Loading MT-Bench dataset...")
            dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
            prompt_key = "prompt"
            assistant_separator = ""
        else:
            print(f"Unknown dataset: {dataset_name}, falling back to HH-RLHF")
            dataset = load_dataset("Dahoas/full-hh-rlhf", split="test")
            prompt_key = "chosen"
            assistant_separator = "\n\nAssistant:"
        
        # Sample random indices
        all_indices = list(range(len(dataset)))
        sampled_indices = random.sample(all_indices, min(num_samples, len(dataset)))
        sampled_data = dataset.select(sampled_indices)
        
        print(f"Generating responses for {len(sampled_indices)} samples...")
        
        results = []
        
        for i, sample in enumerate(tqdm(sampled_data, desc="Generating MAV responses")):
            try:
                # Extract prompt based on dataset format
                if dataset_name == "hh_rlhf":
                    full_conversation = sample[prompt_key]
                    if assistant_separator in full_conversation:
                        prompt = full_conversation.split(assistant_separator)[0] + assistant_separator
                    else:
                        prompt = full_conversation + assistant_separator
                elif dataset_name == "alpaca_eval":
                    prompt = sample[prompt_key]
                elif dataset_name == "mt_bench":
                    # MT-Bench: Handle both turns properly
                    prompts_list = sample[prompt_key] if isinstance(sample[prompt_key], list) else [sample[prompt_key]]
                    
                    # Generate responses for all turns
                    mt_bench_responses = []
                    conversation_history = ""
                    
                    for turn_idx, question in enumerate(prompts_list):
                        # Build conversation history
                        if turn_idx == 0:
                            turn_prompt = f"Human: {question}\n\nAssistant:"
                        else:
                            # Use previous conversation + new question
                            turn_prompt = conversation_history + f"\n\nHuman: {question}\n\nAssistant:"
                        
                        # Generate response for this turn using MAV
                        response_data = mav_inference.generate_response(
                            prompt=turn_prompt,
                            alpha=alpha,
                            max_length=max_new_tokens,
                            temperature=1.0,
                            do_sample=False,
                            top_p=1.0
                        )
                        
                        # Extract just the response part
                        full_response = response_data["response"]
                        turn_response = full_response[len(turn_prompt):].strip()
                        
                        mt_bench_responses.append({
                            'turn': turn_idx + 1,
                            'question': question,
                            'response': turn_response,
                            'alignment_strength': response_data["stats"]["avg_alignment_strength"]
                        })
                        
                        # Update conversation history for next turn
                        conversation_history = turn_prompt + " " + turn_response
                    
                    # For compatibility with the rest of the code, use first turn as primary
                    prompt = f"Human: {prompts_list[0]}\n\nAssistant:"
                    generated_text = mt_bench_responses[0]['response']
                    avg_alignment_strength = np.mean([r['alignment_strength'] for r in mt_bench_responses])
                    
                    # Build result dict for MT-Bench
                    result = {
                        'sample_id': sampled_indices[i],
                        'prompt': prompt,
                        'response': generated_text,
                        'method': f'MAV (alpha={alpha})',
                        'dataset': dataset_name,
                        'model_path': base_model,
                        'mav_checkpoint': checkpoint_path,
                        'alpha': alpha,
                        'alignment_strength': avg_alignment_strength,
                        'mt_bench_turns': mt_bench_responses,
                        'prompt_id': sample.get('prompt_id', 'unknown'),
                        'category': sample.get('category', 'unknown'),
                        'full_prompt': sample[prompt_key]
                    }
                    results.append(result)
                    continue  # Skip normal generation since we already did it
                else:
                    # Fallback for unknown datasets
                    prompt = sample.get(prompt_key, str(sample))
                
                # Skip normal generation for MT-Bench (handled above)
                if dataset_name != "mt_bench":
                    # Generate response using MAV
                    response_data = mav_inference.generate_response(
                        prompt=prompt,
                        alpha=alpha,
                        max_length=max_new_tokens,
                        temperature=1.0,
                        do_sample=False,
                        top_p=1.0
                    )
                    
                    # Extract just the response part
                    full_response = response_data["response"]
                    generated_text = full_response[len(prompt):].strip()
                    
                    # Build result dict
                    result = {
                        'sample_id': sampled_indices[i],
                        'prompt': prompt,
                        'response': generated_text,
                        'method': f'MAV (alpha={alpha})',
                        'dataset': dataset_name,
                        'model_path': base_model,
                        'mav_checkpoint': checkpoint_path,
                        'alpha': alpha,
                        'alignment_strength': response_data["stats"]["avg_alignment_strength"]
                    }
                
                # Add reference data if available
                if dataset_name == "hh_rlhf":
                    result['reference_chosen'] = sample["chosen"].split("\n\nAssistant:")[-1].strip()
                    result['reference_rejected'] = sample["rejected"].split("\n\nAssistant:")[-1].strip()
                elif dataset_name == "alpaca_eval" and "output" in sample:
                    result['reference_output'] = sample["output"]
                
                results.append(result)
                
            except Exception as e:
                # Clear GPU memory on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Error processing sample {i}: {e}")
                results.append({
                    'sample_id': sampled_indices[i],
                    'prompt': prompt if 'prompt' in locals() else "N/A",
                    'response': f"[ERROR: {str(e)}]",
                    'method': f'MAV (alpha={alpha})',
                    'dataset': dataset_name,
                    'error': str(e)
                })
            
            # Clear GPU memory every 10 samples
            if torch.cuda.is_available() and i % 10 == 0:
                torch.cuda.empty_cache()
        
        # Save results
        output_dir = f"/home/ibel/research/MAV/evaluation/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/mav_{dataset_name}_responses_{num_samples}_alpha_{alpha}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"MAV responses saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MAV responses for evaluation")
    parser.add_argument("num_samples", type=int, nargs="?", default=300, help="Number of samples to generate")
    parser.add_argument("--dataset", type=str, default="hh_rlhf", 
                       choices=["hh_rlhf", "alpaca_eval", "mt_bench"],
                       help="Dataset to use for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--alpha", type=float, default=1.0, help="MAV alignment strength (alpha parameter)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to MAV checkpoint (default: best_alignment.pt)")
    
    args = parser.parse_args()
    generate_mav_responses(args.num_samples, args.dataset, args.seed, args.max_new_tokens, 
                          args.alpha, args.checkpoint)