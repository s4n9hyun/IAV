#!/usr/bin/env python3
"""
Generate responses using GenARM (Generative Arithmetic) for evaluation.
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

# Add GenARM directory to path
sys.path.append('/home/ibel/research/genarm_original/language-model-arithmetic/src')
from model_arithmetic import ModelArithmetic, PromptedLLM


def generate_genarm_responses(num_samples=300, dataset_name="hh_rlhf", random_seed=42, 
                             max_new_tokens=1024, alpha=1.0, base_model_path=None, arm_model_path=None):
    """Generate responses using GenARM model."""
    
    print("=== GenARM Response Generation ===")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Default model paths if not provided
    if base_model_path is None:
        base_model_path = "argsearch/llama-7b-sft-float32"
    if arm_model_path is None:
        # Use the specified GenARM checkpoint path
        arm_model_path = "/home/ibel/research/genarm_original/training_trl/checkpoints/HH/arm/args-llama-sft-7b-arm-HH-epoch_1-beta_0.05-lr_5e-4-bs_32"
    
    try:
        print(f"Base model: {base_model_path}")
        print(f"ARM model: {arm_model_path}")
        print(f"Alpha: {alpha}")
        
        # Load dataset based on dataset_name
        if dataset_name == "hh_rlhf":
            print("Loading HH-RLHF test dataset...")
            dataset = load_dataset("Dahoas/full-hh-rlhf", split="test")
            prompt_key = "prompt"
            assistant_separator = ""
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
            prompt_key = "prompt"
            assistant_separator = ""
        
        # Sample random indices
        all_indices = list(range(len(dataset)))
        sampled_indices = random.sample(all_indices, min(num_samples, len(dataset)))
        sampled_data = dataset.select(sampled_indices)
        
        print(f"Generating responses for {len(sampled_indices)} samples...")
        
        # Initialize GenARM model
        prompt_template_base = lambda system_prompt, input_string: f"{input_string}"
        prompt_template_arm = lambda system_prompt, input_string: f"{input_string}"
        
        M_base = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_base, model=base_model_path)
        M_arm = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_arm, model=arm_model_path) if alpha != 0 else None
        
        if alpha == 0:
            print("Alpha is 0, using the base model directly.")
            formula = M_base
        else:
            formula = M_base + alpha * M_arm
        
        M = ModelArithmetic(formula, needs_input_tokens_lm_eval=False, lm_eval_task=None)
        
        # Generate function - use identical settings to other models for fair comparison
        generate = lambda prompt: M.generate_text(
            prompt, 
            max_new_tokens=max_new_tokens, 
            batch_size=None, 
            temperature=1.0,  # Match other models
            top_p=1.0,        # Match other models  
            top_k=0,          # Match other models (greedy)
            do_speculation=False
        )[0].removesuffix(M.tokenizer.eos_token)
        
        results = []
        
        for i, sample in enumerate(tqdm(sampled_data, desc="Generating GenARM responses")):
            try:
                # Extract prompt based on dataset format
                if dataset_name == "hh_rlhf":
                    prompt = sample[prompt_key]
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
                        
                        # Generate response for this turn using GenARM
                        turn_response = generate(turn_prompt)
                        
                        # Extract just the response part
                        if turn_prompt in turn_response:
                            turn_response = turn_response[len(turn_prompt):].strip()
                        
                        mt_bench_responses.append({
                            'turn': turn_idx + 1,
                            'question': question,
                            'response': turn_response
                        })
                        
                        # Update conversation history for next turn
                        conversation_history = turn_prompt + " " + turn_response
                    
                    # For compatibility with the rest of the code, use first turn as primary
                    prompt = f"Human: {prompts_list[0]}\n\nAssistant:"
                    generated_text = mt_bench_responses[0]['response']
                    
                    # Build result dict for MT-Bench
                    result = {
                        'sample_id': sampled_indices[i],
                        'prompt': prompt,
                        'response': generated_text,
                        'method': f'GenARM (alpha={alpha})',
                        'dataset': dataset_name,
                        'base_model_path': base_model_path,
                        'arm_model_path': arm_model_path,
                        'alpha': alpha,
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
                    # Generate response using GenARM
                    full_response = generate(prompt)
                    
                    # Extract just the response part
                    if prompt in full_response:
                        generated_text = full_response[len(prompt):].strip()
                    else:
                        generated_text = full_response.strip()
                    
                    # Build result dict
                    result = {
                        'sample_id': sampled_indices[i],
                        'prompt': prompt,
                        'response': generated_text,
                        'method': f'GenARM (alpha={alpha})',
                        'dataset': dataset_name,
                        'base_model_path': base_model_path,
                        'arm_model_path': arm_model_path,
                        'alpha': alpha
                    }
                
                # Add reference data if available
                if dataset_name == "hh_rlhf":
                    if "chosen" in sample:
                        result['reference_chosen'] = sample["chosen"].split("\\n\\nAssistant:")[-1].strip()
                    if "rejected" in sample:
                        result['reference_rejected'] = sample["rejected"].split("\\n\\nAssistant:")[-1].strip()
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
                    'method': f'GenARM (alpha={alpha})',
                    'dataset': dataset_name,
                    'error': str(e)
                })
            
            # Clear GPU memory every 10 samples
            if torch.cuda.is_available() and i % 10 == 0:
                torch.cuda.empty_cache()
        
        # Save results
        output_dir = f"/home/ibel/research/MAV/evaluation/outputs/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/genarm_{dataset_name}_responses_{num_samples}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"GenARM responses saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate GenARM responses for evaluation")
    parser.add_argument("num_samples", type=int, nargs="?", default=300, help="Number of samples to generate")
    parser.add_argument("--dataset", type=str, default="hh_rlhf", 
                       choices=["hh_rlhf", "alpaca_eval", "mt_bench"],
                       help="Dataset to use for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--alpha", type=float, default=1.0, help="GenARM alignment strength (alpha parameter)")
    parser.add_argument("--base_model", type=str, default=None, 
                       help="Path to base model (default: argsearch/llama-7b-sft-float32)")
    parser.add_argument("--arm_model", type=str, default=None, 
                       help="Path to ARM model checkpoint")
    
    args = parser.parse_args()
    generate_genarm_responses(args.num_samples, args.dataset, args.seed, args.max_new_tokens, 
                             args.alpha, args.base_model, args.arm_model)