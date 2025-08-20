#!/usr/bin/env python3
"""
Generate responses using DPO (Direct Preference Optimization) model for evaluation.
DPO models are fine-tuned using direct preference optimization on human preference data.
Properly handles multi-turn conversations for MT-Bench.
"""

import sys
import json
import torch
import random
import numpy as np
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def generate_dpo_responses(num_samples=300, dataset_name="hh_rlhf", random_seed=42, max_new_tokens=1024):
    """Generate responses using DPO model."""
    
    print("=== DPO Response Generation ===")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Configuration
    base_model = "argsearch/llama-7b-sft-float32"
    # Use the specified DPO model
    dpo_path = "/home/ibel/research/dpo_1epoch_lr_5e-6/outputs/dpo"
    
    # Generation parameters (consistent with other models for fair comparison)
    temperature = 1.0  # Use same as other models
    do_sample = False  # Greedy decoding like other models
    top_p = 1.0
    
    try:
        # Determine device
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")  # Use GPU 1
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        
        print(f"Using device: {device}")
        
        # Load base model
        print(f"Loading base model: {base_model}")
        base_model_for_dpo = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=None  # Load on CPU first, then move to device
        )
        
        # Merge DPO adapter
        print(f"Merging DPO adapter from {dpo_path}...")
        merged_dpo_model = PeftModel.from_pretrained(base_model_for_dpo, dpo_path)
        final_dpo_model = merged_dpo_model.merge_and_unload()
        print("DPO adapter merged successfully.")
        
        # Move to device
        final_dpo_model = final_dpo_model.to(device)
        final_dpo_model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("DPO model loaded successfully!")
        
        # Create generation function
        def generate_dpo(prompt):
            """Generate response using DPO model."""
            inputs = tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(device)
            
            with torch.no_grad():
                outputs = final_dpo_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Extract only generated part
            prompt_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][prompt_length:]
            return tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
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
            from datasets import Dataset
            dataset = Dataset.from_list(alpaca_data)
            prompt_key = "instruction"
            assistant_separator = ""
        elif dataset_name == "arena_hard":
            print("Loading Arena-Hard dataset...")
            try:
                dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
                prompt_key = "conversation_a"
                assistant_separator = ""
            except Exception as e:
                print(f"Arena-Hard dataset not available: {e}, falling back to HH-RLHF")
                dataset = load_dataset("Dahoas/full-hh-rlhf", split="test")
                prompt_key = "chosen"
                assistant_separator = "\n\nAssistant:"
        elif dataset_name == "mt_bench":
            print("Loading MT-Bench dataset...")
            dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
            prompt_key = "prompt"
            assistant_separator = ""  # MT-Bench uses conversational format
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
        
        for i, sample in enumerate(tqdm(sampled_data, desc="Generating DPO responses")):
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
                elif dataset_name == "arena_hard":
                    # Extract the first user turn from conversation
                    if prompt_key in sample and sample[prompt_key]:
                        conversation = sample[prompt_key]
                        if isinstance(conversation, list) and len(conversation) > 0:
                            first_turn = conversation[0]
                            if isinstance(first_turn, dict) and "content" in first_turn:
                                prompt = first_turn["content"]
                            else:
                                prompt = str(first_turn)
                        else:
                            prompt = str(conversation)
                    else:
                        prompt = sample.get("winner_model_a", sample.get("winner_model_b", str(sample)))
                elif dataset_name == "mt_bench":
                    # MT-Bench: Handle BOTH turns properly (FIXED!)
                    prompts_list = sample[prompt_key] if isinstance(sample[prompt_key], list) else [sample[prompt_key]]
                    
                    # Generate responses for all turns
                    mt_bench_responses = []
                    conversation_history = ""
                    
                    for turn_idx, question in enumerate(prompts_list):
                        # Build conversation history
                        if turn_idx == 0:
                            prompt = f"Human: {question}\n\nAssistant:"
                        else:
                            # Use previous conversation + new question
                            prompt = conversation_history + f"\n\nHuman: {question}\n\nAssistant:"
                        
                        # Generate response for this turn
                        turn_response = generate_dpo(prompt)
                        
                        mt_bench_responses.append({
                            'turn': turn_idx + 1,
                            'question': question,
                            'response': turn_response
                        })
                        
                        # Update conversation history for next turn
                        conversation_history = prompt + " " + turn_response
                    
                    # For compatibility with other scripts, use first turn as primary
                    prompt = f"Human: {prompts_list[0]}\n\nAssistant:"
                    generated_text = mt_bench_responses[0]['response']
                    
                    # Build result dict for MT-Bench
                    result = {
                        'sample_id': sampled_indices[i],
                        'prompt': prompt,
                        'response': generated_text,
                        'method': 'DPO',
                        'dataset': dataset_name,
                        'mt_bench_turns': mt_bench_responses,
                        'prompt_id': sample.get('prompt_id', 'unknown'),
                        'category': sample.get('category', 'unknown'),
                        'full_prompt': sample[prompt_key],
                        'temperature': temperature,
                        'max_new_tokens': max_new_tokens,
                        'dpo_checkpoint': dpo_path
                    }
                    results.append(result)
                    continue  # Skip normal generation since we already did it
                else:
                    # Fallback for unknown datasets
                    prompt = sample.get(prompt_key, str(sample))
                
                # Skip normal generation for MT-Bench (handled above)
                if dataset_name != "mt_bench":
                    # Generate response
                    generated_text = generate_dpo(prompt)
                
                # Build result dict (MT-Bench handled above with continue)
                if dataset_name != "mt_bench":
                    result = {
                        'sample_id': sampled_indices[i],
                        'prompt': prompt,
                        'response': generated_text,
                        'method': 'DPO',
                        'dataset': dataset_name,
                        'temperature': temperature,
                        'max_new_tokens': max_new_tokens,
                        'dpo_checkpoint': dpo_path
                    }
                else:
                    # MT-Bench result already built above
                    result.update({
                        'sample_id': sampled_indices[i],
                        'method': 'DPO',
                        'dataset': dataset_name
                    })
                
                # Add reference data if available
                if dataset_name == "hh_rlhf":
                    result['reference_chosen'] = sample["chosen"].split("\n\nAssistant:")[-1].strip()
                    result['reference_rejected'] = sample["rejected"].split("\n\nAssistant:")[-1].strip()
                elif dataset_name == "alpaca_eval" and "output" in sample:
                    result['reference_output'] = sample["output"]
                elif dataset_name == "arena_hard":
                    if "winner" in sample:
                        result['winner'] = sample["winner"]
                    if "conversation_b" in sample:
                        result['conversation_b'] = sample["conversation_b"]
                elif dataset_name == "mt_bench":
                    # MT-Bench specific metadata already added above
                    pass
                
                results.append(result)
                
                # Clear GPU memory periodically
                if torch.cuda.is_available() and i % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                results.append({
                    'sample_id': sampled_indices[i],
                    'prompt': prompt if 'prompt' in locals() else "N/A",
                    'response': f"[ERROR: {str(e)}]",
                    'method': 'DPO',
                    'dataset': dataset_name,
                    'error': str(e)
                })
        
        # Save results
        # Save to dataset-specific subdirectory
        output_dir = f"/home/ibel/research/MAV/evaluation/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/dpo_{dataset_name}_responses_{num_samples}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"DPO responses saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate DPO model responses for evaluation")
    parser.add_argument("num_samples", type=int, nargs="?", default=300, help="Number of samples to generate")
    parser.add_argument("--dataset", type=str, default="hh_rlhf", 
                       choices=["hh_rlhf", "alpaca_eval", "arena_hard", "mt_bench"],
                       help="Dataset to use for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    generate_dpo_responses(args.num_samples, args.dataset, args.seed, args.max_new_tokens)