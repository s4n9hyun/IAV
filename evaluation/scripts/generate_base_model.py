#!/usr/bin/env python3
"""
Generate responses using base model (llama-7b-sft) for evaluation.
"""

import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

def generate_base_model_responses(num_samples=300, dataset_name="hh_rlhf", random_seed=42, max_new_tokens=1024):
    """Generate responses using base model only."""
    
    print("=== Base Model Response Generation ===")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Configuration - Base model only
    base_model = "argsearch/llama-7b-sft-float32"
    
    try:
        # Load base model
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Use GPU 1 if available (GPU 0 might be occupied)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")  # Use GPU 1
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        
        model = model.to(device)
        print(f"Model loaded on: {device}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Base model loaded successfully!")
        
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
            # Convert to HuggingFace dataset format
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
        
        for i, sample in enumerate(tqdm(sampled_data, desc="Generating base model responses")):
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
                            # Get the first user message
                            first_turn = conversation[0]
                            if isinstance(first_turn, dict) and "content" in first_turn:
                                prompt = first_turn["content"]
                            else:
                                prompt = str(first_turn)
                        else:
                            prompt = str(conversation)
                    else:
                        # Fallback: use any available text field
                        prompt = sample.get("winner_model_a", sample.get("winner_model_b", str(sample)))
                elif dataset_name == "mt_bench":
                    # MT-Bench: Handle both turns properly
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
                        inputs = tokenizer(
                            prompt, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=2048  # Maximum prompt length
                        )
                        if torch.cuda.is_available():
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs.get('attention_mask', None),
                                max_new_tokens=max_new_tokens,
                                do_sample=False,  # Greedy decoding
                                temperature=1.0,
                                top_p=1.0,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                use_cache=True
                            )
                        
                        prompt_length = inputs['input_ids'].shape[1]
                        generated_tokens = outputs[0][prompt_length:]
                        turn_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                        
                        mt_bench_responses.append({
                            'turn': turn_idx + 1,
                            'question': question,
                            'response': turn_response
                        })
                        
                        # Update conversation history for next turn
                        conversation_history = prompt + " " + turn_response
                    
                    # For compatibility with the rest of the code, use first turn as primary
                    prompt = f"Human: {prompts_list[0]}\n\nAssistant:"
                    generated_text = mt_bench_responses[0]['response']
                    
                    # Build result dict for MT-Bench
                    result = {
                        'sample_id': sampled_indices[i],
                        'prompt': prompt,
                        'response': generated_text,
                        'method': 'Base Model',
                        'dataset': dataset_name,
                        'model_path': base_model,
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
                    # Tokenize prompt with padding and truncation
                    inputs = tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=2048
                    )
                    if torch.cuda.is_available():
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Generate response with greedy decoding for reproducibility
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs.get('attention_mask', None),
                            max_new_tokens=max_new_tokens,
                            do_sample=False,  # Greedy decoding
                            temperature=1.0,
                            top_p=1.0,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True
                        )
                    
                    # Decode response
                    prompt_length = inputs['input_ids'].shape[1]
                    generated_tokens = outputs[0][prompt_length:]
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Clean up generated text
                    generated_text = generated_text.strip()
                
                # Build result dict (MT-Bench handled above with continue)
                if dataset_name != "mt_bench":
                    result = {
                        'sample_id': sampled_indices[i],
                        'prompt': prompt,
                        'response': generated_text,
                        'method': 'Base Model',
                        'dataset': dataset_name,
                        'model_path': base_model
                    }
                else:
                    # MT-Bench result already built above
                    result.update({
                        'sample_id': sampled_indices[i],
                        'method': 'Base Model',
                        'dataset': dataset_name,
                        'model_path': base_model
                    })
                
                # Add reference data if available
                if dataset_name == "hh_rlhf":
                    result['reference_chosen'] = sample["chosen"].split("\n\nAssistant:")[-1].strip()
                    result['reference_rejected'] = sample["rejected"].split("\n\nAssistant:")[-1].strip()
                elif dataset_name == "alpaca_eval" and "output" in sample:
                    result['reference_output'] = sample["output"]
                elif dataset_name == "arena_hard":
                    # Add arena-specific metadata
                    if "winner" in sample:
                        result['winner'] = sample["winner"]
                    if "conversation_b" in sample:
                        result['conversation_b'] = sample["conversation_b"]
                elif dataset_name == "mt_bench":
                    # Add MT-Bench specific metadata
                    result['prompt_id'] = sample.get('prompt_id', 'unknown')
                    result['category'] = sample.get('category', 'unknown')
                    result['full_prompt'] = sample[prompt_key]  # Store full multi-turn prompt
                
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
                    'method': 'Base Model',
                    'dataset': dataset_name,
                    'error': str(e)
                })
            
            # Clear GPU memory every 10 samples
            if torch.cuda.is_available() and i % 10 == 0:
                torch.cuda.empty_cache()
        
        # Save results
        # Save to dataset-specific subdirectory
        output_dir = f"/home/ibel/research/MAV/evaluation/outputs/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/base_model_{dataset_name}_responses_{num_samples}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Base model responses saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate base model responses for evaluation")
    parser.add_argument("num_samples", type=int, nargs="?", default=300, help="Number of samples to generate")
    parser.add_argument("--dataset", type=str, default="hh_rlhf", 
                       choices=["hh_rlhf", "alpaca_eval", "arena_hard", "mt_bench"],
                       help="Dataset to use for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    generate_base_model_responses(args.num_samples, args.dataset, args.seed, args.max_new_tokens)