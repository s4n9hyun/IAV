#!/usr/bin/env python3
"""
MAV Controllability Evaluation - Test alpha value effects on responses.
This script evaluates how different alpha values affect MAV's response generation,
helping to understand the alignment strength and controllability.
"""

import sys
import torch
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
import os

# Add MAV directory to path
sys.path.append('/home/ibel/research/MAV')
from inference import MAVInference


def evaluate_controllability(num_samples=50, dataset_name="hh_rlhf", checkpoint_path=None):
    """Evaluate MAV controllability across different alpha values."""
    
    print("=== MAV Controllability Evaluation ===")
    
    # Alpha values to test
    alpha_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Configuration
    base_model = "argsearch/llama-7b-sft-float32"
    
    # Default checkpoint path if not provided - use new hierarchical attention model
    if checkpoint_path is None:
        checkpoint_path = "/home/ibel/research/MAV/outputs/mav/simplified-mav-llama-7b-sft-full-hh-epoch_1-beta_0.1-lr_1e-5-l2_0.01/alignment_step_3151.pt"
    
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
        print(f"Testing alpha values: {alpha_values}")
        
        # Initialize MAV inference
        mav_inference = MAVInference(
            alignment_checkpoint_path=checkpoint_path,
            initial_base_model=base_model,
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        print("MAV model loaded successfully!")
        
        # Load test dataset
        if dataset_name == "hh_rlhf":
            print("Loading HH-RLHF test dataset...")
            dataset = load_dataset("Dahoas/full-hh-rlhf", split="test")
            prompt_key = "chosen"
            assistant_separator = "\n\nAssistant:"
        else:
            raise ValueError(f"Dataset {dataset_name} not supported yet")
        
        # Sample prompts for testing
        selected_indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
        test_samples = dataset.select(selected_indices)
        
        print(f"Testing {len(test_samples)} prompts with {len(alpha_values)} alpha values...")
        
        results = []
        
        for i, sample in enumerate(test_samples):
            print(f"\nProcessing sample {i+1}/{len(test_samples)}")
            
            # Extract prompt
            full_conversation = sample[prompt_key]
            if assistant_separator in full_conversation:
                prompt = full_conversation.split(assistant_separator)[0] + assistant_separator
            else:
                prompt = full_conversation + assistant_separator
            
            sample_results = {
                "sample_id": int(selected_indices[i]),
                "prompt": prompt,
                "responses": {},
                "alignment_strengths": {},
                "diversity_metrics": {}
            }
            
            responses = []
            alignment_strengths = []
            
            # Generate responses for different alpha values
            for alpha in alpha_values:
                try:
                    response_data = mav_inference.generate_response(
                        prompt=prompt,
                        alpha=alpha,
                        max_length=128,  # Shorter for faster evaluation
                        temperature=1.0,
                        do_sample=False,
                        top_p=1.0
                    )
                    
                    # Extract just the response part
                    full_response = response_data["response"]
                    generated_text = full_response[len(prompt):].strip()
                    
                    sample_results["responses"][f"alpha_{alpha}"] = generated_text
                    sample_results["alignment_strengths"][f"alpha_{alpha}"] = float(response_data["stats"]["avg_alignment_strength"])
                    
                    responses.append(generated_text)
                    alignment_strengths.append(response_data["stats"]["avg_alignment_strength"])
                    
                    print(f"  Alpha {alpha}: alignment_strength = {response_data['stats']['avg_alignment_strength']:.4f}")
                    
                except Exception as e:
                    print(f"  Error with alpha {alpha}: {e}")
                    sample_results["responses"][f"alpha_{alpha}"] = f"[ERROR: {str(e)}]"
                    sample_results["alignment_strengths"][f"alpha_{alpha}"] = 0.0
                    responses.append("")
                    alignment_strengths.append(0.0)
            
            # Calculate diversity metrics
            sample_results["diversity_metrics"] = calculate_diversity_metrics(responses)
            sample_results["alignment_strength_range"] = {
                "min": float(min(alignment_strengths)),
                "max": float(max(alignment_strengths)),
                "range": float(max(alignment_strengths) - min(alignment_strengths)),
                "std": float(np.std(alignment_strengths))
            }
            
            results.append(sample_results)
            
            # Show example for first few samples
            if i < 3:
                print(f"  Example responses for sample {i+1}:")
                for alpha in alpha_values[:3]:  # Show first 3 alphas
                    response = sample_results["responses"].get(f"alpha_{alpha}", "N/A")
                    print(f"    Alpha {alpha}: {response[:100]}..." if len(response) > 100 else f"    Alpha {alpha}: {response}")
        
        # Calculate overall statistics
        overall_stats = calculate_overall_statistics(results, alpha_values)
        
        # Save results
        output_dir = "/home/ibel/research/MAV/evaluation/controllability"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/controllability_evaluation_{num_samples}_samples.json"
        
        final_results = {
            "config": {
                "num_samples": num_samples,
                "dataset": dataset_name,
                "alpha_values": alpha_values,
                "checkpoint_path": checkpoint_path,
                "base_model": base_model
            },
            "overall_statistics": overall_stats,
            "sample_results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n=== Controllability Evaluation Results ===")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        print(f"\nOverall Statistics:")
        print(f"Average alignment strength by alpha:")
        for alpha in alpha_values:
            avg_strength = overall_stats["avg_alignment_strength_by_alpha"].get(f"alpha_{alpha}", 0)
            print(f"  Alpha {alpha}: {avg_strength:.4f}")
        
        print(f"\nAverage response diversity:")
        print(f"  Character length diversity: {overall_stats['avg_diversity_metrics']['avg_char_length_std']:.2f}")
        print(f"  Unique responses ratio: {overall_stats['avg_diversity_metrics']['avg_unique_ratio']:.3f}")
        
        print(f"\nAlignment strength control:")
        print(f"  Average range: {overall_stats['avg_alignment_range']:.4f}")
        print(f"  Average std: {overall_stats['avg_alignment_std']:.4f}")
        
        return output_file
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_diversity_metrics(responses):
    """Calculate diversity metrics for a set of responses."""
    if not responses or all(not r for r in responses):
        return {"avg_char_length": 0, "char_length_std": 0, "unique_responses": 0, "unique_ratio": 0}
    
    # Filter out empty responses
    valid_responses = [r for r in responses if r and isinstance(r, str)]
    
    if not valid_responses:
        return {"avg_char_length": 0, "char_length_std": 0, "unique_responses": 0, "unique_ratio": 0}
    
    # Character length statistics
    char_lengths = [len(r) for r in valid_responses]
    avg_char_length = np.mean(char_lengths)
    char_length_std = np.std(char_lengths)
    
    # Unique responses
    unique_responses = len(set(valid_responses))
    unique_ratio = unique_responses / len(valid_responses)
    
    return {
        "avg_char_length": avg_char_length,
        "char_length_std": char_length_std,
        "unique_responses": unique_responses,
        "unique_ratio": unique_ratio
    }


def calculate_overall_statistics(results, alpha_values):
    """Calculate overall statistics across all samples."""
    
    # Alignment strength statistics
    alignment_stats = {}
    for alpha in alpha_values:
        alpha_key = f"alpha_{alpha}"
        strengths = [r["alignment_strengths"].get(alpha_key, 0) for r in results]
        alignment_stats[alpha_key] = {
            "avg": np.mean(strengths),
            "std": np.std(strengths),
            "min": np.min(strengths),
            "max": np.max(strengths)
        }
    
    # Diversity statistics
    diversity_stats = []
    for result in results:
        if "diversity_metrics" in result:
            diversity_stats.append(result["diversity_metrics"])
    
    if diversity_stats:
        avg_diversity = {
            "avg_char_length_std": np.mean([d["char_length_std"] for d in diversity_stats]),
            "avg_unique_ratio": np.mean([d["unique_ratio"] for d in diversity_stats])
        }
    else:
        avg_diversity = {"avg_char_length_std": 0, "avg_unique_ratio": 0}
    
    # Alignment control statistics
    alignment_ranges = [r["alignment_strength_range"]["range"] for r in results]
    alignment_stds = [r["alignment_strength_range"]["std"] for r in results]
    
    return {
        "avg_alignment_strength_by_alpha": {k: v["avg"] for k, v in alignment_stats.items()},
        "alignment_strength_stats": alignment_stats,
        "avg_diversity_metrics": avg_diversity,
        "avg_alignment_range": np.mean(alignment_ranges),
        "avg_alignment_std": np.mean(alignment_stds)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MAV controllability across alpha values")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to test")
    parser.add_argument("--dataset", type=str, default="hh_rlhf", choices=["hh_rlhf"], help="Dataset to use")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to MAV checkpoint")
    
    args = parser.parse_args()
    
    output_file = evaluate_controllability(
        num_samples=args.num_samples,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint
    )
    
    if output_file:
        print(f"\nControllability evaluation completed successfully!")
        print(f"Results saved to: {output_file}")
    else:
        print("Controllability evaluation failed!")