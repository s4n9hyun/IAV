#!/usr/bin/env python3
"""MAV inference script with runtime base model switching."""

import argparse
import torch
from transformers import AutoTokenizer
import os

from models import create_mav, MAV, FrozenBaseModel, AlignmentModel


class MAVInference:
    """MAV inference with base model switching."""
    
    def __init__(self, alignment_checkpoint_path, initial_base_model="argsearch/llama-7b-sft-float32", 
                 device="cuda", torch_dtype=torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load alignment checkpoint
        print(f"Loading alignment checkpoint: {alignment_checkpoint_path}")
        checkpoint = torch.load(alignment_checkpoint_path, map_location="cpu")
        
        # Get alignment config
        alignment_config = checkpoint.get("model_config", {})
        alignment_type = alignment_config.get("alignment_type", "large")
        layer_selection = alignment_config.get("layer_selection", "auto")
        
        print(f"Creating Simple MAV (268M alignment parameters)...")
        
        # Create model
        self.model = create_mav(
            base_model_name=initial_base_model,
            device=device,
            torch_dtype=torch_dtype
        )
        
        # Load alignment weights
        self.model.alignment_model.load_state_dict(checkpoint["alignment_model_state_dict"])
        print(f"✓ Loaded alignment model weights")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(initial_base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Print system info
        self.print_system_info()
    
    def print_system_info(self):
        """Print current system information."""
        system_info = self.model.get_system_info()
        base_info = system_info["base_model"]
        alignment_info = system_info["alignment_model"]
        compatibility = system_info["compatibility"]
        
        print(f"\n📊 Current System Configuration:")
        print(f"  Base Model: {base_info['model_name']}")
        print(f"    - Parameters: {base_info['parameters_B']:.1f}B")
        print(f"    - Hidden size: {base_info['hidden_size']}")
        print(f"    - Vocab size: {base_info['vocab_size']}")
        
        print(f"  Alignment Model: Hierarchical Attention")
        print(f"    - Parameters: {alignment_info['parameters_M']:.1f}M")
        print(f"    - Reference vectors: {self.model.alignment_model.num_alignment_refs}")
        
        print(f"  Compatibility:")
        print(f"    - Adaptive pooling: {'Needed' if compatibility['adaptive_pooling_needed'] else 'Not needed'}")
        print(f"    - Pooling ratio: {compatibility['pooling_ratio']:.2f}")
        print(f"    - Vocab compatible: {'✅' if compatibility['vocab_compatible'] else '❌'}")
    
    def generate_response(self, prompt, max_length=128, alpha=1.0, temperature=1.0, 
                         do_sample=False, top_p=1.0):
        """Generate response with alignment control."""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate
        with torch.no_grad():
            generated, stats = self.model.generate_with_alignment(
                input_ids=input_ids,
                max_length=max_length,
                alpha=alpha,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                attention_mask=attention_mask
            )
        
        # Decode
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        return {
            "response": response,
            "stats": stats,
            "alpha_used": alpha,
            "base_model": self.model.base_model.current_model_name
        }
    
    def compare_alphas(self, prompt, alphas=[0.0, 0.5, 1.0, 2.0], max_length=100):
        """Compare responses with different alpha values."""
        print(f"\n🔍 Comparing different alpha values:")
        print(f"Prompt: {prompt}")
        print("-" * 80)
        
        results = []
        for alpha in alphas:
            result = self.generate_response(prompt, max_length=max_length, alpha=alpha)
            results.append(result)
            
            print(f"\nAlpha = {alpha:.1f}:")
            print(f"Response: {result['response']}")
            print(f"Avg alignment strength: {result['stats']['avg_alignment_strength']:.4f}")
            
            # Show attention info if available
            if 'attention_info' in result['stats']:
                attention_info = result['stats']['attention_info']
                if attention_info and 'active_refs' in attention_info:
                    active_ref = attention_info['active_refs'][0].item()
                    print(f"Most active reference vector: #{active_ref}")
        
        return results
    


def main():
    parser = argparse.ArgumentParser(description="MAV Inference")
    
    # Model
    parser.add_argument("--alignment_checkpoint", required=True, help="Path to alignment checkpoint")
    parser.add_argument("--base_model", default="argsearch/llama-7b-sft-float32", help="Initial base model")
    parser.add_argument("--device", default="cuda", help="Device")
    
    # Generation
    parser.add_argument("--prompt", default="How can I be more helpful?", help="Input prompt")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alignment strength")
    parser.add_argument("--max_length", type=int, default=128, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    
    # Demos
    parser.add_argument("--demo_alpha_comparison", action="store_true", help="Demo alpha comparison")
    
    # Options
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    
    args = parser.parse_args()
    
    # Create inference model
    print("🌍 Initializing MAV Inference")
    print("=" * 60)
    
    inference_model = MAVInference(
        alignment_checkpoint_path=args.alignment_checkpoint,
        initial_base_model=args.base_model,
        device=args.device,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    )
    
    print("\n✅ MAV ready for inference!")
    
    # Basic generation
    print(f"\n🚀 Generating response:")
    print(f"Prompt: {args.prompt}")
    print(f"Alpha: {args.alpha}")
    print("-" * 40)
    
    result = inference_model.generate_response(
        prompt=args.prompt,
        alpha=args.alpha,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    print(f"Response: {result['response']}")
    print(f"Base model: {result['base_model']}")
    print(f"Alignment strength: {result['stats']['avg_alignment_strength']:.4f}")
    
    # Alpha comparison demo
    if args.demo_alpha_comparison:
        inference_model.compare_alphas(args.prompt)
    
    print(f"\n🎉 MAV inference complete!")
    print(f"💡 Try different alpha values with --demo_alpha_comparison")


if __name__ == "__main__":
    main()