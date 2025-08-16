import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Tuple
import logging
from src.models.iav_model import IAVModel, DualHeadWrapper
import time


logger = logging.getLogger(__name__)


class IAVInference:
    def __init__(
        self,
        model: IAVModel,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        default_alpha: float = 1.0
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.default_alpha = default_alpha
        self.wrapper = DualHeadWrapper(model, default_alpha)
        
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 128,
        temperature: float = 0.7,
        alpha: float = None,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        return_stats: bool = False
    ) -> Dict:
        
        if alpha is None:
            alpha = self.default_alpha
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        start_time = time.time()
        
        if num_return_sequences == 1:
            generated_ids, stats = self.model.generate_with_alignment(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                temperature=temperature,
                alpha=alpha,
                do_sample=do_sample,
                top_p=top_p
            )
            
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            
            inference_time = time.time() - start_time
            
            result = {
                "generated_text": generated_text,
                "prompt": prompt,
                "alpha": alpha
            }
            
            if return_stats:
                result.update({
                    "stats": stats,
                    "inference_time": inference_time,
                    "tokens_generated": generated_ids.shape[1] - inputs["input_ids"].shape[1]
                })
            
            return result
        
        else:
            results = []
            for _ in range(num_return_sequences):
                generated_ids, stats = self.model.generate_with_alignment(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    temperature=temperature,
                    alpha=alpha,
                    do_sample=do_sample,
                    top_p=top_p
                )
                
                generated_text = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )
                
                results.append({
                    "generated_text": generated_text,
                    "stats": stats if return_stats else None
                })
            
            inference_time = time.time() - start_time
            
            return {
                "results": results,
                "prompt": prompt,
                "alpha": alpha,
                "inference_time": inference_time
            }
    
    def compare_alphas(
        self,
        prompt: str,
        alphas: List[float],
        max_length: int = 128,
        temperature: float = 0.7
    ) -> List[Dict]:
        
        results = []
        
        for alpha in alphas:
            result = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                alpha=alpha,
                do_sample=True,
                return_stats=True
            )
            results.append(result)
        
        return results
    
    def analyze_alignment_effect(
        self,
        prompt: str,
        alpha: float = 1.0,
        top_k_tokens: int = 10
    ) -> Dict:
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        analysis = self.wrapper.analyze_alignment_vector(
            input_ids=inputs["input_ids"],
            tokenizer=self.tokenizer,
            top_k=top_k_tokens
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                alpha=0.0,
                return_components=True
            )
            base_probs = F.softmax(outputs["base_logits"][0, -1, :], dim=-1)
            
            outputs_aligned = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                alpha=alpha,
                return_components=True
            )
            aligned_probs = F.softmax(outputs_aligned["logits"][0, -1, :], dim=-1)
        
        top_base = torch.topk(base_probs, k=5)
        top_aligned = torch.topk(aligned_probs, k=5)
        
        base_predictions = [
            (self.tokenizer.decode(idx.item()), prob.item())
            for idx, prob in zip(top_base.indices, top_base.values)
        ]
        
        aligned_predictions = [
            (self.tokenizer.decode(idx.item()), prob.item())
            for idx, prob in zip(top_aligned.indices, top_aligned.values)
        ]
        
        return {
            "prompt": prompt,
            "alpha": alpha,
            "alignment_analysis": analysis,
            "base_top_predictions": base_predictions,
            "aligned_top_predictions": aligned_predictions
        }
    
    def stream_generate(
        self,
        prompt: str,
        max_length: int = 128,
        temperature: float = 0.7,
        alpha: float = None,
        do_sample: bool = True,
        top_p: float = 0.9
    ):
        
        if alpha is None:
            alpha = self.default_alpha
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        generated = inputs["input_ids"].clone()
        attention_mask = inputs["attention_mask"].clone()
        past_key_values = None
        
        with torch.no_grad():
            for step in range(max_length):
                # Use only last token for KV cache efficiency after first step
                if step == 0:
                    current_input_ids = generated
                    current_attention_mask = attention_mask
                else:
                    current_input_ids = generated[:, -1:]
                    current_attention_mask = None  # Let model handle it
                
                try:
                    hidden_states, past_key_values = self.model.get_hidden_states(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    
                    # Get logits from both heads
                    base_logits = self.model.base_head(hidden_states)
                    alignment_logits = self.model.alignment_head(hidden_states)
                    
                    # Combine with alpha
                    combined_logits = base_logits + alpha * alignment_logits
                    next_token_logits = combined_logits[:, -1, :] / temperature
                    
                except Exception as e:
                    # Fallback without KV cache if there's an issue
                    logger.warning(f"KV cache failed, using fallback: {e}")
                    outputs = self.model(
                        input_ids=generated,
                        attention_mask=attention_mask,
                        alpha=alpha,
                        return_components=True
                    )
                    next_token_logits = outputs["logits"][:, -1, :] / temperature
                    past_key_values = None  # Reset cache
                
                if do_sample:
                    filtered_logits = self._top_p_filtering(next_token_logits, top_p=top_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones_like(next_token)
                ], dim=1)
                
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                yield token_text
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
    
    def _top_p_filtering(self, logits, top_p=0.9, min_tokens_to_keep=1):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits


class BatchInference:
    def __init__(
        self,
        model: IAVModel,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        batch_size: int = 8
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.model.eval()
    
    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 128,
        temperature: float = 0.7,
        alpha: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9
    ) -> List[str]:
        
        all_results = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self._batch_generate_tokens(
                    inputs=inputs,
                    max_length=max_length,
                    temperature=temperature,
                    alpha=alpha,
                    do_sample=do_sample,
                    top_p=top_p
                )
            
            for j in range(len(batch_prompts)):
                generated_text = self.tokenizer.decode(
                    generated_ids[j],
                    skip_special_tokens=True
                )
                all_results.append(generated_text)
        
        return all_results
    
    def _batch_generate_tokens(
        self,
        inputs: Dict,
        max_length: int,
        temperature: float,
        alpha: float,
        do_sample: bool,
        top_p: float
    ) -> torch.Tensor:
        
        generated = inputs["input_ids"].clone()
        attention_mask = inputs["attention_mask"].clone()
        batch_size = generated.shape[0]
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        past_key_values = None
        
        for step in range(max_length):
            if finished.all():
                break
            
            # Use KV cache for efficiency
            if step == 0:
                current_input_ids = generated
                current_attention_mask = attention_mask
            else:
                current_input_ids = generated[:, -1:]
                current_attention_mask = None
            
            try:
                # Use KV cache-aware generation
                hidden_states, past_key_values = self.model.get_hidden_states(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # Get logits from both heads
                base_logits = self.model.base_head(hidden_states)
                alignment_logits = self.model.alignment_head(hidden_states)
                
                # Combine with alpha
                combined_logits = base_logits + alpha * alignment_logits
                next_token_logits = combined_logits[:, -1, :] / temperature
                
            except Exception as e:
                # Fallback without KV cache
                logger.warning(f"Batch KV cache failed, using fallback: {e}")
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    alpha=alpha,
                    return_components=False
                )
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                past_key_values = None
            
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            next_tokens[finished] = self.tokenizer.pad_token_id
            
            generated = torch.cat([generated, next_tokens], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                (~finished).unsqueeze(1).float()
            ], dim=1)
            
            finished |= (next_tokens.squeeze(-1) == self.tokenizer.eos_token_id)
        
        return generated