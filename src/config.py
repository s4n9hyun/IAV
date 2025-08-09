from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    base_model_name: str = "argsearch/llama-7b-sft-float32"
    vocab_size: int = 32000
    hidden_size: int = 4096
    freeze_backbone: bool = True
    device: str = "cuda"
    
    def to_dict(self) -> Dict:
        return {
            "base_model_name": self.base_model_name,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "freeze_backbone": self.freeze_backbone,
            "device": self.device
        }


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    beta: float = 0.1
    lambda_kl: float = 0.1
    lambda_l2: float = 0.01
    
    max_length: int = 512
    num_workers: int = 4
    
    save_dir: str = "./checkpoints"
    log_wandb: bool = False
    wandb_project: str = "iav-training"
    
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    def to_dict(self) -> Dict:
        return {
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "beta": self.beta,
            "lambda_kl": self.lambda_kl,
            "lambda_l2": self.lambda_l2,
            "max_length": self.max_length,
            "num_workers": self.num_workers,
            "save_dir": self.save_dir,
            "log_wandb": self.log_wandb,
            "wandb_project": self.wandb_project,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps
        }


@dataclass
class DataConfig:
    datasets: List[Dict] = field(default_factory=lambda: [
        {
            "name": "hh-rlhf",
            "split": "train",
            "sample_size": 10000
        },
        {
            "name": "ultrafeedback",
            "split": "train",
            "sample_size": 10000
        }
    ])
    
    val_datasets: Optional[List[Dict]] = field(default_factory=lambda: [
        {
            "name": "hh-rlhf",
            "split": "test",
            "sample_size": 1000
        }
    ])
    
    mix_ratio: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            "datasets": self.datasets,
            "val_datasets": self.val_datasets,
            "mix_ratio": self.mix_ratio
        }


@dataclass
class InferenceConfig:
    default_alpha: float = 1.0
    max_length: int = 128
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    batch_size: int = 8
    
    def to_dict(self) -> Dict:
        return {
            "default_alpha": self.default_alpha,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "batch_size": self.batch_size
        }


@dataclass
class EvaluationConfig:
    benchmarks: List[str] = field(default_factory=lambda: [
        "alpaca_eval",
        "arena_hard"
    ])
    
    alpha_values: List[float] = field(default_factory=lambda: [
        0.0, 0.5, 1.0, 1.5
    ])
    
    num_samples: int = 100
    output_dir: str = "./evaluation_results"
    
    def to_dict(self) -> Dict:
        return {
            "benchmarks": self.benchmarks,
            "alpha_values": self.alpha_values,
            "num_samples": self.num_samples,
            "output_dir": self.output_dir
        }


@dataclass
class IAVConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "IAVConfig":
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> "IAVConfig":
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "IAVConfig":
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        inference_config = InferenceConfig(**config_dict.get("inference", {}))
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation", {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            inference=inference_config,
            evaluation=evaluation_config
        )
    
    def to_dict(self) -> Dict:
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "inference": self.inference.to_dict(),
            "evaluation": self.evaluation.to_dict()
        }
    
    def save_yaml(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def create_default_config() -> IAVConfig:
    return IAVConfig()


def load_config(path: str) -> IAVConfig:
    if path.endswith('.yaml') or path.endswith('.yml'):
        return IAVConfig.from_yaml(path)
    elif path.endswith('.json'):
        return IAVConfig.from_json(path)
    else:
        raise ValueError(f"Unsupported config file format: {path}")


ALPHA_PRESETS = {
    "none": 0.0,
    "light": 0.5,
    "standard": 1.0,
    "strong": 1.5,
    "max": 2.0
}


BENCHMARK_CONFIGS = {
    "alpaca_eval": {
        "dataset": "tatsu-lab/alpaca_eval",
        "split": "eval",
        "metric": "win_rate",
        "evaluator": "gpt-4"
    },
    "arena_hard": {
        "dataset": "lmsys/arena-hard-auto-v0.1",
        "split": "test",
        "metric": "win_rate",
        "evaluator": "gpt-4"
    },
    "hh_rlhf": {
        "dataset": "Anthropic/hh-rlhf",
        "split": "test",
        "metric": "reward_score",
        "evaluator": "reward_model"
    }
}