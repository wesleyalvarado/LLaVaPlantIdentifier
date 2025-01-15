# config/training_config.py
import os
import torch
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from transformers import EarlyStoppingCallback

@dataclass
class ModelConfig:
    """Model configuration with correct image size for LLaVA."""
    name: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    image_size: int = 336  # Match CLIP's expected size
    patch_size: int = 14   # CLIP patch size
    dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    checkpoint_dir: str = "checkpoints"
    cache_dir: Optional[str] = "cache"
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    max_memory: Optional[Dict[int, str]] = field(default_factory=lambda: {0: "10GiB"})
    gradient_checkpointing: bool = True
    vision_feature_layer: int = -2
    vision_feature_select_strategy: str = "default"

@dataclass
class OptimizationConfig:
    """Configuration for training optimization."""
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    max_steps: int = 1000
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    use_8bit_quantization: bool = False
    use_4bit_quantization: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

@dataclass
class DataConfig:
    """Configuration for data processing."""
    train_batch_size: int = 1
    eval_batch_size: int = 1
    max_length: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    sample_fraction: float = 1.0
    dataset_name: str = "dpdl-benchmark/oxford_flowers102"

def get_training_args(
    model_dir: str,
    model_config: ModelConfig,
    optim_config: OptimizationConfig,
    data_config: DataConfig
) -> TrainingArguments:
    """Get training arguments with corrected precision settings."""
    
    os.makedirs(model_dir, exist_ok=True)
    if model_config.checkpoint_dir:
        os.makedirs(os.path.join(model_dir, model_config.checkpoint_dir), exist_ok=True)
    
    return TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=data_config.train_batch_size,
        per_device_eval_batch_size=data_config.eval_batch_size,
        gradient_accumulation_steps=optim_config.gradient_accumulation_steps,
        learning_rate=optim_config.learning_rate,
        weight_decay=optim_config.weight_decay,
        max_grad_norm=optim_config.max_grad_norm,
        max_steps=optim_config.max_steps,
        warmup_ratio=optim_config.warmup_ratio,
        num_train_epochs=optim_config.num_train_epochs,
        
        # Evaluation settings
        eval_steps=model_config.eval_steps,
        evaluation_strategy="steps",
        save_steps=model_config.save_steps,
        save_strategy="steps",
        save_total_limit=model_config.save_total_limit,
        
        # Updated precision settings
        bf16=True,  # Use bfloat16 instead of fp16
        fp16=False,  # Disable fp16
        
        # Memory optimization
        gradient_checkpointing=model_config.gradient_checkpointing,
        dataloader_num_workers=data_config.num_workers,
        dataloader_pin_memory=data_config.pin_memory,
        
        # Optimizer settings
        optim="adamw_torch",
        
        # Logging
        logging_steps=10,
        logging_dir=os.path.join(model_dir, "logs"),
        report_to=["tensorboard"],
        
        # Misc
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Hub
        push_to_hub=False
    )

def get_early_stopping_callback(config: OptimizationConfig) -> EarlyStoppingCallback:
    """Get early stopping callback."""
    return EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold
    )

def get_model_kwargs(config: ModelConfig) -> dict:
    """Get model initialization kwargs."""
    return {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": getattr(torch, config.dtype),
        "device_map": config.device_map,
        "low_cpu_mem_usage": config.low_cpu_mem_usage
    }