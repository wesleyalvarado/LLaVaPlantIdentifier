# config/training_config.py
import os
import torch
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
from transformers import EarlyStoppingCallback

@dataclass
class ModelConfig:
    """Model configuration with correct image size for LLaVA."""
    name: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    image_size: int = 336  # Match CLIP's expected size
    patch_size: int = 14   # CLIP patch size
    dtype: str = "bfloat16"  # Changed from float16 for better stability
    device_map: str = "auto"
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    checkpoint_dir: str = "checkpoints"
    cache_dir: Optional[str] = "cache"
    save_steps: int = 50  # More frequent saving
    eval_steps: int = 50   # More frequent evaluation
    save_total_limit: int = 3
    max_memory: Optional[Dict[int, str]] = field(
        default_factory=lambda: {0: "10GiB", "cpu": "30GiB"}
    )
    gradient_checkpointing: bool = True
    vision_feature_layer: int = -2
    vision_feature_select_strategy: str = "default"

@dataclass
class OptimizationConfig:
    """Configuration for training optimization."""
    learning_rate: float = 5e-7  # Reduced for stability
    weight_decay: float = 0.01
    warmup_ratio: float = 0.2    # Increased warmup period
    max_grad_norm: float = 0.5   # Gradient clipping
    gradient_accumulation_steps: int = 8  # Increased for stability
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means train for full num_epochs
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    use_8bit_quantization: bool = False
    use_4bit_quantization: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # Changed to bfloat16
    bnb_4bit_quant_type: str = "nf4"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    scheduler_type: str = "cosine"  # Added cosine scheduler
    lr_scheduler_warmup_steps: int = 100
    loss_scaling_factor: float = 1.0

@dataclass
class DataConfig:
    """Configuration for data processing."""
    train_batch_size: int = 1
    eval_batch_size: int = 1
    max_length: int = 512  # Increased from 64
    num_workers: int = 2
    pin_memory: bool = True
    sample_fraction: float = 1.0
    dataset_name: str = "dpdl-benchmark/oxford_flowers102"
    image_column: str = "image"
    caption_column: str = "label"
    cache_dir: Optional[str] = "dataset_cache"
    preprocessing_num_workers: Optional[int] = 4
    augmentation_enabled: bool = True
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    logging_dir: str = "logs"
    logging_strategy: str = "steps"
    logging_steps: int = 10
    log_level: str = "info"
    save_strategy: str = "steps"
    save_total_limit: int = 3
    save_safetensors: bool = True
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    wandb_project: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

def get_training_args(
    model_dir: str,
    model_config: ModelConfig,
    optim_config: OptimizationConfig,
    data_config: DataConfig,
    logging_config: Optional[LoggingConfig] = None
) -> TrainingArguments:
    """Get training arguments with optimized settings."""
    
    if logging_config is None:
        logging_config = LoggingConfig()
    
    os.makedirs(model_dir, exist_ok=True)
    if model_config.checkpoint_dir:
        os.makedirs(os.path.join(model_dir, model_config.checkpoint_dir), exist_ok=True)
    
    return TrainingArguments(
        # Directory settings
        output_dir=model_dir,
        overwrite_output_dir=True,
        
        # Training hyperparameters
        per_device_train_batch_size=data_config.train_batch_size,
        per_device_eval_batch_size=data_config.eval_batch_size,
        gradient_accumulation_steps=optim_config.gradient_accumulation_steps,
        learning_rate=optim_config.learning_rate,
        weight_decay=optim_config.weight_decay,
        max_grad_norm=optim_config.max_grad_norm,
        max_steps=optim_config.max_steps,
        warmup_ratio=optim_config.warmup_ratio,
        num_train_epochs=optim_config.num_train_epochs,
        
        # Scheduler settings
        lr_scheduler_type=optim_config.scheduler_type,
        warmup_steps=optim_config.lr_scheduler_warmup_steps,
        
        # Evaluation settings
        evaluation_strategy="steps",
        eval_steps=model_config.eval_steps,
        save_strategy="steps",
        save_steps=model_config.save_steps,
        save_total_limit=model_config.save_total_limit,
        
        # Precision settings
        bf16=True,  # Use bfloat16
        fp16=False, # Disable fp16
        fp16_opt_level="O2",
        
        # Memory optimization
        gradient_checkpointing=model_config.gradient_checkpointing,
        optim="adamw_torch",
        adam_beta1=optim_config.adam_beta1,
        adam_beta2=optim_config.adam_beta2,
        adam_epsilon=optim_config.adam_epsilon,
        
        # DataLoader settings
        dataloader_num_workers=data_config.num_workers,
        dataloader_pin_memory=data_config.pin_memory,
        
        # Logging
        logging_dir=os.path.join(model_dir, logging_config.logging_dir),
        logging_strategy=logging_config.logging_strategy,
        logging_steps=logging_config.logging_steps,
        log_level=logging_config.log_level,
        report_to=logging_config.report_to,
        
        # Misc
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        full_determinism=False,
        seed=42,
        
        # Hub settings
        push_to_hub=False,
        hub_token=None,
    )

def get_early_stopping_callback(config: OptimizationConfig) -> EarlyStoppingCallback:
    """Get early stopping callback with configured patience."""
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