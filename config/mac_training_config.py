# config/mac_training_config.py

import os
import torch
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
from transformers import EarlyStoppingCallback

@dataclass
class ModelConfig:
    """Model configuration optimized for M2 Mac."""
    # Model selection - consider using smaller models for Mac
    name: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    image_size: int = 336  # CLIP's expected size
    patch_size: int = 14   # CLIP patch size
    
    # Device and precision settings
    dtype: str = "float16"  # float16 works better on MPS
    device_map: str = "auto"
    trust_remote_code: bool = True
    
    # Memory optimization
    low_cpu_mem_usage: bool = True
    gradient_checkpointing: bool = True
    vision_feature_layer: int = -2
    
    # Checkpoint settings
    checkpoint_dir: str = "checkpoints"
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 2  # Keep fewer checkpoints to save space
    
    # Memory limits - adjust based on your Mac's capabilities
    max_memory: Dict[str, str] = field(
        default_factory=lambda: {
            "mps": "8GB",     # Adjust based on your Mac's memory
            "cpu": "16GB"    # Adjust based on your RAM
        }
    )

@dataclass
class OptimizationConfig:
    """Training optimization settings for M2 Mac."""
    # Learning rate settings
    learning_rate: float = 5e-5        # Slightly higher for slower convergence
    weight_decay: float = 0.01
    warmup_ratio: float = 0.2          # Longer warmup for stability
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Gradient settings
    max_grad_norm: float = 0.5
    gradient_accumulation_steps: int = 8  # Reduced for M2
    
    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means train for full num_epochs
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    # Scheduler settings
    scheduler_type: str = "cosine"      # Cosine with warmup
    lr_scheduler_warmup_steps: int = 100
    
    # Disable quantization for Mac
    use_8bit_quantization: bool = False
    use_4bit_quantization: bool = False
    
    # Loss scaling
    loss_scaling_factor: float = 1.0

@dataclass
class DataConfig:
    """Data processing configuration for M2 Mac."""
    # Batch sizes - keep small for Mac
    train_batch_size: int = 1
    eval_batch_size: int = 1
    
    # Sequence length
    max_length: int = 512
    
    # DataLoader settings
    num_workers: int = 2               # Reduced for Mac
    pin_memory: bool = True
    prefetch_factor: int = 2           # Reduced prefetching
    
    # Dataset settings
    sample_fraction: float = 0.1       # Start with small fraction
    dataset_name: str = "dpdl-benchmark/oxford_flowers102"
    image_column: str = "image"
    caption_column: str = "label"
    
    # Caching
    cache_dir: Optional[str] = "dataset_cache"
    preprocessing_num_workers: Optional[int] = 2  # Reduced for Mac
    
    # Data augmentation
    augmentation_enabled: bool = True   # Enable basic augmentations
    
    # Sample limits
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    # Logging settings
    logging_dir: str = "logs"
    logging_strategy: str = "steps"
    logging_steps: int = 10
    log_level: str = "info"
    
    # Saving settings
    save_strategy: str = "steps"
    save_total_limit: int = 2          # Keep fewer checkpoints
    save_safetensors: bool = True
    
    # Monitoring
    tensorboard_logging: bool = True
    wandb_logging: bool = False        # Disable wandb by default
    wandb_project: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

def get_training_args(
    model_dir: str,
    model_config: ModelConfig,
    optim_config: OptimizationConfig,
    data_config: DataConfig,
    logging_config: Optional[LoggingConfig] = None
) -> TrainingArguments:
    """Get training arguments optimized for M2 Mac."""
    
    if logging_config is None:
        logging_config = LoggingConfig()
    
    # Ensure directories exist
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
        
        # Mac-specific settings
        no_cuda=True,           # Disable CUDA
        use_mps_device=True,    # Enable MPS
        fp16=False,             # Disable fp16 (not supported on MPS)
        bf16=False,             # Disable bf16
        
        # Scheduler settings
        lr_scheduler_type=optim_config.scheduler_type,
        warmup_steps=optim_config.lr_scheduler_warmup_steps,
        
        # Evaluation settings
        evaluation_strategy="steps",
        eval_steps=model_config.eval_steps,
        save_strategy="steps",
        save_steps=model_config.save_steps,
        save_total_limit=model_config.save_total_limit,
        
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
    """Get model initialization kwargs for Mac."""
    return {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": getattr(torch, config.dtype),
        "device_map": config.device_map,
        "low_cpu_mem_usage": config.low_cpu_mem_usage
    }