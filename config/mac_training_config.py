# config/mac_training_config.py

import os
import torch
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
from transformers import EarlyStoppingCallback

@dataclass
class ModelConfig:
    """Model configuration optimized for Mac."""
    name: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    image_size: int = 336  # Match CLIP's expected size
    patch_size: int = 14   # CLIP patch size
    dtype: str = "float16"  # Stick to float16 for Mac
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    gradient_checkpointing: bool = True
    
    # Checkpoint settings
    checkpoint_dir: str = "checkpoints"
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 2  # Keep fewer checkpoints to save memory
    
    # Memory management
    max_memory: Dict[str, str] = field(
        default_factory=lambda: {
            "cpu": "16GB",  # Adjust based on your Mac's RAM
        }
    )

@dataclass
class OptimizationConfig:
    """Training optimization settings for Mac."""
    # Learning rate settings
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means train for full num_epochs
    
    # Gradient settings - reduced for Mac
    max_grad_norm: float = 0.5
    gradient_accumulation_steps: int = 8
    
    # Scheduler settings
    scheduler_type: str = "cosine"
    lr_scheduler_warmup_steps: int = 100
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    # Disable quantization for Mac
    use_8bit_quantization: bool = False
    use_4bit_quantization: bool = False

@dataclass
class DataConfig:
    """Data processing configuration for Mac."""
    # Batch sizes - keep small for Mac
    train_batch_size: int = 1
    eval_batch_size: int = 1
    
    # Sequence length
    max_length: int = 512
    
    # DataLoader settings - reduced for Mac
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Dataset settings
    sample_fraction: float = 0.1
    dataset_name: str = "dpdl-benchmark/oxford_flowers102"
    image_column: str = "image"
    caption_column: str = "label"
    cache_dir: Optional[str] = "dataset_cache"
    preprocessing_num_workers: Optional[int] = 2
    
    # Augmentation
    augmentation_enabled: bool = True
    
    # Sample limits - optional
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    logging_dir: str = "logs"
    logging_strategy: str = "steps"
    logging_steps: int = 10
    log_level: str = "info"
    
    # Saving settings
    save_strategy: str = "steps"
    save_total_limit: int = 2
    save_safetensors: bool = True
    
    # Monitoring
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    wandb_project: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

def get_device():
    """Get appropriate device for training."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_training_args(
    model_dir: str,
    model_config: ModelConfig,
    optim_config: OptimizationConfig,
    data_config: DataConfig,
    logging_config: Optional[LoggingConfig] = None
) -> TrainingArguments:
    """Get training arguments optimized for Mac."""
    
    if logging_config is None:
        logging_config = LoggingConfig()
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    if model_config.checkpoint_dir:
        os.makedirs(os.path.join(model_dir, model_config.checkpoint_dir), exist_ok=True)
    
    # Check device availability
    use_cpu = not torch.backends.mps.is_available()
    
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
        
        # Device settings
        use_cpu=use_cpu,
        
        # Precision settings
        fp16=False,  # Disable fp16
        bf16=False,  # Disable bf16
        
        # Scheduler settings
        lr_scheduler_type=optim_config.scheduler_type,
        warmup_steps=optim_config.lr_scheduler_warmup_steps,
        
        # Memory optimization
        gradient_checkpointing=model_config.gradient_checkpointing,
        optim="adamw_torch",
        adam_beta1=optim_config.adam_beta1,
        adam_beta2=optim_config.adam_beta2,
        adam_epsilon=optim_config.adam_epsilon,
        
        # DataLoader settings
        dataloader_num_workers=data_config.num_workers,
        dataloader_pin_memory=data_config.pin_memory,
        
        # Evaluation settings
        evaluation_strategy="steps",
        eval_steps=model_config.eval_steps,
        
        # Saving settings
        save_strategy="steps",
        save_steps=model_config.save_steps,
        save_total_limit=model_config.save_total_limit,
        
        # Logging settings
        logging_dir=os.path.join(model_dir, logging_config.logging_dir),
        logging_strategy=logging_config.logging_strategy,
        logging_steps=logging_config.logging_steps,
        log_level=logging_config.log_level,
        report_to=logging_config.report_to,
        
        # Other settings
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
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
        "low_cpu_mem_usage": config.low_cpu_mem_usage
    }