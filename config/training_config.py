# config/training_config.py
import os
from transformers import TrainingArguments
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    image_size: int = 224
    dtype: str = "float32"  # Changed from float16
    device_map: str = "cpu"  # Fallback to CPU
    trust_remote_code: bool = True

def get_training_args(model_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=model_dir,
        
        # Further reduce memory usage
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Accumulate gradients instead of processing larger batches
        
        # Memory optimizations
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        
        # Reduce memory pressure
        max_steps=10,  # Limit training steps for testing
        save_steps=5,
        eval_steps=5,
        logging_steps=1,
        
        # Disable features that consume memory
        report_to="none",
        push_to_hub=False,
        
        # Learning rate settings
        learning_rate=5e-6,  # Lower learning rate
        warmup_ratio=0.05,
        
        # Memory optimizations
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,  # Set pin memory here
        
        # Add these memory-saving options
        deepspeed=None,
        optim="adamw_torch",
        torch_compile=False,  # Disable torch compilation


        use_mps_device=True  # Enable MPS for M2 Mac
    )