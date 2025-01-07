# config/training_config.py
import os
import torch
from transformers import TrainingArguments
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    image_size: int = 224
    dtype: str = "float32"
    device_map: str = "auto"

def get_training_args(model_dir: str) -> TrainingArguments:
    # Determine the appropriate device
    if torch.cuda.is_available():
        print("CUDA device is available. Using CUDA.")
    elif torch.backends.mps.is_available():
        print("MPS device is available. Using MPS.")
    else:
        print("No GPU available. Falling back to CPU.")

    return TrainingArguments(
        output_dir=model_dir,
        
        # Memory usage reduction
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        
        # Memory and compatibility optimizations
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        
        # Reduced training scope for testing
        max_steps=10,
        save_steps=5,
        eval_steps=5,
        logging_steps=1,
        
        # Disable memory-intensive features
        report_to="none",
        push_to_hub=False,
        
        # Conservative learning rate
        learning_rate=1e-5,
        warmup_ratio=0.1,
        
        # Precision settings
        fp16=False,
        bf16=False,
        
        # Further memory optimizations
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        
        optim="adamw_torch",
        torch_compile=False,

        # Reduce output verbosity
        disable_tqdm=True,
    )