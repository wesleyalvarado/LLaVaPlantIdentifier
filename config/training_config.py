# config/training_config.py
import os
import torch
from transformers import TrainingArguments
from dataclasses import dataclass
from typing import Optional

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

def get_training_args(model_dir: str) -> TrainingArguments:
    """Get training arguments with memory-efficient settings."""
    return TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        max_steps=10,
        save_strategy="steps",
        save_steps=5,
        evaluation_strategy="steps",
        eval_steps=5,
        save_total_limit=2,
        push_to_hub=False,
        learning_rate=1e-5,
        num_train_epochs=3,
        warmup_ratio=0.1,
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        torch_compile=False,
        disable_tqdm=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )