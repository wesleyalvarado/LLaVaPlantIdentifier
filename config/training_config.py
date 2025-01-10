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
    dtype: str = "float16"
    trust_remote_code: bool = True  # Added trust_remote_code attribute

def get_training_args(model_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        max_steps=10,
        save_steps=5,
        push_to_hub=False,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        fp16=True,  # Enable half-precision training
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        torch_compile=False,
        disable_tqdm=True,
    )