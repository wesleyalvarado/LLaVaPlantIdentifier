# config/training_config.py
import os
from transformers import TrainingArguments
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    image_size: int = 336
    dtype: str = "float16"
    device_map: str = "cpu"
    trust_remote_code: bool = True

def get_training_args(model_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1024,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=os.path.join(model_dir, "logs"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        dataloader_num_workers=0
    )