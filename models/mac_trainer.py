# models/trainer.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import logging
import traceback
import gc
import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.amp import autocast
from pathlib import Path

logger = logging.getLogger(__name__)

class CustomTrainer:
    """Custom trainer implementation optimized for Mac/MPS."""
    
    def __init__(
        self,
        model: nn.Module,
        args: Any,
        train_dataset=None,
        eval_dataset=None,
        class_weights: Optional[torch.Tensor] = None,
        callbacks: Optional[List] = None
    ):
        """Initialize trainer with Mac-specific settings."""
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Device handling for Mac
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS device")
        else:
            self.device = torch.device("cpu")
            logger.info("MPS not available, using CPU")
        
        # Move class weights to correct device if provided
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.callbacks = callbacks or []
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Enable memory optimizations
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            
        # Log initial configuration
        self._log_model_config()

    def _cleanup_memory(self):
        """Mac-specific memory cleanup."""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            # Synchronize MPS stream if available
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

    def training_step(self, model: nn.Module, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Perform a training step."""
        try:
            # Set model to training mode
            model.train()
            
            # Process inputs with correct shapes
            input_ids = inputs['input_ids'].unsqueeze(0) if len(inputs['input_ids'].shape) == 1 else inputs['input_ids']
            attention_mask = inputs['attention_mask'].unsqueeze(0) if len(inputs['attention_mask'].shape) == 1 else inputs['attention_mask']
            labels = inputs['labels'].unsqueeze(0) if len(inputs['labels'].shape) == 1 else inputs['labels']
            
            # Fix pixel_values shape: should be [batch_size, num_images=1, channels=3, height, width]
            pixel_values = inputs['pixel_values']
            if len(pixel_values.shape) == 3:  # [C, H, W]
                pixel_values = pixel_values.unsqueeze(0)  # [1, C, H, W]
            if len(pixel_values.shape) == 4:  # [B or N, C, H, W]
                if pixel_values.shape[0] != 1:
                    # If first dimension is not batch size, it's probably num_images
                    pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension
                else:
                    # If first dimension is batch size, add num_images dimension
                    pixel_values = pixel_values.unsqueeze(1)  # [B, 1, C, H, W]
            
            # Move to device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            pixel_values = pixel_values.to(self.device)
            
            # Debug tensor shapes
            logger.info(f"input_ids shape: {input_ids.shape}")
            logger.info(f"attention_mask shape: {attention_mask.shape}")
            logger.info(f"labels shape: {labels.shape}")
            logger.info(f"pixel_values shape: {pixel_values.shape}")
            
            # Create model inputs
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'pixel_values': pixel_values,
                'return_dict': True,
                'image_sizes': torch.tensor([[336, 336]], device=self.device)  # Add this
            }

            # Forward pass
            outputs = model(**model_inputs)
            loss = outputs.loss

            # Scale loss for gradient accumulation
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            return loss.detach()

        except Exception as e:
            logger.error(f"Error in training step: {e}")
            logger.error(traceback.format_exc())
            return None

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation with Mac-specific optimizations."""
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader()
        
        total_eval_loss = 0
        num_eval_steps = 0
        
        with torch.no_grad():
            for inputs in eval_dataloader:
                try:
                    # Move inputs to device
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = self.model(**inputs)
                    loss = outputs.loss if hasattr(outputs, 'loss') else self.compute_loss(outputs, inputs['labels'])
                    
                    if loss is not None:
                        total_eval_loss += loss.item()
                        num_eval_steps += 1
                        
                    # Memory cleanup after each batch
                    self._cleanup_memory()
                    
                except Exception as e:
                    logger.error(f"Error in evaluation step: {e}")
                    continue
        
        # Compute average loss
        avg_loss = total_eval_loss / num_eval_steps if num_eval_steps > 0 else float('inf')
        
        return {'eval_loss': avg_loss}

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop with Mac-specific optimizations."""
        try:
            if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
                self._load_checkpoint(resume_from_checkpoint)
            
            train_dataloader = self.get_train_dataloader()
            num_epochs = int(self.args.num_train_epochs)
            
            for epoch in range(num_epochs):
                logger.info(f"Starting epoch {epoch}/{num_epochs}")
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
                epoch_loss = 0.0
                valid_steps = 0
                
                for step, inputs in enumerate(progress_bar):
                    # Training step
                    loss = self.training_step(self.model, inputs)
                    
                    if loss is not None:
                        epoch_loss += loss.item()
                        valid_steps += 1
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "avg_loss": f"{(epoch_loss/valid_steps):.4f}"
                        })
                    
                    # Evaluation and checkpointing
                    if self.global_step % self.args.eval_steps == 0:
                        metrics = self.evaluate()
                        logger.info(f"Evaluation metrics: {metrics}")
                        
                        # Save checkpoint if best loss
                        current_loss = metrics.get('eval_loss', float('inf'))
                        if current_loss < self.best_eval_loss:
                            self.best_eval_loss = current_loss
                            self._save_checkpoint(epoch, self.global_step, self.args.output_dir)
                    
                    self.global_step += 1
                    
                    # Memory cleanup
                    self._cleanup_memory()
                    
                    # Check max steps
                    if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                        return
                
                # End of epoch cleanup
                self._cleanup_memory()
                
            # Final save
            self._save_checkpoint(num_epochs - 1, self.global_step, self.args.output_dir)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader with Mac-optimized settings."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=2,  # Reduced for Mac
            pin_memory=True
        )
    
    def _cleanup_memory(self):
        """Clean up memory explicitly."""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @property
    def image_size(self):
        """Get image size from model config."""
        if hasattr(self.model.config, 'vision_config'):
            return self.model.config.vision_config.image_size
        return 336  # Default size

    def get_eval_dataloader(self) -> DataLoader:
        """Get evaluation dataloader with Mac-optimized settings."""
        return DataLoader(
            self.eval_dataset or self.train_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=2,  # Reduced for Mac
            pin_memory=True
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with proper parameter grouping."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                        if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                        if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Create learning rate scheduler."""
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=100,
                num_training_steps=1000
            )
        
        num_training_steps = len(self.train_dataset) * self.args.num_train_epochs
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def _log_model_config(self):
        """Log important model configuration parameters."""
        if hasattr(self.model, 'config'):
            logger.info("Model Configuration:")
            if hasattr(self.model.config, 'image_grid_pinpoints'):
                logger.info(f"  image_grid_pinpoints: {self.model.config.image_grid_pinpoints}")
            if hasattr(self.model.config, 'vision_config'):
                logger.info(f"  vision_config.image_size: {self.model.config.vision_config.image_size}")
                if hasattr(self.model.config.vision_config, 'patch_size'):
                    logger.info(f"  vision_config.patch_size: {self.model.config.vision_config.patch_size}")
            if hasattr(self.model.config, 'image_token_index'):
                logger.info(f"  image_token_index: {self.model.config.image_token_index}")

    def save_model(self, output_dir: str):
        """Save model with Mac-specific handling."""
        try:
            # Clear memory before saving
            self._cleanup_memory()
            
            # Move model to CPU for saving
            model_to_save = self.model.cpu()
            model_to_save.save_pretrained(
                output_dir,
                safe_serialization=True
            )
            
            # Move model back to MPS/device
            self.model = self.model.to(self.device)
            
            logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise