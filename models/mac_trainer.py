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
        """Perform a training step with Mac-specific optimizations."""
        try:
            # Memory cleanup before step
            self._cleanup_memory()
            
            # Set model to training mode
            model.train()
            
            # Process inputs
            pixel_values = inputs['pixel_values'].to(self.device)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            labels = inputs['labels'].to(self.device)
            
            # Create model inputs
            model_inputs = {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'image_sizes': torch.tensor([[336, 336]], device=self.device),
                'return_dict': True
            }

            # Forward pass - no autocast for MPS
            outputs = model(**model_inputs)
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = self.compute_loss(outputs, model_inputs['labels'])
                
            # Validate loss
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss detected: {loss}")
                return None
                
            # Scale loss for gradient accumulation
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps            

            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.args.max_grad_norm,
                    error_if_nonfinite=True
                )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            # Memory cleanup after step
            self._cleanup_memory()

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

    def get_eval_dataloader(self) -> DataLoader:
        """Get evaluation dataloader with Mac-optimized settings."""
        return DataLoader(
            self.eval_dataset or self.train_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=2,  # Reduced for Mac
            pin_memory=True
        )

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