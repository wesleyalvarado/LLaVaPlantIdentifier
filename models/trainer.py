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
from torch.amp import autocast, GradScaler  # Changed from torch.cuda.amp
from pathlib import Path

logger = logging.getLogger(__name__)

class CustomTrainer:
    """Custom trainer implementation for LLaVA model."""
    
    def __init__(
        self,
        model: nn.Module,
        args: Any,
        train_dataset=None,
        eval_dataset=None,
        class_weights: Optional[torch.Tensor] = None,
        callbacks: Optional[List] = None
    ):
        """Initialize trainer.
        
        Args:
            model: The model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            class_weights: Optional tensor of class weights for loss calculation
            callbacks: Optional list of training callbacks
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = next(model.parameters()).device
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.callbacks = callbacks or []
        self.scaler = GradScaler(enabled=self.args.fp16)
        
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
    
    def _validate_gradients(self, model: nn.Module) -> bool:
        """Check gradients for invalid values."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                valid = not bool(torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid:
                    logger.warning(f"Invalid gradients in {name}")
                    return False
        return True
    
    def on_evaluate(self, metrics: Dict[str, float]):
        """Handle evaluation callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_evaluate'):
                callback.on_evaluate(self.args, self.global_step, metrics)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
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

    def compute_loss(self, model_outputs: Any, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss with improved numerical stability."""
        try:
            # Get logits and convert to float32 for stability
            logits = model_outputs.logits.to(torch.float32)
            labels = labels.to(logits.device).long()

            # Get shapes
            batch_size, seq_length, vocab_size = logits.shape
            label_length = labels.size(1)

            # Pad or truncate labels to match logits sequence length
            if seq_length > label_length:
                labels = torch.nn.functional.pad(
                    labels, 
                    (0, seq_length - label_length),
                    value=-100
                )
            else:
                labels = labels[:, :seq_length]

            # Reshape for loss calculation
            logits = logits.reshape(-1, vocab_size)
            labels = labels.reshape(-1)

            # Apply label smoothing and loss scaling
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction='mean',
                label_smoothing=0.1
            )

            # Compute scaled loss
            loss = loss_fct(logits, labels)

            # Add loss scaling factor
            loss = loss / self.args.gradient_accumulation_steps

            # Validate loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss detected!")
                return None

            # Log stats for debugging
            if loss is not None:
                with torch.no_grad():
                    logger.debug(f"Loss computation stats:")
                    logger.debug(f"  Loss value: {loss.item():.4f}")
                    logger.debug(f"  Logits shape: {logits.shape}")
                    logger.debug(f"  Labels shape: {labels.shape}")
                    logger.debug(f"  Logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")

            return loss

        except Exception as e:
            logger.error(f"Error computing loss: {e}")
            logger.error(traceback.format_exc())
            raise

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            eval_pred: Tuple of predictions and labels
            
        Returns:
            Dictionary of metric names and values
        """
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        
        # Calculate accuracy
        valid_mask = labels != -100
        correct = (predictions[valid_mask] == labels[valid_mask]).sum()
        total = valid_mask.sum()
        accuracy = correct.float() / total
        
        # Calculate perplexity
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        perplexity = torch.exp(loss)
        
        return {
            "accuracy": accuracy.item(),
            "perplexity": perplexity.item(),
            "loss": loss.item()
        }

    def training_step(self, model: nn.Module, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Perform a training step with comprehensive error handling and stability checks."""
        try:
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Set model to training mode
            model.train()
            
            # Process pixel values
            pixel_values = inputs['pixel_values'].to(self.device)
            logger.debug(f"Initial pixel_values shape: {pixel_values.shape}")
            
            # Convert to required shape [B, num_patches, C, H, W]
            if len(pixel_values.shape) == 3:  # [C, H, W]
                pixel_values = pixel_values.unsqueeze(0)  # Add batch dim
            
            if len(pixel_values.shape) == 4:  # [B, C, H, W]
                patch_size = model.config.vision_config.patch_size
                H, W = pixel_values.shape[2:]
                h_patches = H // patch_size
                w_patches = W // patch_size
                num_patches = (h_patches * w_patches) + 1  # Add 1 for CLS token
                logger.debug(f"Grid size: {h_patches}x{w_patches} = {num_patches-1} patches (+1 CLS token)")
                
                # Reshape to [B, num_patches, C, patch_size, patch_size]
                B, C = pixel_values.shape[:2]
                pixel_values = pixel_values.unsqueeze(1).expand(-1, num_patches, -1, -1, -1)
                logger.debug(f"Reshaped pixel_values: {pixel_values.shape}")

            # Get other inputs and ensure they have batch dimension
            input_ids = inputs['input_ids'].to(self.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                
            attention_mask = inputs['attention_mask'].to(self.device)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
                
            labels = inputs['labels'].to(self.device)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)

            # Validate inputs
            if torch.isnan(pixel_values).any():
                logger.error("NaN detected in pixel_values")
                return None
                
            if torch.isinf(pixel_values).any():
                logger.error("Inf detected in pixel_values")
                return None

            # Create model inputs
            model_inputs = {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'image_sizes': torch.tensor([[336, 336]], device=self.device, dtype=torch.long),
                'return_dict': True
            }

            # Log shapes for debugging
            logger.debug("Input shapes:")
            for key, value in model_inputs.items():
                if isinstance(value, torch.Tensor):
                    logger.debug(f"  {key}: {value.shape}")

            # Forward pass
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device_type, enabled=self.args.fp16):
                outputs = model(**model_inputs)
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    loss = self.compute_loss(outputs, model_inputs['labels'])
                    
                # Validate loss before proceeding
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Invalid loss detected: {loss}")
                    return None
                    
                logger.debug(f"Loss value: {loss.item()}")
                    
                # Scale loss for gradient accumulation
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps            

            # Backward pass
            self.scaler.scale(loss).backward()
            
            if self.args.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.args.max_grad_norm
                )

            # Optimizer step with gradient scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            return loss.detach()

        except Exception as e:
            logger.error(f"Error in training step: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _save_checkpoint(self, epoch: int, step: int, save_dir: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_eval_loss': self.best_eval_loss
        }
        
        checkpoint_path = Path(save_dir) / f'checkpoint-{step}'
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(checkpoint, checkpoint_path / 'training_state.pt')
        self.model.save_pretrained(checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path / 'training_state.pt')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint['step']
        self.best_eval_loss = checkpoint['best_eval_loss']
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation and return metrics."""
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader()
        
        total_eval_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs in eval_dataloader:
                # Process pixel values with same reshaping as training
                pixel_values = inputs['pixel_values'].to(self.device)
                if len(pixel_values.shape) == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                
                # Apply patch reshaping
                if len(pixel_values.shape) == 4:
                    patch_size = self.model.config.vision_config.patch_size
                    H, W = pixel_values.shape[2:]
                    h_patches = H // patch_size
                    w_patches = W // patch_size
                    num_patches = (h_patches * w_patches) + 1
                    B, C = pixel_values.shape[:2]
                    pixel_values = pixel_values.unsqueeze(1).expand(-1, num_patches, -1, -1, -1)
                
                # Create model inputs
                model_inputs = {
                    'pixel_values': pixel_values,
                    'input_ids': inputs['input_ids'].to(self.device),
                    'attention_mask': inputs['attention_mask'].to(self.device),
                    'labels': inputs['labels'].to(self.device),
                    'image_sizes': torch.tensor([[336, 336]], device=self.device),
                    'vision_feature_layer': -2,
                    'vision_feature_select_strategy': 'default'
                }
                
                # Forward pass
                outputs = self.model(**model_inputs)
                loss = self.compute_loss(outputs, model_inputs['labels'])
                
                total_eval_loss += loss.item()
                all_predictions.append(outputs.logits.detach().cpu())
                all_labels.append(inputs['labels'].cpu())
        
        # Compute metrics
        metrics = {
            'eval_loss': total_eval_loss / len(eval_dataloader)
        }
        
        return metrics

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop with comprehensive error handling and stability monitoring."""
        try:
            # Load checkpoint if specified
            if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
                self._load_checkpoint(resume_from_checkpoint)
                        
            train_dataloader = self.get_train_dataloader()
            self.model.zero_grad(set_to_none=True)
                    
            num_epochs = int(self.args.num_train_epochs)
            nan_count = 0  # Track consecutive NaN/error occurrences
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 3  # Number of evaluations without improvement before stopping
            
            for epoch in range(num_epochs):
                logger.info(f"Starting epoch {epoch}/{num_epochs}")
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
                epoch_loss = 0.0
                valid_steps = 0
                        
                for step, inputs in enumerate(progress_bar):
                    # Clear memory before each step
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                            
                    # Perform training step
                    loss = self.training_step(self.model, inputs)
                    
                    # Handle invalid loss
                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        nan_count += 1
                        logger.warning(f"Invalid loss detected (count: {nan_count})")
                        
                        if nan_count >= 3:  # Three strikes rule
                            logger.error("Too many invalid losses detected. Stopping training.")
                            return
                        continue
                    
                    # Valid loss obtained
                    nan_count = 0  # Reset counter
                    epoch_loss += loss.item()
                    valid_steps += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{(epoch_loss/valid_steps):.4f}"
                    })
                            
                    # Gradient accumulation and optimization
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        # Optimizer step
                        self.optimizer.step()
                        self.scheduler.step()
                        self.model.zero_grad(set_to_none=True)
                        self.global_step += 1
                            
                        # Save checkpoint
                        if self.global_step % self.args.save_steps == 0:
                            self._save_checkpoint(
                                epoch,
                                self.global_step,
                                self.args.output_dir
                            )
                            
                        # Evaluation
                        if self.global_step % self.args.eval_steps == 0:
                            metrics = self.evaluate()
                            logger.info(f"Evaluation metrics: {metrics}")
                            self.on_evaluate(metrics)
                            
                            # Early stopping check
                            current_loss = metrics.get('eval_loss', float('inf'))
                            if current_loss < best_loss:
                                best_loss = current_loss
                                patience_counter = 0
                            else:
                                patience_counter += 1
                                if patience_counter >= max_patience:
                                    logger.info("Early stopping triggered - no improvement in evaluation loss")
                                    return
                            
                        # Check max steps
                        if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                            progress_bar.close()
                            logger.info("Reached maximum steps, stopping training")
                            return
                
                # End of epoch logging
                if valid_steps > 0:
                    avg_epoch_loss = epoch_loss / valid_steps
                    logger.info(f"Epoch {epoch} completed with average loss: {avg_epoch_loss:.4f}")
                
                # End of epoch cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                    
            # Final cleanup and model saving
            self._save_checkpoint(
                num_epochs - 1,
                self.global_step,
                self.args.output_dir
            )
            logger.info("Training completed successfully")
                    
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Get evaluation dataloader.
        
        Args:
            eval_dataset: Optional evaluation dataset
        """
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        if dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
            
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    def save_model(self, output_dir: str):
        """Save the model.
        
        Args:
            output_dir: Directory to save model
        """
        try:
            # Clear memory before saving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Save the model
            self.model.save_pretrained(
                output_dir,
                safe_serialization=True,
                max_shard_size="500MB"
            )
            logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    
def ensure_2d(tensor):
    """Ensure tensor is at least 2D (add batch dimension if necessary)."""
    if len(tensor.shape) == 1:  # If the tensor is 1D
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    return tensor

