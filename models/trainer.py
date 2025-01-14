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
from torch.cuda.amp import autocast, GradScaler
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
        class_weights: Optional[torch.Tensor] = None
    ):
        """Initialize trainer.
        
        Args:
            model: The model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            class_weights: Optional tensor of class weights for loss calculation
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = next(model.parameters()).device
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.early_stopping_count = 0
        
        # Enable memory optimizations
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            
        # Log initial configuration
        self._log_model_config()

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
        """Compute loss with optional class weighting."""
        try:
            # Get logits
            logits = model_outputs.logits  # [batch_size, sequence_length, vocab_size]
            labels = labels.to(logits.device)

            # Log original shapes
            logger.debug(f"Original shapes:")
            logger.debug(f"  Logits: {logits.shape}")
            logger.debug(f"  Labels: {labels.shape}")

            # Pad labels to match logits sequence length
            if logits.size(1) > labels.size(1):
                pad_length = logits.size(1) - labels.size(1)
                # Pad with -100 to ignore these positions in loss calculation
                labels = torch.nn.functional.pad(
                    labels, 
                    (0, pad_length), 
                    value=-100
                )
            
            # Reshape both tensors
            batch_size = logits.size(0)
            seq_length = logits.size(1)
            num_classes = logits.size(2)
            
            logits = logits.reshape(-1, num_classes)  # [batch_size * seq_length, vocab_size]
            labels = labels.reshape(-1)  # [batch_size * seq_length]

            # Create loss function
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(
                    weight=self.class_weights,
                    ignore_index=-100,
                    reduction='mean'
                )
            else:
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=-100,
                    reduction='mean'
                )

            # Log reshaped dimensions
            logger.debug(f"Reshaped dimensions:")
            logger.debug(f"  Logits: {logits.shape}")
            logger.debug(f"  Labels: {labels.shape}")

            # Compute loss
            loss = loss_fct(logits, labels)
            
            return loss

        except Exception as e:
            logger.error(f"Error computing loss: {e}")
            logger.error(f"Logits shape: {logits.shape if 'logits' in locals() else 'N/A'}")
            logger.error(f"Labels shape: {labels.shape if 'labels' in locals() else 'N/A'}")
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

    def training_step(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """Perform a single training step.
        
        Args:
            model: The model to train
            inputs: Input dictionary containing tensors
                
        Returns:
            Loss tensor
        """
        try:
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Set model to training mode
            model.train()
            
            # Log input shapes
            logger.debug("Original input shapes:")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    logger.debug(f"  {key}: shape {value.shape}")
            
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
            
            # Process model inputs
            model_inputs = {}
            for k in ['input_ids', 'attention_mask', 'labels']:
                if k in inputs:
                    tensor = inputs[k].clone().detach().to(self.device)
                    if len(tensor.shape) == 1:
                        tensor = tensor.unsqueeze(0)
                    model_inputs[k] = tensor
            
            # Add processed image inputs
            model_inputs.update({
                'pixel_values': pixel_values,
                'image_sizes': torch.tensor([[336, 336]], device=self.device),
                'vision_feature_layer': -2,
                'vision_feature_select_strategy': 'default'
            })
            
            # Log final inputs
            logger.debug("Final model inputs:")
            for key, value in model_inputs.items():
                if isinstance(value, torch.Tensor):
                    logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.debug(f"  {key}: {value}")
                    
                if key == 'input_ids':
                    num_tokens = (value == model.config.image_token_index).sum().item()
                    logger.debug(f"  Number of image tokens: {num_tokens}")
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(**model_inputs)
                
                # Log shapes before loss computation
                logger.debug(f"Model output logits shape: {outputs.logits.shape}")
                logger.debug(f"Labels shape: {model_inputs['labels'].shape}")
                
                loss = self.compute_loss(outputs, model_inputs['labels'])
            
            # Scale loss and backward
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            return loss.detach()
        
                
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            logger.error(traceback.format_exc())
            raise

    def _save_checkpoint(self, epoch: int, step: int, save_dir: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'early_stopping_count': self.early_stopping_count
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
        self.early_stopping_count = checkpoint['early_stopping_count']
        
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
                # Process inputs
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = self.compute_loss(outputs, inputs['labels'])
                
                total_eval_loss += loss.item()
                all_predictions.append(outputs.logits.detach().cpu())
                all_labels.append(inputs['labels'].cpu())
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metrics = self.compute_metrics((all_predictions, all_labels))
        metrics['eval_loss'] = total_eval_loss / len(eval_dataloader)
        
        return metrics

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop.
        
        Args:
            resume_from_checkpoint: Optional checkpoint to resume from
        """
        try:
            # Load checkpoint if specified
            if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
                self._load_checkpoint(resume_from_checkpoint)
                
            train_dataloader = self.get_train_dataloader()
            self.model.zero_grad()
            
            num_epochs = int(self.args.num_train_epochs)
            
            for epoch in range(num_epochs):
                logger.info(f"Starting epoch {epoch}/{num_epochs}")
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
                
                for step, inputs in enumerate(progress_bar):
                    # Clear memory before each step
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Perform training step
                    loss = self.training_step(self.model, inputs)
                    
                    # Update progress bar
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                    
                    # Gradient accumulation and optimization
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        # Unscale gradients
                        self.scaler.unscale_(self.optimizer)
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.args.max_grad_norm
                        )
                        
                        # Optimizer step with gradient scaling
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.model.zero_grad()
                        
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
                        
                        # Early stopping check
                        eval_loss = metrics['eval_loss']
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.early_stopping_count = 0
                            # Save best model
                            self._save_checkpoint(
                                epoch,
                                self.global_step,
                                os.path.join(self.args.output_dir, "best_model")
                            )
                        else:
                            self.early_stopping_count += 1
                            if (self.early_stopping_count >= 
                                self.args.early_stopping_patience):
                                logger.info("Early stopping triggered")
                                return
                    
                    # Check max steps
                    if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                        progress_bar.close()
                        logger.info("Reached maximum steps, stopping training")
                        return
                
                # End of epoch cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
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
