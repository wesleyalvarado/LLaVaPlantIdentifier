# models/trainer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import logging
import traceback
import gc

logger = logging.getLogger(__name__)

class CustomTrainer:
    """Custom trainer implementation for LLaVA model."""
    
    def __init__(self, model, args, train_dataset=None, eval_dataset=None):
        """Initialize trainer.
        
        Args:
            model: The model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = next(model.parameters()).device

        # Enable memory optimizations
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            
        # Log initial configuration
        self._log_model_config()

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
            logger.info("Original input shapes:")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: shape {value.shape}")
            
            # Process pixel values - needs to be [B, num_patches, C, H, W]
            pixel_values = inputs['pixel_values'].to(self.device)
            logger.info(f"Initial pixel_values shape: {pixel_values.shape}")
            
            # Convert to required shape [B, num_patches, C, H, W]
            if len(pixel_values.shape) == 3:  # [C, H, W]
                pixel_values = pixel_values.unsqueeze(0)  # Add batch dim [B, C, H, W]
            
            if len(pixel_values.shape) == 4:  # [B, C, H, W]
                # Calculate number of patches
                patch_size = model.config.vision_config.patch_size
                H, W = pixel_values.shape[2:]
                h_patches = H // patch_size
                w_patches = W // patch_size
                num_patches = (h_patches * w_patches) + 1  # Add 1 for CLS token
                logger.info(f"Grid size: {h_patches}x{w_patches} = {num_patches-1} patches (+1 CLS token)")
                
                # Reshape to [B, num_patches, C, patch_size, patch_size]
                B, C = pixel_values.shape[:2]
                pixel_values = pixel_values.unsqueeze(1).expand(-1, num_patches, -1, -1, -1)
                logger.info(f"Reshaped pixel_values: {pixel_values.shape}")
            
            # Process text inputs
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
                'vision_feature_layer': -2,  # Use second to last layer
                'vision_feature_select_strategy': 'default'  # Use model's default strategy
            })
            
            # Debug final input shapes
            logger.info("Final model inputs:")
            for key, value in model_inputs.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.info(f"  {key}: {value}")
                
                if key == 'input_ids':
                    num_tokens = (value == model.config.image_token_index).sum().item()
                    logger.info(f"  Number of image tokens: {num_tokens}")
            
            # Forward pass
            outputs = model(**model_inputs)
            loss = outputs.loss
            
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            loss.backward()
            return loss.detach()
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def train(self):
        """Main training loop."""
        try:
            train_dataloader = self.get_train_dataloader()
            self.model.zero_grad()
            
            num_epochs = int(self.args.num_train_epochs)
            for epoch in range(num_epochs):
                logger.info(f"Starting epoch {epoch}/{num_epochs}")
                
                for step, inputs in enumerate(train_dataloader):
                    # Clear memory before each step
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Perform training step
                    loss = self.training_step(self.model, inputs)
                    
                    # Gradient accumulation and clipping
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.args.max_grad_norm
                        )
                        self.model.zero_grad()
                    
                    # Log progress
                    if step % 10 == 0:
                        logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                        
                    # Check max steps
                    if self.args.max_steps > 0 and step >= self.args.max_steps:
                        logger.info("Reached max steps, stopping training")
                        break
                        
                if self.args.max_steps > 0 and step >= self.args.max_steps:
                    break
                    
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