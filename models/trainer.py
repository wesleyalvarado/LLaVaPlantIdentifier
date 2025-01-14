# models/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Union, Any
import logging
import traceback
import gc

from utils.image_utils import prepare_image_inputs, validate_image_tensors

logger = logging.getLogger(__name__)

class CustomTrainer:
    def __init__(self, model, args, train_dataset=None, eval_dataset=None):
        """Initialize the trainer.
        
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
            
        # Log model configuration
        self._log_model_config()
        
    def _log_model_config(self):
        """Log important model configuration parameters"""
        if hasattr(self.model, 'config'):
            logger.info("Model Configuration:")
            if hasattr(self.model.config, 'image_grid_pinpoints'):
                logger.info(f"  image_grid_pinpoints: {self.model.config.image_grid_pinpoints}")
            if hasattr(self.model.config, 'vision_config'):
                if hasattr(self.model.config.vision_config, 'image_size'):
                    logger.info(f"  vision_config.image_size: {self.model.config.vision_config.image_size}")
            if hasattr(self.model.config, 'image_token_index'):
                logger.info(f"  image_token_index: {self.model.config.image_token_index}")

    def train(self):
        """Main training loop"""
        try:
            train_dataloader = self.get_train_dataloader()
            self.model.zero_grad()
            
            num_epochs = int(self.args.num_train_epochs)
            for epoch in range(num_epochs):
                for step, inputs in enumerate(train_dataloader):
                    # Clear memory before each step
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    if inputs is None:
                        continue
                    
                    inputs = {k: v for k, v in inputs.items()
                             if k in ['pixel_values', 'input_ids', 'attention_mask', 'labels']}
                    
                    loss = self.training_step(self.model, inputs)
                    
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.args.max_grad_norm
                        )
                        self.model.zero_grad()
                    
                    if step % 10 == 0:
                        logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                        
                    if self.args.max_steps > 0 and step >= self.args.max_steps:
                        break
                        
                if self.args.max_steps > 0 and step >= self.args.max_steps:
                    break
                    
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
            """Perform a single training step."""
            try:
                # Clear memory before processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Set model to training mode
                model.train()
                
                # Move inputs to device and handle image processing
                model_inputs = {}
                
                if 'pixel_values' in inputs:
                    pixel_values = inputs['pixel_values'].to(self.device)
                    logger.info(f"Original pixel_values shape: {pixel_values.shape}")
                    
                    # Process image inputs
                    processed_pixels, image_sizes = prepare_image_inputs(
                        pixel_values,
                        model,
                        self.device
                    )
                    
                    # Validate tensors
                    if not validate_image_tensors(processed_pixels, image_sizes):
                        raise ValueError("Invalid image tensor shapes or types")
                        
                    model_inputs['pixel_values'] = processed_pixels
                    model_inputs['image_sizes'] = image_sizes
                
                # Add other inputs (input_ids, attention_mask, labels)
                for k in ['input_ids', 'attention_mask', 'labels']:
                    if k in inputs:
                        tensor = inputs[k].to(self.device)
                        if len(tensor.shape) == 1:
                            tensor = tensor.unsqueeze(0)  # Add batch dimension if needed
                        model_inputs[k] = tensor
                
                # Debug input shapes
                logger.info("Model inputs:")
                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"  {k}: shape {v.shape}, dtype {v.dtype}, device {v.device}")
                        if k == 'input_ids':
                            # Log if image token is present
                            image_token_index = model.config.image_token_index
                            image_tokens = (v == image_token_index).sum().item()
                            logger.info(f"  Number of image tokens found: {image_tokens}")
                    else:
                        logger.info(f"  {k}: {v}")

                # Add required configuration parameters
                model_inputs.update({
                    'vision_feature_layer': -2,
                    'vision_feature_select_strategy': 'default',
                    'return_dict': True
                })

                # Forward pass with gradient checkpointing if enabled
                with torch.amp.autocast('cuda', enabled=self.args.fp16):
                    outputs = model(**model_inputs)
                    loss = outputs.loss
                    
                logger.info(f"Loss value: {loss.item():.4f}")
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()
                return loss.detach()
                
            except Exception as e:
                logger.error(f"Error in training step: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    def get_train_dataloader(self) -> DataLoader:
        """Get the training dataloader.
        
        Returns:
            DataLoader: The training dataloader
        """
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
        """Get the evaluation dataloader.
        
        Args:
            eval_dataset: Optional evaluation dataset to use instead of self.eval_dataset
            
        Returns:
            DataLoader: The evaluation dataloader
        """
        dataset_to_use = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return DataLoader(
            dataset_to_use,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    def save_model(self, output_dir):
        """Save the model.
        
        Args:
            output_dir: Directory to save the model to
        """
        try:
            # Clear memory before saving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            self.model.save_pretrained(
                output_dir,
                safe_serialization=True,
                max_shard_size="500MB"  # Shard the model for memory efficiency
            )
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise