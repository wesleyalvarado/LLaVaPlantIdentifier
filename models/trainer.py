# trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Union, Any
import logging
import traceback
import gc

logger = logging.getLogger(__name__)

class CustomTrainer:
    def __init__(self, model, args, train_dataset=None, eval_dataset=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = next(model.parameters()).device

    def train(self):
        """Main training loop"""
        try:
            train_dataloader = self.get_train_dataloader()
            
            self.model.zero_grad()
            
            num_epochs = int(self.args.num_train_epochs)
            for epoch in range(num_epochs):
                for step, inputs in enumerate(train_dataloader):
                    if inputs is None:
                        continue
                    
                    inputs = {
                        k: v for k, v in inputs.items()
                        if k in ['pixel_values', 'input_ids', 'attention_mask', 'labels']  # Adjust keys as needed
                }

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
        try:
            # Set model to training mode
            model.train()
            
            # Filter inputs for model
            model_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
                if k in ['pixel_values', 'input_ids', 'attention_mask', 'labels']
            }
            
            # Handle image processing
            if 'pixel_values' in model_inputs:
                pixel_values = model_inputs['pixel_values']
                
                # CLIP model expects [batch_size, channels, height, width]
                if len(pixel_values.shape) == 3:  # [C, H, W]
                    pixel_values = pixel_values.unsqueeze(0)  # [1, C, H, W]
                
                # Calculate patches
                patch_size = 14  # CLIP's patch size
                height, width = pixel_values.shape[2:]
                num_patches_h = height // patch_size
                num_patches_w = width // patch_size
                num_patches = num_patches_h * num_patches_w
                
                model_inputs['pixel_values'] = pixel_values.contiguous()
                model_inputs['image_sizes'] = [(height, width)]
                
                logger.info(f"Image shape: {pixel_values.shape}")
                logger.info(f"Number of patches (H x W): {num_patches_h} x {num_patches_w} = {num_patches}")
                
                # Debug visualization
                logger.info(f"Patch calculation:")
                logger.info(f"  Height: {height} / {patch_size} = {num_patches_h} patches")
                logger.info(f"  Width: {width} / {patch_size} = {num_patches_w} patches")
                logger.info(f"  Total patches: {num_patches}")
            
            # Add batch dimension to other tensors
            for k in ['input_ids', 'attention_mask', 'labels']:
                if k in model_inputs and len(model_inputs[k].shape) == 1:
                    model_inputs[k] = model_inputs[k].unsqueeze(0)
            
            # Forward pass
            outputs = model(**model_inputs)
            loss = outputs.loss
            
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            loss.backward()
            return loss.detach()
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            logger.error(f"Image shape: {pixel_values.shape if 'pixel_values' in locals() else 'not available'}")
            logger.error(f"Model inputs: {[(k, v.shape) if torch.is_tensor(v) else (k, v) for k, v in model_inputs.items()]}")
            logger.error(traceback.format_exc())
            raise

    def get_train_dataloader(self) -> DataLoader:
        """Return a DataLoader for training"""
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
        """Return a DataLoader for evaluation"""
        dataset_to_use = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return DataLoader(
            dataset_to_use,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    def save_model(self, output_dir):
        """Save the model"""
        self.model.save_pretrained(output_dir)