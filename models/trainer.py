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
        """Perform a training step"""
        try:
            # Memory tracking
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Set model to training mode
            model.train()
            
            # Move inputs to device
            model_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
                if k in ['pixel_values', 'input_ids', 'attention_mask', 'labels']
            }
            
            # Forward pass
            outputs = model(**model_inputs)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            return loss.detach()
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
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