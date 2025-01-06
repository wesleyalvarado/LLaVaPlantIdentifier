# models/trainer.py

from transformers import Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Union, Any
import logging
import traceback
from utils.tensor_utils import check_tensor_shape, validate_tensor_outputs

logger = logging.getLogger(__name__)

def memory_efficient_collate_fn(batch):
    """
    Collate function that handles None values and validates tensors
    
    Args:
        batch: List of samples to be batched
    
    Returns:
        Batched dictionary or None if batching fails
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if not batch:
        logger.warning("No valid items in the batch, returning None")
        return None
    
    try:
        # Required keys for LLaVA training
        required_keys = {'input_ids', 'attention_mask', 'pixel_values', 'labels'}
        item_keys = set(batch[0].keys())
        
        if not required_keys.issubset(item_keys):
            missing_keys = required_keys - item_keys
            logger.error(f"Missing required keys in batch item: {missing_keys}")
            return None
        
        # Stack tensors with validation
        batched = {}
        for key in required_keys:
            try:
                # Handle different tensor shapes
                def process_tensor(item):
                    tensor = item[key]
                    # Squeeze extra dimensions
                    while tensor.ndim > 2:
                        tensor = tensor.squeeze(0)
                    return tensor
                
                stacked = torch.stack([process_tensor(item) for item in batch])
                
                # Validate tensor shape
                check_tensor_shape(stacked, f"Batched {key}")
                batched[key] = stacked
            
            except Exception as e:
                logger.error(f"Failed to stack {key}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None
        
        return batched
        
    except Exception as e:
        logger.error(f"Batch creation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

class CustomTrainer(Trainer):
   
   def __init__(self, *args, **kwargs):
       # Remove dataloader_pin_memory if it exists in kwargs
       if 'dataloader_pin_memory' in kwargs:
           del kwargs['dataloader_pin_memory']
       super().__init__(*args, **kwargs)

   def get_train_dataloader(self) -> DataLoader:
       """
       Return a DataLoader for training with memory-efficient collate function.
       
       Returns:
           DataLoader configured for training
       """
       return DataLoader(
           self.train_dataset,
           batch_size=self.args.per_device_train_batch_size,
           collate_fn=memory_efficient_collate_fn,
           num_workers=self.args.dataloader_num_workers,
           pin_memory=self.args.dataloader_pin_memory if hasattr(self.args, 'dataloader_pin_memory') else False
       )

   def training_step(
       self, 
       model: nn.Module, 
       inputs: Dict[str, Union[torch.Tensor, Any]], 
       *args
   ) -> torch.Tensor:
       """
       Perform a training step with comprehensive debugging.
       
       Args:
           model: Neural network model
           inputs: Dictionary of input tensors
       
       Returns:
           Detached loss tensor
       """
       try:
           logger.debug("Starting training step")
           model.train()
           
           # Debug inputs before preparation
           if inputs is None:
               logger.warning("Inputs is None")
               return torch.tensor(0.0, device=self.args.device)
           
           logger.debug(f"Input keys: {inputs.keys()}")
           
           # Log memory before input preparation
           if torch.cuda.is_available():
               logger.debug(f"GPU Memory before input prep: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
           
           logger.debug("Preparing inputs")
           inputs = self._prepare_inputs(inputs)
           
           # Special handling for pixel_values
           if 'pixel_values' in inputs:
               pixel_values = inputs['pixel_values']
               logger.debug(f"Original pixel_values shape: {pixel_values.shape}")
               
               # Specific handling for unusual shapes
               if pixel_values.ndim == 4 and pixel_values.shape[1] == 3 and pixel_values.shape[0] == 3:
                   logger.debug("Detected [3, 3, channels, size] shape, reshaping")
                   pixel_values = pixel_values[0]
               
               # Ensure 4D tensor: [batch, channels, height, width]
               while pixel_values.ndim > 4:
                   pixel_values = pixel_values.squeeze(0)
               
               if pixel_values.ndim == 3:
                   pixel_values = pixel_values.unsqueeze(0)
               
               inputs['pixel_values'] = pixel_values
           
           # Log tensor details
           for key, value in inputs.items():
               if isinstance(value, torch.Tensor):
                   logger.debug(f"Input {key} shape: {value.shape}")
                   logger.debug(f"Input {key} dtype: {value.dtype}")
           
           # Log memory before forward pass
           if torch.cuda.is_available():
               logger.debug(f"GPU Memory before forward pass: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
           
           # Compute loss with additional debugging
           logger.debug("Starting forward pass")
           with self.compute_loss_context_manager():
               try:
                   logger.debug("Computing model outputs...")
                   loss = self.compute_loss(model, inputs)
                   
                   if loss is None:
                       logger.error("Loss computation resulted in None")
                       raise ValueError("Loss computation failed")
                   
                   logger.debug(f"Loss value: {loss.item()}")
               except Exception as e:
                   logger.error(f"Error in loss computation: {str(e)}")
                   logger.error(f"Traceback: {traceback.format_exc()}")
                   raise
           
           logger.debug("Forward pass complete")
           
           # Log memory after forward pass
           if torch.cuda.is_available():
               logger.debug(f"GPU Memory after forward pass: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

           # Standard loss scaling and backward pass
           if self.args.n_gpu > 1:
               loss = loss.mean()

           logger.debug("Starting backward pass")
           if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
               loss = loss / self.args.gradient_accumulation_steps

           if self.do_grad_scaling:
               self.scaler.scale(loss).backward()
           elif self.use_apex:
               with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                   scaled_loss.backward()
           elif self.deepspeed:
               loss = self.deepspeed.backward(loss)
           else:
               loss.backward()
               
           logger.debug("Backward pass complete")
           
           # Log final memory usage
           if torch.cuda.is_available():
               logger.debug(f"GPU Memory after backward pass: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

           return loss.detach()

       except Exception as e:
           logger.error(f"Error in training step: {str(e)}")
           logger.error(f"Detailed traceback: {traceback.format_exc()}")
           raise

   def train(self):
       """
       Override train method with enhanced error logging
       
       Returns:
           Training results
       """
       try:
           logger.info("Starting training")
           return super().train()
       except Exception as e:
           logger.error(f"Training failed: {e}")
           logger.error(f"Last successful batch: {self.state.log_history[-1] if self.state.log_history else 'No successful batches'}")
           logger.error(f"Detailed traceback: {traceback.format_exc()}")
           raise