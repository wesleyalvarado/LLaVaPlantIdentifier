# models/trainer.py

from transformers import Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Union, Any
import logging
import traceback
import psutil
from utils.tensor_utils import check_tensor_shape, validate_tensor_outputs
import gc

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
       """
       logger.debug("Creating training dataloader")
       return DataLoader(
           self.train_dataset,  # Changed from eval_dataset to train_dataset
           batch_size=1,  # Force batch size of 1
           collate_fn=memory_efficient_collate_fn,
           num_workers=0,
           pin_memory=False,
           shuffle=True  # Add shuffling for training
       )
  
   def get_eval_dataloader(self) -> DataLoader:
       """
       Return a DataLoader for evaluation with memory-efficient settings.
       """
       logger.debug("Creating evaluation dataloader")
       return DataLoader(
           self.eval_dataset,
           batch_size=1,
           collate_fn=memory_efficient_collate_fn,
           num_workers=0,
           pin_memory=False
       )
  
   def training_step(
       self, 
       model: nn.Module, 
       inputs: Dict[str, Union[torch.Tensor, Any]], 
       *args
   ) -> torch.Tensor:
       """
       Perform a training step with comprehensive memory tracking and error handling.
       
       Args:
           model: Neural network model
           inputs: Dictionary of input tensors
       
       Returns:
           Detached loss tensor
       """
       try:
           # Initial memory tracking
           process = psutil.Process()
           logger.info(f"System memory usage before step: {process.memory_info().rss / 1024 / 1024:.2f} MB")
           
           # Clear memory before starting
           gc.collect()
           if torch.cuda.is_available():
               torch.cuda.empty_cache()
               initial_gpu_memory = torch.cuda.memory_allocated()/1024**2
               logger.debug(f"Initial GPU Memory: {initial_gpu_memory:.2f}MB")
           
           logger.debug("Starting training step")
           model.train()
           
           # Input validation
           if inputs is None:
               logger.warning("Inputs is None")
               return torch.tensor(0.0, device=self.args.device)
           
           logger.debug(f"Input keys: {list(inputs.keys())}")
           
           # Track memory before input preparation
           if torch.cuda.is_available():
               logger.debug(f"Memory before input prep: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
           
           # Prepare inputs
           try:
               logger.debug("Preparing inputs")
               inputs = self._prepare_inputs(inputs)
           except Exception as prep_error:
               logger.error(f"Input preparation failed: {prep_error}")
               raise
           
           # Process pixel values with detailed logging
           if 'pixel_values' in inputs:
               try:
                   pixel_values = inputs['pixel_values']
                   logger.debug(f"Original pixel_values shape: {pixel_values.shape}")
                   logger.debug(f"Original pixel_values dtype: {pixel_values.dtype}")
                   logger.debug(f"Original pixel_values device: {pixel_values.device}")
                   
                   # Handle shapes with validation
                   if pixel_values.ndim > 4:
                       logger.debug(f"Squeezing pixel_values from shape {pixel_values.shape}")
                       pixel_values = pixel_values.squeeze()
                   if pixel_values.ndim == 3:
                       logger.debug("Adding batch dimension to pixel_values")
                       pixel_values = pixel_values.unsqueeze(0)
                   
                   inputs['pixel_values'] = pixel_values
                   logger.debug(f"Final pixel_values shape: {pixel_values.shape}")
                   
               except Exception as pixel_error:
                   logger.error(f"Pixel values processing failed: {pixel_error}")
                   raise
           
           # Log all tensor shapes and devices
           for key, value in inputs.items():
               if isinstance(value, torch.Tensor):
                   logger.debug(f"{key} - Shape: {value.shape}, Device: {value.device}, Dtype: {value.dtype}")
           
           # Forward pass with memory tracking
           logger.debug("Starting forward pass")
           try:
               with self.compute_loss_context_manager():
                   if torch.cuda.is_available():
                       logger.debug(f"Memory before loss computation: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                   
                   loss = self.compute_loss(model, inputs)
                   
                   if loss is None:
                       raise ValueError("Loss computation returned None")
                   
                   logger.debug(f"Loss value: {loss.item()}")
                   logger.debug(f"Loss dtype: {loss.dtype}")
                   logger.debug(f"Loss device: {loss.device}")
                   
           except Exception as loss_error:
               logger.error(f"Loss computation failed: {loss_error}")
               raise
           
           # Scale loss if needed
           if self.args.gradient_accumulation_steps > 1:
               logger.debug("Scaling loss for gradient accumulation")
               loss = loss / self.args.gradient_accumulation_steps
           
           # Backward pass with memory tracking
           logger.debug("Starting backward pass")
           try:
               if torch.cuda.is_available():
                   logger.debug(f"Memory before backward: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
               
               loss.backward()
               
               if torch.cuda.is_available():
                   logger.debug(f"Memory after backward: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
               
           except Exception as backward_error:
               logger.error(f"Backward pass failed: {backward_error}")
               raise
           
           logger.debug("Backward pass complete")
           
           # Final memory cleanup
           gc.collect()
           if torch.cuda.is_available():
               torch.cuda.empty_cache()
               final_gpu_memory = torch.cuda.memory_allocated()/1024**2
               logger.debug(f"Final GPU Memory: {final_gpu_memory:.2f}MB")
               logger.debug(f"Memory change during step: {final_gpu_memory - initial_gpu_memory:.2f}MB")
           
           # Final system memory check
           logger.info(f"System memory usage after step: {process.memory_info().rss / 1024 / 1024:.2f} MB")
           
           return loss.detach()

       except Exception as e:
           logger.error(f"Error in training step: {str(e)}")
           if torch.cuda.is_available():
               logger.error(f"GPU Memory at error: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
           logger.error(f"System memory at error: {process.memory_info().rss / 1024 / 1024:.2f} MB")
           logger.error(f"Detailed traceback: {traceback.format_exc()}")
           raise

   def train(self):
       try:
           logger.info("Starting training")
           logger.debug("Initial memory cleanup")
           gc.collect()
           if torch.cuda.is_available():
               torch.cuda.empty_cache()
           return super().train()
       except Exception as e:
           logger.error(f"Training failed: {e}")
           logger.error(f"Last successful batch: {self.state.log_history[-1] if self.state.log_history else 'No successful batches'}")
           logger.error(f"Detailed traceback: {traceback.format_exc()}")
           raise