# data/mac_dataset.py

import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import logging
import gc
from typing import Optional, Dict, Any, List
from PIL import Image
import numpy as np
from transformers import ProcessorMixin
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

class MemoryEfficientPlantDataset(Dataset):
    """Memory-efficient dataset implementation optimized for M2 Mac."""
    
    def __init__(
        self,
        processor: ProcessorMixin,
        split: str = "train",
        sample_fraction: float = 0.1,
        cache_dir: Optional[str] = None,
        max_length: int = 64,
        image_size: int = 336,
        device: Optional[torch.device] = None,
        batch_size: int = 4
    ):
        """Initialize dataset."""
        try:
            self.processor = processor
            self.max_length = max_length
            self.image_size = image_size
            self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            self.cache_dir = cache_dir
            self.batch_size = batch_size
            
            # Load dataset
            logger.info("Loading dataset...")
            self.full_dataset = load_dataset("dpdl-benchmark/oxford_flowers102")
            
            # Log dataset structure
            logger.info(f"Dataset structure: {self.full_dataset}")
            logger.info(f"Available splits: {self.full_dataset.keys()}")
            
            # Get class names from train split features
            self.class_names = self.full_dataset['train'].features['label'].names
            logger.info(f"Loaded {len(self.class_names)} flower categories")
            
            # Get the correct split
            current_split = self.full_dataset[split]
            total_samples = len(current_split)
            num_samples = max(1, int(total_samples * sample_fraction))
            
            # Log the structure of one item before selection
            sample_item = current_split[0]
            logger.info(f"Sample item structure: {type(sample_item)}")
            logger.info(f"Sample item content: {sample_item}")
            
            # Select subset
            self.dataset = current_split.select(range(num_samples))
            logger.info(f"Loading {num_samples}/{total_samples} samples ({sample_fraction*100:.1f}%) from dataset...")
            
            # Store image token for verification
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
            logger.info(f"Image token ID: {self.image_token_id}")
            
            # Process samples
            self.processed_data = []
            self._process_samples_in_batches()
            
            logger.info(f"Successfully loaded {len(self.processed_data)} samples")
            
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            logger.error(traceback.format_exc())
            raise

    def __len__(self):
        """Get length of dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """Get item from dataset."""
        if not 0 <= idx < len(self.processed_data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.processed_data)}")
        return self.processed_data[idx]

    def _process_single_sample(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single dataset sample for LLaVA."""
        try:
            # Create text strings
            prompt = "<image>\nWhat type of flower is shown in this image? Please identify the flower species."
            target = f"This image shows a flower of class {self.class_names[item['label']]}"
            
            # Process image
            image = item['image']
            image_inputs = self.processor.image_processor(
                image,
                return_tensors="pt",
                size={'height': self.image_size, 'width': self.image_size}
            )
            
            # Process text with padding to max length
            text_inputs = self.processor.tokenizer(
                text=prompt,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            label_inputs = self.processor.tokenizer(
                text=target,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Debug logging
            logger.debug(f"Processing item with label: {item['label']}")
            logger.debug(f"image_inputs shape: {image_inputs['pixel_values'].shape}")
            logger.debug(f"text_inputs shape: {text_inputs['input_ids'].shape}")
            
            # Create input dictionary with proper shapes
            inputs = {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),  # Remove batch dim [C, H, W]
                'input_ids': text_inputs['input_ids'].squeeze(0),         # Remove batch dim [seq_len]
                'attention_mask': text_inputs['attention_mask'].squeeze(0),# Remove batch dim [seq_len]
                'labels': label_inputs['input_ids'].squeeze(0),           # Remove batch dim [seq_len]
                'class_name': self.class_names[item['label']],
                'numerical_label': item['label']
            }
            
            # Move tensors to device
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            
            # Debug final shapes
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    logger.debug(f"Final {key} shape: {value.shape}")
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            logger.error(traceback.format_exc())
            return None

    def _process_samples_in_batches(self):
        """Process samples in batches."""
        try:
            total_samples = len(self.dataset)
            
            # Process each item individually for better error tracking
            for idx in range(total_samples):
                try:
                    item = self.dataset[idx]
                    processed = self._process_single_sample(item)
                    if processed is not None:
                        self.processed_data.append(processed)
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    logger.error(traceback.format_exc())
                
                # Log progress
                if idx % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{total_samples} samples...")
                
                # Cleanup memory periodically
                if idx % self.batch_size == 0:
                    self._cleanup_memory()
                    
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            logger.error(traceback.format_exc())

    def _cleanup_memory(self):
        """Clean up memory explicitly."""
        gc.collect()
        if torch.backends.mps.is_available():
            # Clear MPS cache if available
            torch.mps.empty_cache()
            # Synchronize MPS stream if available
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()