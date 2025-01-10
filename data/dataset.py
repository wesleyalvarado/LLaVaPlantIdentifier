# data/dataset.py
import torch
import logging
import traceback
from torch.utils.data import Dataset
from datasets import load_dataset
import os
import gc
import numpy as np

logger = logging.getLogger(__name__)

class MemoryEfficientPlantDataset(Dataset):
    def __init__(self, processor, split="train", sample_fraction=0.1, image_size=336):
        """Initialize the dataset with memory-efficient loading"""
        logger.info(f"Loading {split} dataset...")
        
        # Load the full dataset
        full_dataset = load_dataset(
            "nelorth/oxford-flowers",
            split=split,
            trust_remote_code=True
        )
        
        # Store categories
        self.categories = {
            i: name for i, name in enumerate(full_dataset.features['label'].names)
        }
        
        # Calculate minimum samples needed (at least 2 per class)
        num_classes = len(self.categories)
        min_total_samples = num_classes * 2  # 2 samples per class minimum
        requested_samples = int(len(full_dataset) * sample_fraction)
        total_samples = max(min_total_samples, requested_samples)
        
        logger.info(f"Taking {total_samples} samples from dataset of size {len(full_dataset)}")
        
        # Shuffle and select samples
        self.dataset = full_dataset.shuffle(seed=42).select(range(total_samples))
        
        self.processor = processor
        self.image_size = image_size
        
        logger.info(f"Successfully loaded {len(self.dataset)} images")
        logger.info(f"Number of categories: {len(self.categories)}")

    def __getitem__(self, idx):
        try:
            if idx >= len(self.dataset):
                raise IndexError(f"Index {idx} out of range for dataset of size {len(self.dataset)}")
                
            item = self.dataset[idx]
            logger.debug(f"Processing item {idx}")
            
            try:
                image = item['image']
                if image is None:
                    logger.warning(f"Image is None for index {idx}")
                    return None
                
                # Process image with corrected dimensions
                with torch.no_grad():
                    image_inputs = self.processor.image_processor(
                        image,
                        return_tensors="pt",
                        do_resize=True,
                        size=self.image_size,
                        do_center_crop=True,
                        do_normalize=True,
                    )
                    
                    # Fix pixel_values dimensions
                    pixel_values = image_inputs['pixel_values']
                    if pixel_values.ndim > 4:  # If we have extra dimensions
                        pixel_values = pixel_values.squeeze()  # Remove extra dims
                        if pixel_values.ndim == 3:  # If we removed too many
                            pixel_values = pixel_values.unsqueeze(0)  # Add batch dim back
                    
                    # Process text
                    label_idx = item['label']
                    prompt = f"Identify this {self.categories[label_idx]} flower."
                    
                    text_inputs = self.processor.tokenizer(
                        f"[INST] {prompt} [/INST]",
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128
                    )
                    
                    processed = {
                        'pixel_values': pixel_values,  # Using corrected pixel values
                        'input_ids': text_inputs['input_ids'],
                        'attention_mask': text_inputs['attention_mask'],
                        'labels': torch.tensor([label_idx])
                    }
                    
                    # Verify shapes
                    assert processed['pixel_values'].shape[-3:] == (3, self.image_size, self.image_size), \
                        f"Unexpected pixel_values shape: {processed['pixel_values'].shape}"
                    
                    return processed
                    
            except Exception as proc_error:
                logger.error(f"Processor failed for index {idx}: {proc_error}")
                logger.error(f"Processor error traceback: {traceback.format_exc()}")
                return None
                
        except Exception as e:
            logger.error(f"Sample processing failed for index {idx}: {e}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            return None

    def __len__(self):
        return len(self.dataset)