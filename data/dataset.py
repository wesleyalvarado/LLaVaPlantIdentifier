# data/dataset.pyimport torch
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
        
        # Preprocess and store samples
        self.processed_data = []
        for idx in range(len(self.dataset)):
            try:
                processed_item = self._process_item(idx)
                if processed_item is not None:
                    self.processed_data.append(processed_item)
            except Exception as e:
                logger.warning(f"Skipping sample {idx} due to processing error: {e}")
        
        logger.info(f"Successfully loaded {len(self.processed_data)} processed images")
        logger.info(f"Number of categories: {len(self.categories)}")

    def _process_item(self, idx):
        """Process a single item with careful dimension handling"""
        try:
            item = self.dataset[idx]
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
                
                # Clean up pixel values dimensions
                pixel_values = image_inputs['pixel_values']
                
                # Ensure pixel_values is 3D (C, H, W)
                if pixel_values.ndim > 3:
                    # Remove extra dimensions, aiming for (C, H, W)
                    while pixel_values.ndim > 3:
                        pixel_values = pixel_values.squeeze(0)
                
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
                
                # Clean up input_ids and attention_mask dimensions
                input_ids = text_inputs['input_ids'].squeeze()
                attention_mask = text_inputs['attention_mask'].squeeze()
                
                processed = {
                    'pixel_values': pixel_values,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': torch.tensor(label_idx)
                }
                
                # Verify shapes
                assert processed['pixel_values'].ndim == 3, \
                    f"Unexpected pixel_values dimensions: {processed['pixel_values'].ndim}"
                assert processed['pixel_values'].shape[1:] == (self.image_size, self.image_size), \
                    f"Unexpected pixel_values shape: {processed['pixel_values'].shape}"
                
                return processed
        
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            logger.error(traceback.format_exc())
            return None

    def __getitem__(self, idx):
        """Return preprocessed sample directly"""
        return self.processed_data[idx]

    def __len__(self):
        """Return length of processed data"""
        return len(self.processed_data)