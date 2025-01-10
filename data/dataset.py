# data/dataset.py
import torch
import logging
import traceback
from torch.utils.data import Dataset
from datasets import load_dataset
import os
import gc

logger = logging.getLogger(__name__)

class MemoryEfficientPlantDataset(Dataset):
    def __init__(self, processor, split="train", sample_fraction=0.1, image_size=336):
        """
        Initialize the dataset with memory-efficient loading
        """
        logger.info(f"Loading {split} dataset...")
        
        # Load feature info first (non-streaming)
        temp_dataset = load_dataset(
            "nelorth/oxford-flowers",
            split=split,
            trust_remote_code=True
        )
        
        # Store categories
        self.categories = {
            i: name for i, name in enumerate(temp_dataset.features['label'].names)
        }
        
        # Calculate minimum number of samples needed (at least 2 per class)
        min_samples_per_class = max(2, int(102 * sample_fraction))  # 102 is number of classes
        total_samples = min_samples_per_class * len(self.categories)
        
        # Now load the actual dataset
        self.dataset = load_dataset(
            "nelorth/oxford-flowers",
            split=split,
            trust_remote_code=True
        )
        
        # Take required number of samples
        self.dataset = list(self.dataset.shuffle().take(total_samples))
        
        self.processor = processor
        self.image_size = image_size
        logger.info(f"Loaded {len(self.dataset)} images for {split}")
        logger.info(f"Found {len(self.categories)} plant categories")

    # Rest of the class implementation remains the same...
    def __getitem__(self, idx):
        """
        Memory-efficient item retrieval
        """
        try:
            if idx >= len(self.dataset):
                raise IndexError(f"Index {idx} out of range for dataset of size {len(self.dataset)}")
                
            item = self.dataset[idx]
            logger.debug(f"Processing item {idx}")
            
            # Process image and create prompt
            try:
                image = item['image']
                if image is None:
                    return None
                
                # Process image
                with torch.no_grad():
                    image_inputs = self.processor.image_processor(
                        image,
                        return_tensors="pt",
                        do_resize=True,
                        size=self.image_size,
                        do_center_crop=True,
                        do_normalize=True,
                    )
                    
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
                    
                    # Combine inputs
                    processed = {
                        'pixel_values': image_inputs['pixel_values'],
                        'input_ids': text_inputs['input_ids'],
                        'attention_mask': text_inputs['attention_mask'],
                        'labels': torch.tensor([label_idx])
                    }
                    
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