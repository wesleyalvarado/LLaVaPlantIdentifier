# data/dataset.py
import torch
import logging
import traceback
from torch.utils.data import Dataset
from utils.image_utils import validate_image_data, convert_to_pil_image, process_pil_image  # Add these imports
from datasets import load_dataset
import os
import gc  # Add garbage collector

# Set MPS memory limit
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

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
        del temp_dataset
        
        # Now load the actual dataset in streaming mode
        self.dataset = load_dataset(
            "nelorth/oxford-flowers",
            split=split,
            streaming=True  # Enable streaming mode
        )
        
        # Convert to list but limit size
        self.dataset = list(self.dataset.take(int(102 * sample_fraction)))  # Oxford flowers has 102 classes
        
        self.processor = processor
        self.image_size = image_size
        logger.debug(f"Using image size: {self.image_size}")
        
        logger.info(f"Loaded {len(self.dataset)} images for {split}")
        logger.info(f"Found {len(self.categories)} plant categories")

    def __getitem__(self, idx):
        """
        Memory-efficient item retrieval
        """
        try:
            # Clear any cached tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            item = self.dataset[idx]
            logger.debug(f"Processing item {idx}")
            
            # Process image with memory optimization
            try:
                # Convert image and process in one step
                image = convert_to_pil_image(item['image'], idx)
                if image is None:
                    return None
                
                # Process image with memory constraints
                with torch.no_grad():  # Disable gradient tracking
                    image_inputs = self.processor.image_processor(
                        image,
                        return_tensors="pt",
                        do_resize=True,
                        size=self.image_size,
                        do_center_crop=True,
                        do_normalize=True,
                        do_convert_rgb=True
                    )
                    
                    # Free up memory
                    del image
                    gc.collect()
                    
                    # Process text efficiently
                    label_idx = item['label']
                    prompt = f"Identify this {self.categories[label_idx]} flower."
                    
                    text_inputs = self.processor.tokenizer(
                        f"[INST] {prompt} [/INST]",
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128
                    )
                    
                    # Combine inputs with memory optimization
                    processed = {
                        'pixel_values': image_inputs['pixel_values'].detach(),  # Detach to free memory
                        'input_ids': text_inputs['input_ids'].detach(),
                        'attention_mask': text_inputs['attention_mask'].detach(),
                        'labels': torch.tensor([label_idx])
                    }
                    
                    # Clean up intermediate tensors
                    del image_inputs, text_inputs
                    gc.collect()
                    
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

def memory_efficient_collate_fn(batch):
    """
    Memory-efficient collate function
    """
    try:
        # Clear cache before batch processing
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Filter None values
        batch = [item for item in batch if item is not None]
        
        if not batch:
            return None
            
        # Process tensors with memory optimization
        with torch.no_grad():
            result = {
                key: torch.stack([item[key] for item in batch]).detach()
                for key in batch[0].keys()
            }
            
            # Clean up
            del batch
            gc.collect()
            
            return result
            
    except Exception as e:
        logger.error(f"Batch collation error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None