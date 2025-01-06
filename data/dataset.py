import torch
import logging
import traceback
from torch.utils.data import Dataset
from datasets import load_dataset

from utils.image_utils import validate_image_data, convert_to_pil_image, process_pil_image
from utils.tensor_utils import fix_pixel_values_shape, validate_processed_sample

logger = logging.getLogger(__name__)

def memory_efficient_collate_fn(batch):
    """
    Custom collate function that handles None values and validates batch
    
    Args:
        batch: List of processed samples
    
    Returns:
        Collated batch or None
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if not batch:
        logger.warning("No valid items in the batch")
        return None
    
    try:
        # Combine pixel values
        pixel_values = torch.stack([item['pixel_values'].squeeze(0) for item in batch])
        
        # Combine input ids
        input_ids = torch.stack([item['input_ids'].squeeze(0) for item in batch])
        
        # Combine attention mask
        attention_mask = torch.stack([item['attention_mask'].squeeze(0) for item in batch])
        
        # Combine labels
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    except Exception as e:
        logger.error(f"Batch collation error: {e}")
        logger.error(f"Batch processing failed. Details: {traceback.format_exc()}")
        return None

class MemoryEfficientPlantDataset(Dataset):
    def __init__(self, processor, split="train", sample_fraction=0.1, image_size=336):
        """
        Initialize the dataset with memory-efficient loading
        
        Args:
            processor: LLaVA processor for image and text processing
            split: Dataset split to use (train/test)
            sample_fraction: Fraction of dataset to load
            image_size: Target image size for processing
        """
        logger.info(f"Loading {split} dataset...")
        
        # Load full dataset
        full_dataset = load_dataset("nelorth/oxford-flowers", split=split, trust_remote_code=True)
        
        # Select a fraction of the dataset
        num_samples = max(10, int(len(full_dataset) * sample_fraction))
        self.dataset = full_dataset.select(range(num_samples))
        
        # Store processor and configuration
        self.processor = processor
        self.image_size = image_size
        logger.debug(f"Using image size: {self.image_size}")
        
        # Extract category names
        self.categories = {
            i: name for i, name in enumerate(self.dataset.features['label'].names)
        }
        
        logger.info(f"Loaded {len(self.dataset)} images for {split}")
        logger.info(f"Found {len(self.categories)} plant categories")

    def __len__(self):
        """
        Return the number of samples in the dataset
        
        Returns:
            int: Number of samples
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve and process a single sample
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            Processed sample or None if processing fails
        """
        try:
            # Retrieve original item
            item = self.dataset[idx]
            logger.debug(f"Processing item {idx}")
            
            # Validate and process image
            if not validate_image_data(item['image'], idx):
                logger.warning(f"Image validation failed for index {idx}")
                return None
                
            image = convert_to_pil_image(item['image'], idx)
            if image is None:
                logger.warning(f"Image conversion failed for index {idx}")
                return None
                
            image = process_pil_image(image, self.image_size, idx)
            if image is None:
                logger.warning(f"Image processing failed for index {idx}")
                return None
            
            # Get label information
            label_idx = item['label']
            label = self.categories[label_idx]
            logger.debug(f"Label index: {label_idx}, category: {label}")
            
            prompt = f"Identify this {label} flower."
            
            try:
                # Specific processing for LLaVA v1.6
                image_inputs = self.processor.image_processor(
                    image, 
                    return_tensors="pt",
                    do_resize=True,
                    size=self.image_size,
                    do_center_crop=True,
                    do_normalize=True,
                    do_convert_rgb=True
                )
                
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
                    'attention_mask': text_inputs['attention_mask']
                }
                
                # Validate and standardize processed sample
                validated_sample = validate_processed_sample(processed, self.image_size)
                
                if validated_sample is None:
                    logger.warning(f"Invalid processed sample at index {idx}")
                    return None
                
                # Add labels to the sample
                validated_sample['labels'] = torch.tensor([label_idx])
                
                return validated_sample
                
            except Exception as proc_error:
                logger.error(f"Processor failed for index {idx}: {proc_error}")
                logger.error(f"Processor error traceback: {traceback.format_exc()}")
                return None
                
        except Exception as e:
            logger.error(f"Sample processing failed for index {idx}: {e}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            return None