# data/dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import logging
import gc
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)
class MemoryEfficientPlantDataset(Dataset):
    """Memory-efficient dataset implementation for plant identification using LLaVA."""
    
    def __init__(
        self,
        split: str = "train",
        sample_fraction: float = 1.0,
        processor = None
    ):
        if processor is None:
            raise ValueError("processor is required")
            
        self.processor = processor
        
        # Load dataset first to get class names
        logger.info("Loading dataset to get class names...")
        temp_dataset = load_dataset("dpdl-benchmark/oxford_flowers102", split=split)
        self.class_names = temp_dataset.features['label'].names
        logger.info(f"Loaded {len(self.class_names)} flower categories")
        
        # Calculate dataset size
        num_samples = max(1, int(len(temp_dataset) * sample_fraction))
        
        # Load subset of dataset
        logger.info(f"Loading {num_samples} samples from dataset...")
        self.dataset = load_dataset(
            "dpdl-benchmark/oxford_flowers102",
            split=f"{split}[:{num_samples}]"
        )
        
        # Store image token for verification
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        logger.info(f"Image token ID: {self.image_token_id}")
        
        # Process and store samples
        self.processed_data = []
        self._process_samples()
        
        logger.info(f"Successfully loaded {len(self.processed_data)} samples")

    def _process_single_sample(self, idx: int) -> Optional[Dict[str, Any]]:
        """Process a single dataset sample."""
        try:
            item = self.dataset[idx]
            
            # Create text inputs
            prompt = "<image>\nWhat type of flower is shown in this image? Please identify the flower species."
            target = f"This image shows a flower of class {self.class_names[item['label']]}."
            
            # Process image and ensure single view
            image = item['image']
            image_inputs = self.processor.image_processor(
                image,
                return_tensors="pt"
            )
            
            # Handle multiple views - take only the first view
            pixel_values = image_inputs['pixel_values']
            if len(pixel_values.shape) >= 4:
                logger.info(f"Original pixel_values shape: {pixel_values.shape}")
                # If shape is [1, 5, 3, 336, 336], we want [3, 336, 336]
                pixel_values = pixel_values.squeeze(0)[0]  # Take first view
                logger.info(f"Reshaped pixel_values to: {pixel_values.shape}")
            
            # Now pixel_values should be [3, 336, 336]
            
            # Process text input
            text_inputs = self.processor.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors=None
            )
            
            # Create input dictionary with image sizes
            inputs = {
                'pixel_values': pixel_values,
                'input_ids': torch.tensor(text_inputs['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(text_inputs['attention_mask'], dtype=torch.long),
                'image_sizes': torch.tensor([336, 336], dtype=torch.long)  # Add image sizes
            }

            
            # Process target
            with self.processor.tokenizer.as_target_tokenizer():
                labels = self.processor.tokenizer(
                    target,
                    padding="max_length",
                    truncation=True,
                    max_length=64,
                    return_tensors=None
                )['input_ids']
            inputs['labels'] = torch.tensor(labels, dtype=torch.long)
            
            # Add metadata
            inputs['class_name'] = self.class_names[item['label']]
            inputs['numerical_label'] = item['label']
            inputs['prompt'] = prompt
            inputs['target'] = target
            
            # Debug first sample
            if idx == 0:
                logger.info("First sample shape information:")
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  {key}: shape = {value.shape}")
                        if key == 'input_ids':
                            num_image_tokens = (value == self.image_token_id).sum().item()
                            logger.info(f"  Number of image tokens: {num_image_tokens}")
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            return None

    def _process_samples(self):
        """Process all samples and store results."""
        for idx in range(len(self.dataset)):
            try:
                processed = self._process_single_sample(idx)
                if processed is not None:
                    self.processed_data.append(processed)
                
                if idx % 100 == 0 and idx > 0:
                    gc.collect()
                    logger.info(f"Processed {idx} samples...")
                    
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.processed_data[idx]