# data/dataset.py
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

logger = logging.getLogger(__name__)

class MemoryEfficientPlantDataset(Dataset):
    """Memory-efficient dataset implementation for plant identification using LLaVA."""
    
    def __init__(
        self,
        processor: ProcessorMixin,
        split: str = "train",
        sample_fraction: float = 1.0,
        cache_dir: Optional[str] = None,
        max_length: int = 64,
        image_size: int = 336,
        device: Optional[torch.device] = None
    ):
        """Initialize the dataset.
        
        Args:
            processor: LLaVA processor for tokenization and image processing
            split: Dataset split ('train' or 'test')
            sample_fraction: Fraction of dataset to load (0.0-1.0)
            cache_dir: Optional directory for caching processed samples
            max_length: Maximum length for text inputs
            image_size: Target image size
            device: Optional device for tensors
        """
        if not isinstance(processor, ProcessorMixin):
            raise ValueError("processor must be an instance of ProcessorMixin")
            
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        # Load dataset first to get class names
        logger.info("Loading dataset to get class names...")
        temp_dataset = load_dataset("dpdl-benchmark/oxford_flowers102", split=split)
        self.class_names = temp_dataset.features['label'].names
        logger.info(f"Loaded {len(self.class_names)} flower categories")
        
        # Calculate dataset size
        total_samples = len(temp_dataset)
        num_samples = max(1, int(total_samples * sample_fraction))
        
        # Load subset of dataset
        logger.info(f"Loading {num_samples}/{total_samples} samples ({sample_fraction*100:.1f}%) from dataset...")
        self.dataset = load_dataset(
            "dpdl-benchmark/oxford_flowers102",
            split=f"{split}[:{num_samples}]"
        )
        
        # Store image token for verification
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        logger.info(f"Image token ID: {self.image_token_id}")
        
        # Process and store samples
        self.processed_data: List[Dict[str, Any]] = []
        self._process_samples()
        
        logger.info(f"Successfully loaded {len(self.processed_data)} samples")
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _get_cache_path(self, idx: int) -> Optional[Path]:
        """Get cache file path for sample index."""
        if not self.cache_dir:
            return None
        return Path(self.cache_dir) / f"sample_{idx}.pt"

    def _load_from_cache(self, idx: int) -> Optional[Dict[str, Any]]:
        """Load processed sample from cache."""
        cache_path = self._get_cache_path(idx)
        if cache_path and cache_path.exists():
            try:
                return torch.load(cache_path, weights_only=True, map_location=self.device)
            except Exception as e:
                logger.warning(f"Failed to load cache for sample {idx}: {e}")
        return None

    def _save_to_cache(self, idx: int, processed: Dict[str, Any]) -> None:
        """Save processed sample to cache."""
        cache_path = self._get_cache_path(idx)
        if cache_path:
            try:
                torch.save(processed, cache_path)
            except Exception as e:
                logger.warning(f"Failed to cache sample {idx}: {e}")

    def _process_single_sample(self, idx: int) -> Optional[Dict[str, Any]]:
        """Process a single dataset sample.
            
        Args:
            idx: Sample index
                    
        Returns:
            Processed sample dictionary or None if processing fails
        """
        try:
            # Check cache first
            cached = self._load_from_cache(idx)
            if cached is not None:
                return cached
                    
            item = self.dataset[idx]
                
            # Create text inputs as explicit strings
            prompt = str("<image>\nWhat type of flower is shown in this image? Please identify the flower species.")
            target = str(f"This image shows a flower of class {self.class_names[item['label']]}.")
                
            # Process image and ensure single view
            image = item['image']
            if isinstance(image, (list, tuple)):
                image = image[0]  # Take first image if multiple views
                        
            # Process image with explicit size
            image_inputs = self.processor.image_processor(
                image,
                return_tensors="pt",
                size={'height': self.image_size, 'width': self.image_size}
            )
                
            # Handle multiple views in pixel values
            pixel_values = image_inputs['pixel_values']
            logger.info(f"Original pixel_values shape: {pixel_values.shape}")
                
            # If we have shape [1, 5, 3, H, W], take the first view
            if len(pixel_values.shape) >= 4 and pixel_values.shape[1] == 5:
                pixel_values = pixel_values.squeeze(0)[0]  # Take first view only
                logger.info(f"Reshaped pixel_values to: {pixel_values.shape}")
                
            # Process text input using explicit string
            text_inputs = self.processor.tokenizer(
                text=prompt,  # Explicit text parameter
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )
                
            # Process target text
            label_inputs = self.processor.tokenizer(
                text=target,  # Explicit text parameter
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )
                
            # Create input dictionary
            inputs = {
                'pixel_values': pixel_values.to(self.device).to(torch.float16),
                'input_ids': torch.tensor(text_inputs['input_ids'], dtype=torch.long).to(self.device),
                'attention_mask': torch.tensor(text_inputs['attention_mask'], dtype=torch.long).to(self.device),
                'labels': torch.tensor(label_inputs['input_ids'], dtype=torch.long).to(self.device),
                'image_sizes': torch.tensor([self.image_size, self.image_size], dtype=torch.long).to(self.device),
                'class_name': self.class_names[item['label']],
                'numerical_label': item['label'],
                'prompt': prompt,
                'target': target
            }
                            
            # Debug first sample
            if idx == 0:
                logger.info("First sample shape information:")
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  {key}: shape = {value.shape}")
                        if key == 'input_ids':
                            num_image_tokens = (value == self.image_token_id).sum().item()
                            logger.info(f"  Number of image tokens: {num_image_tokens}")
                
            # Cache processed sample
            self._save_to_cache(idx, inputs)
                
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
                    logger.info(f"Processed {idx} samples...")
                    # Memory cleanup
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            Processed sample dictionary
        """
        return self.processed_data[idx]

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset handling."""
        labels = [sample['numerical_label'] for sample in self.processed_data]
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        weights = total_samples / (len(self.class_names) * (class_counts + eps))
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # Clip weights to avoid extreme values
        weights = torch.clamp(weights, min=0.1, max=10.0)
        
        return weights.to(self.device)

    def cleanup(self):
        """Clean up resources."""
        self.processed_data.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()