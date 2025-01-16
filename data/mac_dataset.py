# data/mac_dataset.py

import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import logging
import gc
from typing import Optional, Dict, Any, List, Tuple
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
        """Initialize dataset with Mac-optimized settings.
        
        Args:
            processor: LLaVA processor for tokenization and image processing
            split: Dataset split ('train' or 'test')
            sample_fraction: Fraction of dataset to load (0.0-1.0)
            cache_dir: Optional directory for caching processed samples
            max_length: Maximum length for text inputs
            image_size: Target image size
            device: Optional device for tensors
            batch_size: Batch size for processing samples (keep small for Mac)
        """
        try:
            if not isinstance(processor, ProcessorMixin):
                raise ValueError("processor must be an instance of ProcessorMixin")
                
            self.processor = processor
            self.max_length = max_length
            self.image_size = image_size
            self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            self.cache_dir = cache_dir
            self.batch_size = batch_size
            
            logger.info(f"Initializing dataset with device: {self.device}")
            
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
            self._process_samples_in_batches()
            
            logger.info(f"Successfully loaded {len(self.processed_data)} samples")
            
            # Memory cleanup
            self._cleanup_memory()
            
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            logger.error(traceback.format_exc())
            raise

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
                return torch.load(cache_path, map_location=self.device)
            except Exception as e:
                logger.warning(f"Failed to load cache for sample {idx}: {e}")
        return None

    def _save_to_cache(self, idx: int, processed: Dict[str, Any]) -> None:
        """Save processed sample to cache."""
        cache_path = self._get_cache_path(idx)
        if cache_path:
            try:
                # Move tensors to CPU before saving
                processed_cpu = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in processed.items()
                }
                torch.save(processed_cpu, cache_path)
            except Exception as e:
                logger.warning(f"Failed to cache sample {idx}: {e}")

    def _process_single_sample(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single dataset sample.
        
        Args:
            item: Raw dataset item
                    
        Returns:
            Processed sample dictionary or None if processing fails
        """
        try:
            # Create text inputs as explicit strings
            prompt = str("<image>\nWhat type of flower is shown in this image? Please identify the flower species.")
            target = str(f"This image shows a flower of class {self.class_names[item['label']]}.")
            
            # Process image and ensure correct format
            image = item['image']
            if isinstance(image, (list, tuple)):
                image = image[0]  # Take first image if multiple views
            
            # Process image with explicit size
            image_inputs = self.processor.image_processor(
                image,
                return_tensors="pt",
                size={'height': self.image_size, 'width': self.image_size}
            )
            
            # Move to device and convert to float16 for memory efficiency
            pixel_values = image_inputs['pixel_values'].to(self.device, dtype=torch.float16)
            
            # Process text input
            text_inputs = self.processor.tokenizer(
                text=prompt,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )
            
            # Process target text
            label_inputs = self.processor.tokenizer(
                text=target,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )
            
            # Create input dictionary with explicit device placement
            inputs = {
                'pixel_values': pixel_values,
                'input_ids': torch.tensor(text_inputs['input_ids'], dtype=torch.long, device=self.device),
                'attention_mask': torch.tensor(text_inputs['attention_mask'], dtype=torch.long, device=self.device),
                'labels': torch.tensor(label_inputs['input_ids'], dtype=torch.long, device=self.device),
                'image_sizes': torch.tensor([self.image_size, self.image_size], dtype=torch.long, device=self.device),
                'class_name': self.class_names[item['label']],
                'numerical_label': item['label'],
                'prompt': prompt,
                'target': target
            }
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            return None

    def _process_samples_in_batches(self):
        """Process all samples in batches with memory management."""
        try:
            total_samples = len(self.dataset)
            for start_idx in range(0, total_samples, self.batch_size):
                # Clear memory before processing batch
                self._cleanup_memory()
                
                end_idx = min(start_idx + self.batch_size, total_samples)
                batch = self.dataset[start_idx:end_idx]
                
                for i, item in enumerate(batch):
                    idx = start_idx + i
                    
                    # Try loading from cache first
                    cached = self._load_from_cache(idx)
                    if cached is not None:
                        self.processed_data.append(cached)
                        continue
                    
                    # Process sample if not cached
                    try:
                        processed = self._process_single_sample(item)
                        if processed is not None:
                            self.processed_data.append(processed)
                            self._save_to_cache(idx, processed)
                    except Exception as e:
                        logger.warning(f"Error processing sample {idx}: {e}")
                        continue
                
                # Log progress
                if (start_idx + self.batch_size) % 100 == 0:
                    logger.info(f"Processed {min(start_idx + self.batch_size, total_samples)}/{total_samples} samples...")
                    
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            logger.error(traceback.format_exc())
            raise

    def _cleanup_memory(self):
        """Clean up memory explicitly."""
        gc.collect()
        if torch.backends.mps.is_available():
            # Clear MPS cache if available
            torch.mps.empty_cache()

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
        try:
            labels = [sample['numerical_label'] for sample in self.processed_data]
            
            # Ensure we have the correct number of classes
            num_classes = len(self.class_names)
            class_counts = np.bincount(labels, minlength=num_classes)
            total_samples = len(labels)
            
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            weights = total_samples / (num_classes * (class_counts + eps))
            
            # Convert to tensor and move to correct device
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            
            # Clip weights to avoid extreme values
            weights = torch.clamp(weights, min=0.1, max=10.0)
            
            logger.info(f"Class weights shape: {weights.shape}")
            logger.info(f"Class weights range: min={weights.min():.4f}, max={weights.max():.4f}")
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating class weights: {e}")
            # Return uniform weights if calculation fails
            return torch.ones(len(self.class_names), device=self.device)

    def cleanup(self):
        """Clean up resources explicitly."""
        self.processed_data.clear()
        self._cleanup_memory()