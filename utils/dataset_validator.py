# utils/dataset_validator.py
import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for dataset validation."""
    min_samples_per_class: int = 10
    max_aspect_ratio: float = 2.0
    min_image_size: int = 32
    max_image_size: int = 1024
    allowed_pixel_range: Tuple[float, float] = (0.0, 255.0)
    required_keys: List[str] = None
    max_memory_usage: float = 0.8  # Maximum fraction of available GPU memory to use

class DatasetValidator:
    """Comprehensive dataset validation system."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validation_results = {}
        
    def validate_sample(self, sample: Dict) -> List[str]:
        """Validate a single dataset sample."""
        errors = []
        
        # Check required keys
        if self.config.required_keys:
            missing_keys = [key for key in self.config.required_keys if key not in sample]
            if missing_keys:
                errors.append(f"Missing required keys: {missing_keys}")
        
        # Validate pixel values
        if 'pixel_values' in sample:
            pixel_values = sample['pixel_values']
            if torch.is_tensor(pixel_values):
                if torch.isnan(pixel_values).any():
                    errors.append("NaN values found in pixel_values")
                if torch.isinf(pixel_values).any():
                    errors.append("Infinite values found in pixel_values")
                    
                # Check value range
                min_val, max_val = pixel_values.min().item(), pixel_values.max().item()
                if min_val < self.config.allowed_pixel_range[0] or max_val > self.config.allowed_pixel_range[1]:
                    errors.append(f"Pixel values outside allowed range: [{min_val}, {max_val}]")
        
        # Validate labels
        if 'labels' in sample:
            labels = sample['labels']
            if torch.is_tensor(labels):
                if labels.dtype not in [torch.long, torch.int32, torch.int64]:
                    errors.append(f"Incorrect label dtype: {labels.dtype}")
        
        return errors

    def validate_dataset_statistics(self, dataset: Dataset) -> Dict:
        """Analyze dataset statistics and distributions."""
        stats = {
            'num_samples': len(dataset),
            'class_distribution': {},
            'image_sizes': [],
            'aspect_ratios': [],
            'pixel_statistics': {
                'mean': [],
                'std': [],
                'min': float('inf'),
                'max': float('-inf')
            }
        }
        
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                
                # Update class distribution
                if 'numerical_label' in sample:
                    label = sample['numerical_label']
                    stats['class_distribution'][label] = stats['class_distribution'].get(label, 0) + 1
                
                # Check image properties
                if 'pixel_values' in sample:
                    pixel_values = sample['pixel_values']
                    if torch.is_tensor(pixel_values):
                        if len(pixel_values.shape) >= 3:  # [C, H, W] or [B, C, H, W]
                            h, w = pixel_values.shape[-2:]
                            stats['image_sizes'].append((h, w))
                            stats['aspect_ratios'].append(w / h)
                            
                            # Update pixel statistics
                            stats['pixel_statistics']['mean'].append(pixel_values.mean().item())
                            stats['pixel_statistics']['std'].append(pixel_values.std().item())
                            stats['pixel_statistics']['min'] = min(stats['pixel_statistics']['min'], 
                                                                pixel_values.min().item())
                            stats['pixel_statistics']['max'] = max(stats['pixel_statistics']['max'], 
                                                                pixel_values.max().item())
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # Compute final statistics
        if stats['pixel_statistics']['mean']:
            stats['pixel_statistics']['mean'] = np.mean(stats['pixel_statistics']['mean'])
            stats['pixel_statistics']['std'] = np.mean(stats['pixel_statistics']['std'])
        
        return stats

    def validate_memory_usage(self, dataset: Dataset, batch_size: int) -> Tuple[bool, str]:
        """Validate memory usage with given batch size."""
        if not torch.cuda.is_available():
            return True, "No GPU available - skipping memory validation"
            
        try:
            # Try loading a single batch
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            batch = next(iter(loader))
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            if memory_used > self.config.max_memory_usage:
                return False, f"Batch size too large - using {memory_used:.1%} of GPU memory"
                
            return True, f"Memory usage acceptable - {memory_used:.1%} of GPU memory"
            
        except Exception as e:
            return False, f"Memory validation failed: {str(e)}"
    
    def validate_dataset(self, dataset: Dataset, batch_size: int) -> Dict:
        """Run comprehensive dataset validation."""
        validation_results = {
            'overall_status': 'passed',
            'errors': [],
            'warnings': [],
            'statistics': None,
            'memory_check': None
        }
        
        # Validate individual samples
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                errors = self.validate_sample(sample)
                if errors:
                    validation_results['errors'].extend([f"Sample {idx}: {error}" for error in errors])
            except Exception as e:
                validation_results['errors'].append(f"Failed to validate sample {idx}: {str(e)}")
        
        # Collect dataset statistics
        try:
            validation_results['statistics'] = self.validate_dataset_statistics(dataset)
        except Exception as e:
            validation_results['errors'].append(f"Failed to compute dataset statistics: {str(e)}")
        
        # Check memory usage
        memory_ok, memory_msg = self.validate_memory_usage(dataset, batch_size)
        validation_results['memory_check'] = {'status': memory_ok, 'message': memory_msg}
        
        # Generate warnings for potential issues
        stats = validation_results['statistics']
        if stats:
            # Check class balance
            if 'class_distribution' in stats:
                class_counts = list(stats['class_distribution'].values())
                if max(class_counts) / min(class_counts) > 10:
                    validation_results['warnings'].append(
                        "Severe class imbalance detected - consider augmentation or resampling"
                    )
            
            # Check image size consistency
            if 'image_sizes' in stats:
                unique_sizes = set(stats['image_sizes'])
                if len(unique_sizes) > 1:
                    validation_results['warnings'].append(
                        f"Inconsistent image sizes detected: {len(unique_sizes)} different sizes"
                    )
        
        # Set final status
        if validation_results['errors']:
            validation_results['overall_status'] = 'failed'
        elif validation_results['warnings']:
            validation_results['overall_status'] = 'passed_with_warnings'
        
        return validation_results

    def log_validation_results(self, results: Dict):
        """Log validation results in a structured format."""
        logger.info("\n=== Dataset Validation Results ===")
        logger.info(f"Overall Status: {results['overall_status']}")
        
        if results['errors']:
            logger.error("\nErrors Found:")
            for error in results['errors']:
                logger.error(f"  - {error}")
                
        if results['warnings']:
            logger.warning("\nWarnings:")
            for warning in results['warnings']:
                logger.warning(f"  - {warning}")
        
        if results['statistics']:
            logger.info("\nDataset Statistics:")
            stats = results['statistics']
            logger.info(f"  Total Samples: {stats['num_samples']}")
            logger.info(f"  Number of Classes: {len(stats['class_distribution'])}")
            logger.info(f"  Pixel Value Range: [{stats['pixel_statistics']['min']:.2f}, {stats['pixel_statistics']['max']:.2f}]")
            logger.info(f"  Mean Pixel Value: {stats['pixel_statistics']['mean']:.2f}")
            logger.info(f"  Pixel Value Std: {stats['pixel_statistics']['std']:.2f}")
        
        if results['memory_check']:
            logger.info(f"\nMemory Check: {results['memory_check']['message']}")