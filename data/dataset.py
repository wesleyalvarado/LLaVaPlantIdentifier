# data/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import AutoTokenizer
from datasets import load_dataset
import logging
from typing import Optional, Dict, Any
import gc

# Setup logging
logger = logging.getLogger(__name__)

class MemoryEfficientPlantDataset(Dataset):
    """Memory-efficient dataset implementation for plant identification using LLaVA."""
    
    def __init__(
        self,
        split: str = "train",
        sample_fraction: float = 1.0,
        image_size: int = 336  # LLaVA expected size
    ):
        """
        Initialize the dataset.
        
        Args:
            split: Dataset split ('train' or 'test')
            sample_fraction: Fraction of dataset to use (0.0 to 1.0)
            image_size: Size of images (both height and width)
        """
        try:
            logger.info(f"Initializing {split} dataset...")
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
            
            # Load dataset first to get class names
            temp_dataset = load_dataset("nelorth/oxford-flowers", split=split)
            self.class_names = temp_dataset.features['label'].names
            logger.info(f"Loaded {len(self.class_names)} flower categories")
            
            # Setup image transforms
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Calculate dataset size
            num_samples = max(1, int(len(temp_dataset) * sample_fraction))
            
            # Load subset of dataset
            self.dataset = load_dataset(
                "nelorth/oxford-flowers",
                split=f"{split}[:{num_samples}]"
            )
            
            # Process and store samples
            self.processed_data = []
            self._process_samples()
            
            logger.info(f"Successfully loaded {len(self.processed_data)} samples")
            
        except Exception as e:
            logger.error(f"Dataset initialization failed: {e}")
            raise

    def _process_samples(self):
        """Process all samples and store results."""
        for idx in range(len(self.dataset)):
            try:
                processed = self._process_single_sample(idx)
                if processed is not None:
                    self.processed_data.append(processed)
                
                # Periodic garbage collection
                if idx % 100 == 0:
                    gc.collect()
                    if idx > 0:
                        logger.info(f"Processed {idx} samples...")
                    
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue

    def _process_single_sample(self, idx: int) -> Optional[Dict[str, Any]]:
        """Process a single dataset sample."""
        try:
            item = self.dataset[idx]
            
            # Transform image
            image_tensor = self.transform(item['image'])
            
            # Get flower name
            label = item['label']
            class_name = self.class_names[label]
            
            # Create text descriptions
            prompt = f"What type of flower is shown in this image? Please identify the flower species."
            target = f"This image shows a {class_name}."
            
            # Tokenize text
            encoded = self.tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=64,
                return_tensors='pt'
            )
            
            return {
                'pixel_values': image_tensor,
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'label': torch.tensor(label),
                'class_name': class_name,
                'prompt': prompt,
                'target': target
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            return None

    def __len__(self) -> int:
        """Return the number of processed samples."""
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a processed sample."""
        return self.processed_data[idx]

def create_dataloaders(
    train_fraction: float = 0.8,
    batch_size: int = 1,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_fraction: Fraction of data to use for training
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    try:
        # Create datasets
        train_dataset = MemoryEfficientPlantDataset(
            split="train",
            sample_fraction=train_fraction
        )
        
        val_dataset = MemoryEfficientPlantDataset(
            split="test",
            sample_fraction=0.2  # Use 20% of test set for validation
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        
        logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise

def test_dataset():
    """Test the dataset implementation."""
    try:
        # Test dataset creation
        print("\nTesting dataset creation...")
        dataset = MemoryEfficientPlantDataset(
            split="train",
            sample_fraction=0.01  # Use small fraction for testing
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Print some class names
        print("\nSample flower categories:")
        for i in range(min(5, len(dataset.class_names))):
            print(f"Class {i}: {dataset.class_names[i]}")
        
        # Test single item
        item = dataset[0]
        print("\nSample item details:")
        print(f"Class name: {item['class_name']}")
        print(f"Prompt: {item['prompt']}")
        print(f"Target: {item['target']}")
        
        print("\nTensor shapes:")
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                print(f"- {k}: {v.shape}")
        
        # Test dataloader
        print("\nTesting dataloader creation...")
        train_loader, val_loader = create_dataloaders(
            train_fraction=0.01,
            batch_size=2
        )
        
        print("\nDataloader sizes:")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_dataset()