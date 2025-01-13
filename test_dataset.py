# First, make sure we're in the right directory
import os
import sys
import logging

# Add the project root to Python path
project_root = "/content/LLaVaPlantIdentifier"
sys.path.append(project_root)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Now import from data.dataset
from data.dataset import MemoryEfficientPlantDataset

def test_dataset(split: str = "train", sample_fraction: float = 0.01):
    """Test the dataset functionality"""
    try:
        logger.info("Creating test dataset...")
        dataset = MemoryEfficientPlantDataset(
            split=split,
            sample_fraction=sample_fraction
        )
        
        # Test getting first item
        logger.info("Testing first item...")
        first_item = dataset[0]
        
        # Print info about the first item
        print("\nFirst item contents:")
        for key, value in first_item.items():
            if hasattr(value, 'shape'):
                print(f"{key}: shape {value.shape}")
            else:
                print(f"{key}: {value}")
                
        return True
        
    except Exception as e:
        logger.error(f"Dataset testing failed: {e}")
        logger.error(f"Current directory: {os.getcwd()}")
        return False

if __name__ == "__main__":
    success = test_dataset()
    print(f"\nTest {'succeeded' if success else 'failed'}")