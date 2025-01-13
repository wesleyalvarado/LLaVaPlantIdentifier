import os
import sys
import logging
import torch
import gc
from transformers import AutoProcessor
from huggingface_hub import model_info

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables and clear GPU memory"""
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Print GPU info
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        logger.info(f"Available GPU memory: {free_memory / 1024**3:.2f} GB")
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")

def verify_llava_installation():
    """Verify LLaVA package installation and available classes"""
    try:
        import llava
        logger.info("Successfully imported llava package")
        
        # Import specific classes we need
        from llava.model.language_model.llava_mistral import (
            LlavaMistralForCausalLM,
            LlavaMistralConfig
        )
        logger.info("Successfully imported LlavaMistralForCausalLM and LlavaMistralConfig")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import required LLaVA modules: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error during LLaVA verification: {str(e)}")
        return False

def test_model_initialization():
    """Test model loading and initialization"""
    try:
        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        logger.info(f"Loading model: {model_name}")
        
        # First try loading processor
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info("Successfully loaded processor")
        
        # Load configuration
        logger.info("Loading configuration...")
        from llava.model.language_model.llava_mistral import LlavaMistralConfig
        config = LlavaMistralConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info(f"Configuration loaded. Model type: {config.model_type}")
        
        # Load model with LlavaMistralForCausalLM
        logger.info("Loading model using LlavaMistralForCausalLM...")
        from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
        model = LlavaMistralForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        logger.info("Successfully loaded model")
        
        # Print model info
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        
        return True
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_dataset_loading():
    """Test dataset initialization"""
    try:
        from data.dataset import MemoryEfficientPlantDataset
        
        # Create a small test dataset
        dataset = MemoryEfficientPlantDataset(
            split="train",
            sample_fraction=0.01,  # Use only 1% of data for testing
            image_size=224
        )
        
        logger.info(f"Successfully created dataset with {len(dataset)} samples")
        
        # Test loading a single item
        sample = dataset[0]
        logger.info("Successfully loaded a sample from dataset")
        logger.info(f"Sample keys: {list(sample.keys())}")
        
        return True
    except Exception as e:
        logger.error(f"Dataset testing failed: {str(e)}")
        return False

def main():
    """Main test function"""
    try:
        logger.info("Starting test sequence...")
        
        # Setup environment
        setup_environment()
        logger.info("Environment setup complete")
        
        # Verify LLaVA installation
        if not verify_llava_installation():
            logger.error("LLaVA package verification failed")
            return False
        
        # Test dataset loading
        if not test_dataset_loading():
            logger.error("Dataset testing failed")
            return False
            
        # Test model initialization
        if not test_model_initialization():
            logger.error("Model initialization failed")
            return False
            
        logger.info("All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Testing failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nTest completion status: {'Success' if success else 'Failed'}")